"""Fused causal-conv MoE router (the T4 'cudnn' winner — 1.13-1.15x fwd+bwd vs torch.compile).

Drop-in for BiBoMoERouter's conv path (gate_type='sigmoid', router_activation='none'). Combines the
two T4-measured edges over compiled eager:

  forward : cuDNN conv (padding=K-1, no F.pad copy) + ONE fused Triton epilogue kernel that does
            sigmoid + selection-bias + top-k argmax + UNBIASED gather in-register — kills the ~295us
            native torch.topk + gather that compiled must keep as a library call.
  backward: cuDNN convolution_backward called DIRECTLY on a saved-once contiguous (B,H,S) input
            (autograd otherwise copies x->contiguous twice + casts/transposes) + a fused
            epilogue-backward kernel (sigmoid' + gather^T scatter in one).

Math is bit-identical to the eager MiMo/DeepSeek-V3 sigmoid gate (verified vs BiBoMoERouter: idx 1.0,
weights rel 4e-8, loss abs 8e-6, grads tight, bias-update exact). norm_topk_prob (÷Σ) and
routed_scaling (×c) stay in eager autograd (a tiny k-wide reduction).

Causal conv:  out[b,s,e] = Σ_k Σ_h x[b, s-(K-1)+k, h] · W[e,h,k]   (src row <0 -> 0)
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_conv_router", "FusedConvRouterCuDNN"]


@triton.jit
def _router_epilogue_fwd_kernel(Logit_ptr, Bias_ptr, Idx_ptr, W_ptr, N, sln, sle,
                                HAS_BIAS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    """sigmoid + (selection)bias + top-k argmax + unbiased gather in ONE pass. sel=scores+bias picks;
    weights=scores (UNBIASED) at the picks. BLOCK_N rows/program (vectorized argmax over E)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    mask_n = offs_n < N
    mask_e = offs_e < E
    logit = tl.load(Logit_ptr + offs_n[:, None] * sln + offs_e[None, :] * sle,
                    mask=mask_n[:, None] & mask_e[None, :], other=0.0).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp(-logit))
    sel = scores
    if HAS_BIAS:
        b = tl.load(Bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        sel = sel + b[None, :]
    sel = tl.where(mask_e[None, :], sel, -1e30)
    for k in tl.static_range(TOPK):
        am = tl.argmax(sel, axis=1)
        onehot = offs_e[None, :] == am[:, None]
        w_k = tl.sum(tl.where(onehot, scores, 0.0), axis=1)
        tl.store(Idx_ptr + offs_n * TOPK + k, am.to(tl.int64), mask=mask_n)
        tl.store(W_ptr + offs_n * TOPK + k, w_k, mask=mask_n)
        sel = tl.where(onehot, -1e30, sel)


@triton.jit
def _router_epilogue_bwd_kernel(Logit_ptr, Idx_ptr, Gw_ptr, Gout_ptr, N,
                                sln, sle, sin, sik, sgn, sgk, son, soe,
                                E: tl.constexpr, TOPK: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    """grad_logits = scatter(grad_w -> picked idx slots) * sigmoid'(logit). One kernel."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    mask_n = offs_n < N
    mask_e = offs_e < E
    logit = tl.load(Logit_ptr + offs_n[:, None] * sln + offs_e[None, :] * sle,
                    mask=mask_n[:, None] & mask_e[None, :], other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-logit))
    sp = s * (1.0 - s)
    gscore = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        ik = tl.load(Idx_ptr + offs_n * sin + k * sik, mask=mask_n, other=0).to(tl.int32)
        gk = tl.load(Gw_ptr + offs_n * sgn + k * sgk, mask=mask_n, other=0.0)
        gscore += tl.where(offs_e[None, :] == ik[:, None], gk[:, None], 0.0)
    gout = gscore * sp
    tl.store(Gout_ptr + offs_n[:, None] * son + offs_e[None, :] * soe,
             gout.to(Gout_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_e[None, :])


def _epilogue_fwd(logits, bias, top_k):
    N, E = logits.shape
    idx = torch.empty(N, top_k, device=logits.device, dtype=torch.long)
    w = torch.empty(N, top_k, device=logits.device, dtype=torch.float32)
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _router_epilogue_fwd_kernel[grid](
        logits, bias if bias is not None else logits, idx, w, N,
        logits.stride(0), logits.stride(1),
        HAS_BIAS=bias is not None, E=E, TOPK=top_k,
        BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
    return idx, w


class FusedConvRouterCuDNN(torch.autograd.Function):
    """(B,S,H) hidden + (E,H,K) conv weight + (E,) selection bias -> (idx (B*S,k) long, weights
    (B*S,k) fp32 UNBIASED). cuDNN conv fwd + fused epilogue; cuDNN convolution_backward + fused
    epilogue-bwd. grad flows to hidden and weight (bias is selection-only, no grad)."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k):
        import torch.nn.functional as F
        B, S, H = x.shape
        E, _, K = weight.shape
        xc = x.transpose(1, 2).contiguous()                     # (B,H,S) once, reused in bwd
        conv = F.conv1d(xc, weight, padding=K - 1)[..., :S]     # (B,E,S) causal, no F.pad copy
        logits = conv.transpose(1, 2).reshape(B * S, E)         # (B*S,E)
        idx, weights = _epilogue_fwd(logits, bias, top_k)
        ctx.save_for_backward(xc, weight, logits, idx)
        ctx.K, ctx.S, ctx.E = K, S, E
        ctx.mark_non_differentiable(idx)
        return idx, weights

    @staticmethod
    def backward(ctx, grad_idx, grad_weights):
        import torch.nn.functional as F
        xc, weight, logits, idx = ctx.saved_tensors
        K, S, E = ctx.K, ctx.S, ctx.E
        N, top_k = idx.shape
        grad_logits = torch.empty(N, E, device=xc.device, dtype=xc.dtype)
        gw = grad_weights.contiguous()
        BLOCK_N = 128
        _router_epilogue_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            logits, idx, gw, grad_logits, N,
            logits.stride(0), logits.stride(1), idx.stride(0), idx.stride(1),
            gw.stride(0), gw.stride(1), grad_logits.stride(0), grad_logits.stride(1),
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
        B = xc.shape[0]
        grad_full = F.pad(grad_logits.view(B, S, E).transpose(1, 2), (0, K - 1))   # (B,E,S+K-1)
        grad_xc, grad_w = torch.ops.aten.convolution_backward(
            grad_full, xc, weight, [0], [1], [K - 1], [1], False, [0], 1, [True, True, False])[:2]
        return grad_xc.transpose(1, 2), grad_w, None, None   # (B,H,S)->(B,S,H); bias/top_k no grad


def fused_conv_router(hidden, weight, bias, top_k, norm_topk_prob=True, routed_scaling_factor=1.0):
    """Conv MoE router fast path. hidden (B,S,H), weight (E,H,K) nn.Conv1d weight, bias (E,) fp32/None.
    Returns (idx (B,S,k) long, norm_weights (B,S,k) fp32) — identical to BiBoMoERouter's conv output.
    Weight is cast to hidden's dtype (autocast keeps params fp32; grad chains back to the fp32 param)."""
    B, S, _ = hidden.shape
    w = weight.to(hidden.dtype) if weight.dtype != hidden.dtype else weight
    idx, wt = FusedConvRouterCuDNN.apply(hidden, w, bias, top_k)
    if top_k > 1 and norm_topk_prob:
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
    wt = wt * routed_scaling_factor
    return idx.view(B, S, top_k).long(), wt.view(B, S, top_k).float()
