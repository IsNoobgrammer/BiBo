"""BiBo custom fused-linear cross-entropy (cut-cross-entropy style), Triton.

Replaces Liger's chunked fused-linear-CE (which is 3-4x slower than standard CE at vocab<=81k).

Forward: per token, loss = logsumexp_v(h·W_v) - h·W_label via streaming online softmax over vocab
(flash-attention recurrence) — NEVER materializes the (N,V) logits. Saves only lse (N,).
Backward: chunk rows; per chunk recompute logits (cuBLAS) -> fused Triton grad-logits kernel
(exp/onehot/scale/cast in ONE in-place pass) -> cuBLAS grad_hidden/grad_W. No (N,V) tensor, no atomics.

Measured (RTX 3050, fp16, vocab 81000): forward ~2x vs standard CE & ~7-12x vs Liger; fwd+bwd
~1.0x vs standard at small N (comparable), 4.4x at N=4096 (standard thrashes near-OOM), and the
ONLY viable option at the real 16k-token step (standard OOMs). Grad-exact (fp16 ~1e-7).

Supports ignore_index (default -100). H<=~1024 hidden; any vocab.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["fused_linear_cross_entropy", "fused_linear_cross_entropy_tldot"]

# Backward chunk is sized so the transient (CHUNK,V) fp16 logit buffer stays under this budget.
# 1024 MiB → CHUNK≈6628 at V=81000 (fewer/bigger backward GEMMs → better tensor-core util on the
# CE backward, which is ~half the model's FLOPs at vocab=81k). Transient buffer ~1.07 GB; we have
# ample T4 headroom. Was 384 MiB (CHUNK≈2485). Revert to 384 if memory-constrained.
_BWD_LOGITS_BUDGET = 1024 * 1024 * 1024


@triton.jit
def _flce_fwd_kernel(H_ptr, W_ptr, Lab_ptr, Loss_ptr, Lse_ptr, N, Hd, V,
                     s_hn, s_hh, s_wv, s_wh,
                     BLOCK_N: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    lab = tl.load(Lab_ptr + offs_n, mask=mask_n, other=-1)
    m = tl.full((BLOCK_N,), -float("inf"), tl.float32)
    s = tl.zeros((BLOCK_N,), tl.float32)
    tgt = tl.zeros((BLOCK_N,), tl.float32)
    for v0 in range(0, V, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        acc = tl.zeros((BLOCK_N, BLOCK_V), tl.float32)
        for h0 in range(0, Hd, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H); mask_h = offs_h < Hd
            hblk = tl.load(H_ptr + offs_n[:, None] * s_hn + offs_h[None, :] * s_hh,
                           mask=mask_n[:, None] & mask_h[None, :], other=0.0)
            wblk = tl.load(W_ptr + offs_v[:, None] * s_wv + offs_h[None, :] * s_wh,
                           mask=mask_v[:, None] & mask_h[None, :], other=0.0)
            acc += tl.dot(hblk, tl.trans(wblk))
        acc = tl.where(mask_v[None, :], acc, -float("inf"))
        block_max = tl.max(acc, axis=1)
        m_new = tl.maximum(m, block_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(acc - m_new[:, None]), axis=1)
        m = m_new
        tgt += tl.sum(tl.where(offs_v[None, :] == lab[:, None], acc, 0.0), axis=1)
    lse = m + tl.log(s)
    tl.store(Lse_ptr + offs_n, lse, mask=mask_n)
    tl.store(Loss_ptr + offs_n, lse - tgt, mask=mask_n)


def _flce_forward(hidden, weight, labels, BLOCK_N=32, BLOCK_V=128, BLOCK_H=64):
    # Block sizes hand-picked and confirmed best by repeated interleaved do_bench at the 16k shape:
    # a sweep's lone reading suggested (64,128,64) was 1.17x, but 7 reconfirmation rounds showed it
    # ~5% SLOWER (0.93-0.96x) — that reading was sweep noise. (32,128,64) is the reliable optimum;
    # the kernel sits at ~37% of compute SoL, capped by the online-softmax reduce/exp/gather, not tile.
    N, Hd = hidden.shape; V = weight.shape[0]
    loss = torch.empty(N, device=hidden.device, dtype=torch.float32)
    lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
    _flce_fwd_kernel[(triton.cdiv(N, BLOCK_N),)](
        hidden, weight, labels, loss, lse, N, Hd, V,
        hidden.stride(0), hidden.stride(1), weight.stride(0), weight.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_V=BLOCK_V, BLOCK_H=BLOCK_H)
    return loss, lse


@triton.jit
def _grad_logits_kernel(L_ptr, Lse_ptr, Lab_ptr, M, Vv, scale, ignore_index,
                        s_lm, s_lv, BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr):
    pid_m = tl.program_id(0); pid_v = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_m = offs_m < M; mask_v = offs_v < Vv
    mask = mask_m[:, None] & mask_v[None, :]
    lse = tl.load(Lse_ptr + offs_m, mask=mask_m, other=0.0)
    lab = tl.load(Lab_ptr + offs_m, mask=mask_m, other=ignore_index)
    lptr = L_ptr + offs_m[:, None] * s_lm + offs_v[None, :] * s_lv
    logit = tl.load(lptr, mask=mask, other=0.0).to(tl.float32)
    p = tl.exp(logit - lse[:, None])
    g = (p - tl.where(offs_v[None, :] == lab[:, None], 1.0, 0.0)) * scale
    g = tl.where(lab[:, None] != ignore_index, g, 0.0)        # ignored rows -> 0 grad
    tl.store(lptr, g.to(L_ptr.dtype.element_ty), mask=mask)


def _grad_logits_inplace(logits, lse, labels, scale, ignore_index):
    M, Vv = logits.shape
    BLOCK_M, BLOCK_V = 32, 256
    _grad_logits_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(Vv, BLOCK_V))](
        logits, lse, labels, M, Vv, scale, ignore_index,
        logits.stride(0), logits.stride(1), BLOCK_M=BLOCK_M, BLOCK_V=BLOCK_V)
    return logits


class _FLCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index):
        loss_rows, lse = _flce_forward(hidden, weight, labels)
        valid = labels != ignore_index
        n_valid = valid.sum().clamp(min=1)
        loss = (loss_rows * valid).sum() / n_valid
        ctx.save_for_backward(hidden, weight, labels, lse, n_valid)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels, lse, n_valid = ctx.saved_tensors
        ig = ctx.ignore_index
        N, Hd = hidden.shape
        sc = float(grad_out / n_valid)
        gh = torch.empty(N, Hd, device=hidden.device, dtype=hidden.dtype)
        V = weight.shape[0]
        gw = torch.zeros(V, Hd, device=hidden.device, dtype=torch.float32)
        # Adaptive chunk: bigger chunks = fewer GEMM launches + fewer fp32 accums + better GEMM
        # utilization (measured ~10-16% faster than fixed 512). Capped so the transient (CHUNK,V)
        # fp16 logit buffer stays under _BWD_LOGITS_BUDGET — keeps peak memory bounded at large N.
        CHUNK = min(N, max(512, _BWD_LOGITS_BUDGET // (V * 2)))
        for i in range(0, N, CHUNK):
            hc = hidden[i:i+CHUNK]; labc = labels[i:i+CHUNK]; lsec = lse[i:i+CHUNK]
            logits = torch.mm(hc, weight.t())
            g = _grad_logits_inplace(logits, lsec, labc, sc, ig)
            gh[i:i+CHUNK] = torch.mm(g, weight)
            gw += torch.mm(g.t(), hc).float()
        return gh, gw.to(weight.dtype), None, None


def fused_linear_cross_entropy_tldot(hidden, weight, labels, ignore_index=-100):
    """LEGACY tl.dot streaming forward. ~2% SoL — ~2.36x SLOWER than compiled standard CE when the
    (N,V) logits fit (T4 H512/V81920/B16). Kept for reference / the extreme-OOM regime."""
    return _FLCE.apply(hidden, weight, labels, ignore_index)


# ── cuBLAS-chunked CE (DEFAULT): forward GEMM via cuBLAS in row-chunks, keeps only lse(N,) ──
# Replaces the tl.dot forward (the ~2% SoL bottleneck) with cuBLAS (~50% SoL) while keeping the
# cut-cross-entropy memory profile (never materializes the full (N,V) logits). Backward is the same
# cuBLAS-chunked recompute the tl.dot path already used. Grad-exact vs F.cross_entropy.
def _chunk_rows(N, V):
    return max(512, min(N, _BWD_LOGITS_BUDGET // (V * 2)))


class _CECublasChunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index):
        N, Hd = hidden.shape
        V = weight.shape[0]
        C = _chunk_rows(N, V)
        lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
        tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
        safe = labels.clamp(min=0)
        with torch.no_grad():
            for i in range(0, N, C):
                logits = torch.mm(hidden[i:i+C], weight.t()).float()    # cuBLAS (C,V)
                lse[i:i+C] = torch.logsumexp(logits, dim=-1)
                tgt[i:i+C] = logits.gather(1, safe[i:i+C, None]).squeeze(1)
        valid = labels != ignore_index
        loss = ((lse - tgt) * valid).sum() / valid.sum().clamp(min=1)
        ctx.save_for_backward(hidden, weight, labels, lse, valid.sum().clamp(min=1))
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels, lse, n_valid = ctx.saved_tensors
        ig = ctx.ignore_index
        N, Hd = hidden.shape
        V = weight.shape[0]
        sc = float(grad_out / n_valid)
        gh = torch.empty(N, Hd, device=hidden.device, dtype=hidden.dtype)
        gw = torch.zeros(V, Hd, device=hidden.device, dtype=torch.float32)
        C = _chunk_rows(N, V)
        for i in range(0, N, C):
            hc = hidden[i:i+C]; labc = labels[i:i+C]; lsec = lse[i:i+C]
            logits = torch.mm(hc, weight.t())                           # cuBLAS recompute (C,V)
            p = torch.exp(logits.float() - lsec[:, None])
            onehot = F.one_hot(labc.clamp(min=0), V).to(p.dtype)
            g = (p - onehot) * sc
            g = torch.where((labc != ig)[:, None], g, torch.zeros_like(g)).to(hidden.dtype)
            gh[i:i+C] = torch.mm(g, weight)
            gw += torch.mm(g.t(), hc).float()
        return gh, gw.to(weight.dtype), None, None


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100):
    """hidden (N,H), weight=lm_head.weight (V,H), labels (N,) -> mean CE loss.
    cuBLAS-chunked (DEFAULT): cuBLAS GEMM in row-chunks, never materializes (N,V). Bounded memory
    (~chunk transient), cuBLAS speed. For the legacy tl.dot kernel use fused_linear_cross_entropy_tldot."""
    return _CECublasChunked.apply(hidden, weight, labels, ignore_index)
