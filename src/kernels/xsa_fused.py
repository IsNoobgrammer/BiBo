"""Fused XSA (Exclusive Self Attention) Triton kernel.

z_i = y_i - (y_i . v_i / ||v_i||^2) v_i   (vector rejection of the attn output from its self-value)

Fuses the whole eager apply_xsa (normalize + dot + scale + subtract + repeat_kv) into ONE
forward and ONE backward kernel:
  - GQA handled by BROADCASTING V across the query group IN-KERNEL (like SDPA enable_gqa) — the
    (B,H,S,D) repeat_kv copy and the full-size normalized-V are NEVER materialized.
  - fp32 accumulation in-register (free) for the norm/dot reductions.
  - grad_Y = reject(grad_z, v_hat)  [same symmetric-idempotent operator as forward]
  - grad_V analytic, accumulated over the group (one V load reused across `group` query heads).

Grid = B*Hkv*S programs; each handles one kv-row and loops its `group` query heads.
Grad-exact vs eager apply_xsa (verified atol 1e-3 fp16). Wrap: fused_xsa(Y, V).
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_xsa", "patch_xsa_with_triton", "unpatch_xsa"]

_ORIG_APPLY_XSA = None


@triton.jit
def _xsa_fwd_kernel(Y, V, Z, S, D, H, Hkv, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)                 # over B*Hkv*S
    s = pid % S
    t = pid // S
    kv = t % Hkv
    b = t // Hkv
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    v_base = (b * Hkv + kv) * S * D + s * D
    v = tl.load(V + v_base + offs, mask=mask, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=0)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)

    for j in range(GROUP):
        h = kv * GROUP + j
        y_base = (b * H + h) * S * D + s * D
        y = tl.load(Y + y_base + offs, mask=mask, other=0.0).to(tl.float32)
        coeff = tl.sum(y * v, axis=0) * inv
        z = y - coeff * v
        tl.store(Z + y_base + offs, z.to(Z.dtype.element_ty), mask=mask)


@triton.jit
def _xsa_bwd_kernel(GZ, Y, V, GY, GV, S, D, H, Hkv, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    s = pid % S
    t = pid // S
    kv = t % Hkv
    b = t // Hkv
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    v_base = (b * Hkv + kv) * S * D + s * D
    v = tl.load(V + v_base + offs, mask=mask, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=0)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)

    gv_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in range(GROUP):
        h = kv * GROUP + j
        base = (b * H + h) * S * D + s * D
        y = tl.load(Y + base + offs, mask=mask, other=0.0).to(tl.float32)
        gz = tl.load(GZ + base + offs, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(y * v, axis=0)
        gzv = tl.sum(gz * v, axis=0)
        coeff = dot * inv
        # grad_Y = gz - (gz.v)/n2 * v   (rejection of gz)
        gy = gz - gzv * inv * v
        tl.store(GY + base + offs, gy.to(GY.dtype.element_ty), mask=mask)
        # grad_V row contribution: -gzv*inv*y + 2*dot*gzv*inv^2*v - coeff*gz
        gv_acc += -gzv * inv * y + 2.0 * dot * gzv * inv * inv * v - coeff * gz

    tl.store(GV + v_base + offs, gv_acc.to(GV.dtype.element_ty), mask=mask)


class _FusedXSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y, V):
        Y = Y.contiguous()
        V = V.contiguous()
        B, H, S, D = Y.shape
        Hkv = V.shape[1]
        group = H // Hkv
        Z = torch.empty_like(Y)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * Hkv * S,)
        _xsa_fwd_kernel[grid](Y, V, Z, S, D, H, Hkv, GROUP=group, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(Y, V)
        ctx.shape = (B, H, S, D, Hkv, group, BLOCK_D)
        return Z

    @staticmethod
    def backward(ctx, gZ):
        Y, V = ctx.saved_tensors
        B, H, S, D, Hkv, group, BLOCK_D = ctx.shape
        gZ = gZ.contiguous()
        GY = torch.empty_like(Y)
        GV = torch.empty_like(V)
        grid = (B * Hkv * S,)
        _xsa_bwd_kernel[grid](gZ, Y, V, GY, GV, S, D, H, Hkv, GROUP=group, BLOCK_D=BLOCK_D)
        return GY, GV


def fused_xsa(attn_output: torch.Tensor, value_states: torch.Tensor,
              enable_gqa: bool = True) -> torch.Tensor:
    """Fused XSA. enable_gqa is implied (V is always broadcast in-kernel, never materialized)."""
    return _FusedXSA.apply(attn_output, value_states)


def patch_xsa_with_triton():
    """Monkey-patch BiBoAttention's apply_xsa to the fused Triton kernel."""
    global _ORIG_APPLY_XSA
    import src.modeling.attn.base as base
    if _ORIG_APPLY_XSA is None:
        _ORIG_APPLY_XSA = base.apply_xsa
    base.apply_xsa = lambda Y, V, enable_gqa=True: fused_xsa(Y, V, enable_gqa)


def unpatch_xsa():
    """Restore the eager apply_xsa."""
    global _ORIG_APPLY_XSA
    if _ORIG_APPLY_XSA is not None:
        import src.modeling.attn.base as base
        base.apply_xsa = _ORIG_APPLY_XSA
        _ORIG_APPLY_XSA = None
