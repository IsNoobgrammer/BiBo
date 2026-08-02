"""XSA — Exclusive Self Attention (arxiv 2603.09078)"""
import torch
import torch.nn.functional as F
from .utils import repeat_kv

__all__ = ['apply_xsa']


def apply_xsa(attn_output: torch.Tensor, value_states: torch.Tensor,
              enable_gqa: bool = True, alpha: torch.Tensor = None) -> torch.Tensor:
    """
    Remove the component of the attended output along each value direction.

    Y <- Y - a (Y . Vn) Vn,  where Vn = normalize(V) and a = tanh(alpha) per head

    `alpha` is the per-head (H,) LOGIT, so the applied strength is tanh(alpha): 1 = full
    rejection (the original hard-coded behaviour), 0 = XSA off, negative = AMPLIFY the
    self-component. None keeps a = 1. Bounded by construction, which is the point -- an
    unbounded strength can blow the residual up, and |a| > 1 over-rejects past orthogonal.

    GQA handling (H_kv < H):
      - enable_gqa=True (default): broadcast V across the query group WITHOUT
        materializing a repeat_kv copy (SDPA-style). The (B,H,S,D) `V_rep` and the
        full-size `Vn` are never written — the dot/rejection broadcast over the group.
      - enable_gqa=False: legacy path — repeat_kv materializes V to full heads first.

    Args:
        attn_output: Attention output (B, H, S, D)
        value_states: Value tensor (B, H_kv, S, D), H_kv may be < H (GQA)
        enable_gqa: broadcast V in-place instead of materializing repeat_kv
        alpha: optional per-head (H,) rejection-strength logit; applied as tanh(alpha)

    Returns:
        XSA-corrected attention output (B, H, S, D)
    """
    B, n_heads, S, D = attn_output.shape           # S = q_len (query positions)
    n_kv = value_states.shape[1]
    g = n_heads // n_kv
    if not enable_gqa and g != 1:
        value_states = repeat_kv(value_states, g)
        n_kv, g = n_heads, 1
    # Align V to the QUERY positions: each query rejects along ITS OWN value. value_states holds
    # all kv_len positions; the queries are the last q_len of them (packed training: q_len==kv_len,
    # a no-op; cached decode: q_len==1 -> the newest token's value). Fixes the q_len!=kv_len case.
    v_aligned = value_states[:, :, -S:, :]
    Yg = attn_output.view(B, n_kv, g, S, D)
    Vn = F.normalize(v_aligned, dim=-1).unsqueeze(2)          # (B, n_kv, 1, S, D)
    coeff = (Yg * Vn).sum(dim=-1, keepdim=True)
    if alpha is not None:
        # head h == kv * g + j, which is exactly how Yg was reshaped -- and how the fused kernel
        # indexes A. A .view() here and a different ordering there would silently permute the
        # per-head strengths, so the two must stay written the same way.
        coeff = coeff * torch.tanh(alpha.float()).to(coeff.dtype).view(1, n_kv, g, 1, 1)
    return (Yg - coeff * Vn).reshape(B, n_heads, S, D)
