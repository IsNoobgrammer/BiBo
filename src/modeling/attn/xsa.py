"""XSA — Exclusive Self Attention (arxiv 2603.09078)"""
import torch
import torch.nn.functional as F
from .utils import repeat_kv

__all__ = ['apply_xsa']


def apply_xsa(attn_output: torch.Tensor, value_states: torch.Tensor,
              enable_gqa: bool = True) -> torch.Tensor:
    """
    Remove the component of the attended output along each value direction.

    Y <- Y - (Y . Vn) Vn,  where Vn = normalize(V)

    GQA handling (H_kv < H):
      - enable_gqa=True (default): broadcast V across the query group WITHOUT
        materializing a repeat_kv copy (SDPA-style). The (B,H,S,D) `V_rep` and the
        full-size `Vn` are never written — the dot/rejection broadcast over the group.
      - enable_gqa=False: legacy path — repeat_kv materializes V to full heads first.

    Args:
        attn_output: Attention output (B, H, S, D)
        value_states: Value tensor (B, H_kv, S, D), H_kv may be < H (GQA)
        enable_gqa: broadcast V in-place instead of materializing repeat_kv

    Returns:
        XSA-corrected attention output (B, H, S, D)
    """
    B, n_heads, S, D = attn_output.shape
    n_kv = value_states.shape[1]
    g = n_heads // n_kv
    if not enable_gqa and g != 1:
        value_states = repeat_kv(value_states, g)
        n_kv, g = n_heads, 1
    Yg = attn_output.view(B, n_kv, g, S, D)
    Vn = F.normalize(value_states, dim=-1).unsqueeze(2)        # (B, n_kv, 1, S, D)
    return (Yg - (Yg * Vn).sum(dim=-1, keepdim=True) * Vn).reshape(B, n_heads, S, D)
