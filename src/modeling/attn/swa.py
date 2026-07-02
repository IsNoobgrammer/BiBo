"""Sliding-window attention (SWA) — eager only, by design.

The whole SWA flavor lives here: band-mask construction, optional padding mask, and the
per-head attention sink (unscaled — SSMax is OFF on SWA layers, docs/attention_layers.md §3).
Deliberately NOT routed through SDPA/mem-efficient: the intended fast path is a dedicated
sink-aware banded kernel; until that lands, SWA runs the reference eager core so the kernel
has an exact numerics target. See docs/attention_layers.md.
"""
import torch
from .utils import causal_band_mask, padding_bias, eager_attention_forward

__all__ = ['swa_attention']


def swa_attention(query, key, value, sinks, *, sliding_window, num_key_value_groups, scaling,
                  padding_mask=None, dropout=0.0, training=False):
    """SWA forward. `query` (B,H,q,d); `key`/`value` GROUPED (B,H_kv,kv,d); `sinks` per-head
    bias (H,) or None; `padding_mask` 2D (B,K_total) 1=real/0=pad or None.
    Returns (attn_output (B,H,q,d), probs (B,H,q,kv) — real weights, sink mass dropped)."""
    q_len, kv_len = query.shape[-2], key.shape[-2]
    # Band mask is built even at q_len==1 decode so the window is enforced when the cache
    # holds more than sliding_window keys (uncropped/external caches).
    attn_mask = causal_band_mask(q_len, kv_len, sliding_window, query.dtype, query.device)
    if padding_mask is not None:
        attn_mask = attn_mask + padding_bias(padding_mask, kv_len, query.dtype)
    return eager_attention_forward(
        query, key, value, attn_mask, scaling, num_key_value_groups,
        dropout=dropout, training=training, sinks=sinks)
