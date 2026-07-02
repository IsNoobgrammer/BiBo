"""Full (global) attention.

Training and single-token decode take the SDPA is_causal fast path (the backend SKIPS the
upper triangle; AGENTS.md decision 9b) — no mask is ever built there. An explicit additive
mask exists only where is_causal is WRONG or insufficient: padding, cached multi-token
prefill (is_causal is TOP-LEFT aligned — wrong with a cache; F9), a G3 sink, or
output_attentions (eager core, the only path that returns weights). The per-head sink rides
SDPA as one extra mask column equal to the sink bias plus a zero K/V column
(q·0 + β_h = β_h in the softmax denominator; the zero V drops it from the output).
"""
import torch
import torch.nn.functional as F
from .utils import repeat_kv, causal_band_mask, padding_bias, eager_attention_forward

__all__ = ['full_attention']


def full_attention(query, key, value, sinks, *, num_key_value_groups, scaling,
                   padding_mask=None, dropout=0.0, training=False, output_attentions=False):
    """Global-attention forward. `query` (B,H,q,d); `key`/`value` GROUPED (B,H_kv,kv,d); `sinks`
    per-head bias (H,) or None; `padding_mask` 2D (B,K_total) 1=real/0=pad or None.
    Returns (attn_output (B,H,q,d), probs or None — real weights only with output_attentions)."""
    q_len, kv_len = query.shape[-2], key.shape[-2]
    dropout_p = dropout if training else 0.0
    need_mask = (output_attentions or sinks is not None or padding_mask is not None
                 or (q_len > 1 and kv_len > q_len))
    if not need_mask:
        # Training (q_len==kv_len) and single-token decode — the hot path.
        attn_output = F.scaled_dot_product_attention(
            query, repeat_kv(key, num_key_value_groups), repeat_kv(value, num_key_value_groups),
            attn_mask=None, is_causal=q_len > 1, dropout_p=dropout_p, scale=scaling)
        return attn_output, None

    attn_mask = causal_band_mask(q_len, kv_len, None, query.dtype, query.device)
    if padding_mask is not None:
        attn_mask = attn_mask + padding_bias(padding_mask, kv_len, query.dtype)  # (B,1,q,kv)
    if output_attentions:
        return eager_attention_forward(
            query, key, value, attn_mask, scaling, num_key_value_groups,
            dropout=dropout_p, training=training, sinks=sinks)

    k_rep = repeat_kv(key, num_key_value_groups)
    v_rep = repeat_kv(value, num_key_value_groups)
    if sinks is not None:
        bsz, num_heads = query.shape[0], query.shape[1]
        sink_col = sinks.to(query.dtype).reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
        attn_mask = torch.cat(
            [attn_mask.expand(bsz, num_heads, q_len, kv_len), sink_col], dim=-1)
        zero_col = k_rep.new_zeros(k_rep.shape[0], k_rep.shape[1], 1, k_rep.shape[3])
        k_rep = torch.cat([k_rep, zero_col], dim=-2)
        v_rep = torch.cat([v_rep, zero_col], dim=-2)
    attn_output = F.scaled_dot_product_attention(
        query, k_rep, v_rep, attn_mask=attn_mask, is_causal=False,
        dropout_p=dropout_p, scale=scaling)
    return attn_output, None
