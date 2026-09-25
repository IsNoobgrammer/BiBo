"""Full (global) attention.

Training and single-token decode take the SDPA is_causal fast path (the backend SKIPS the
upper triangle; AGENTS.md decision 9b) — no mask is ever built there. An explicit additive
mask exists only where is_causal is WRONG or insufficient: padding, cached multi-token
prefill (is_causal is TOP-LEFT aligned — wrong with a cache; F9), or output_attentions
(eager core, the only path that returns weights).
"""
import torch.nn.functional as F
from .utils import repeat_kv, causal_band_mask, padding_bias, eager_attention_forward
from .swa import swa_attention

__all__ = ['full_attention']

# Training hot path through compiled FlexAttention (a causal band as wide as the sequence) instead
# of SDPA flash. Flash's backward accumulates dQ with atomics, so it is the one op that made the
# board's gradients differ run to run; flex computes dQ in its own pass and is bitwise repeatable.
# Measured at B64 Hq4 Hkv2 S1024 D128: fwd+bwd 2.60 ms vs flash 2.10 (flash's own deterministic
# mode: 3.72), i.e. ~+8 ms per board step. False = SDPA as before.
GLOBAL_FLEX = True


def full_attention(query, key, value, *, num_key_value_groups, scaling,
                   padding_mask=None, dropout=0.0, training=False, output_attentions=False):
    """Global-attention forward. `query` (B,H,q,d); `key`/`value` GROUPED (B,H_kv,kv,d);
    `padding_mask` 2D (B,K_total) 1=real/0=pad or None.
    Returns (attn_output (B,H,q,d), probs or None — weights only with output_attentions)."""
    q_len, kv_len = query.shape[-2], key.shape[-2]
    dropout_p = dropout if training else 0.0
    need_mask = (output_attentions or padding_mask is not None
                 or (q_len > 1 and kv_len > q_len))
    if not need_mask and GLOBAL_FLEX and training and q_len == kv_len and query.is_cuda:
        return swa_attention(query, key, value, sliding_window=kv_len,
                             num_key_value_groups=num_key_value_groups, scaling=scaling,
                             dropout=dropout, training=training)
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
            dropout=dropout_p, training=training)

    attn_output = F.scaled_dot_product_attention(
        query, repeat_kv(key, num_key_value_groups), repeat_kv(value, num_key_value_groups),
        attn_mask=attn_mask, is_causal=False, dropout_p=dropout_p, scale=scaling)
    return attn_output, None
