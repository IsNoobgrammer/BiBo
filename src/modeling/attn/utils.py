"""Attention utilities shared by the SWA and full-attention modules"""
import torch
import torch.nn.functional as F

__all__ = ['repeat_kv', 'causal_band_mask', 'padding_bias', 'eager_attention_forward']


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for GQA.
    
    Args:
        hidden_states: (batch, num_key_value_heads, seq_len, head_dim)
        n_rep: num repetitions (num_query_heads // num_key_value_heads)
    
    Returns:
        (batch, num_query_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def causal_band_mask(q_len, kv_len, sliding_window, dtype, device):
    """Additive attention mask (0 / -inf), shape (1, 1, q_len, kv_len) — broadcasts over (B,H).
    Causal, plus a sliding-window band when sliding_window is not None. Bottom-right aligned via
    absolute query positions, so it's correct for packed training (q_len==kv_len), cached decode
    (q_len==1, kv_len large), and a window-cropped KV (keys are always the LAST kv_len positions)."""
    i = torch.arange(q_len, device=device).unsqueeze(1) + (kv_len - q_len)   # abs query pos
    j = torch.arange(kv_len, device=device).unsqueeze(0)
    allow = j <= i                                                            # causal
    if sliding_window is not None:
        allow = allow & ((i - j) < sliding_window)                            # window band
    mask = torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), float("-inf"), dtype=dtype, device=device))
    return mask[None, None]                                                    # (1,1,q,kv)


def padding_bias(padding_mask, kv_len, dtype):
    """Additive bias (B,1,1,kv_len) from a 2D (B, K_total) padding mask (1=real, 0=pad), sliced
    bottom-right to the returned KV window (SWA cache layers crop from the front). finfo.min
    (not -inf) so a fully-padded row softmaxes to uniform garbage instead of NaN (pad rows are
    ignored downstream)."""
    return (1.0 - padding_mask[:, None, None, -kv_len:].to(dtype)) * torch.finfo(dtype).min


def eager_attention_forward(query, key, value, attention_mask, scaling, num_key_value_groups,
                            dropout=0.0, training=False, sinks=None):
    """Eager attention core — faithful to MiMo-V2.5 `eager_attention_forward` (GPT-OSS-style sink).

    Two deviations from MiMo: the final `.transpose(1,2)` is deferred to the caller (BiBo runs
    XSA on the (B,H,q,d) output first, then the shared tail transposes), and the pre-softmax
    row-max subtraction is dropped (see inline note). `key`/`value` are GROUPED (GQA) and
    repeated here. `attention_mask` is additive (0 / -inf), broadcast over (B,H). `sinks` is a
    per-head bias (or None): when set, one value-less per-head sink column is concatenated AFTER
    the mask, included in the softmax denominator, then dropped before the V matmul.
    Returns (attn_output (B,H,q,d), probs (real weights, sink dropped))."""
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Bottom-right slice: the returned KV window is always the LAST kv_len positions
        # (SWA cache layers crop from the front). No-op for an exact-size mask.
        causal_mask = attention_mask[:, :, :, -key_states.shape[-2]:]
        attn_weights = attn_weights + causal_mask
    if sinks is not None:
        s = sinks.reshape(1, -1, 1, 1).to(attn_weights.dtype).expand(
            query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, s], dim=-1)
    # ponytail: no pre-softmax row-max subtraction — F.softmax upcasts to fp32 and is
    # shift-stable on its own (rows are never all -inf: the causal diagonal / sink col is finite)
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if sinks is not None:
        probs = probs[..., :-1]
    probs = F.dropout(probs, p=dropout, training=training)
    attn_output = torch.matmul(probs, value_states)
    return attn_output, probs
