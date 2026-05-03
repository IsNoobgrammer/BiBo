"""Sliding window attention"""
import math
import torch
import torch.nn as nn
from typing import Optional
from .ssmax import apply_ssmax_query_scaling

__all__ = ['eager_sliding_window_attention']


def eager_sliding_window_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sliding_window: int,
    head_dim: int,
    attention_dropout: float,
    training: bool,
    use_ssmax: bool,
    ssmax_scale: Optional[torch.nn.Parameter] = None,
    stride: Optional[int] = None,
) -> torch.Tensor:
    """
    Causal sliding-window attention.
    
    Each query sees at most `sliding_window` recent keys.
    
    Args:
        query_states: (batch, num_heads, q_len, head_dim)
        key_states: (batch, num_heads, kv_len, head_dim)
        value_states: (batch, num_heads, kv_len, head_dim)
        attention_mask: Optional mask
        sliding_window: Window size
        head_dim: Head dimension
        attention_dropout: Dropout prob
        training: Training mode
        use_ssmax: Apply SSMax scaling
        ssmax_scale: SSMax scale param
        stride: Reserved for strided window (future)
    
    Returns:
        Attention output
    """
    del stride  # Reserved for future

    kv_len = key_states.shape[-2]
    q_len = query_states.shape[-2]
    if use_ssmax:
        query_states = apply_ssmax_query_scaling(query_states, kv_len, ssmax_scale)
        
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    query_positions = torch.arange(kv_len - q_len, kv_len, device=query_states.device)
    key_positions = torch.arange(kv_len, device=query_states.device)
    window_mask = key_positions.unsqueeze(0) < (query_positions.unsqueeze(1) - sliding_window + 1)
    attn_weights = attn_weights.masked_fill(window_mask.view(1, 1, q_len, kv_len), torch.finfo(attn_weights.dtype).min)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=training)
    return torch.matmul(attn_weights, value_states)
