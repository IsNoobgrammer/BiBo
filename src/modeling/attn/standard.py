"""Standard softmax attention"""
import math
import torch
import torch.nn as nn
from typing import Optional
from .ssmax import apply_ssmax_query_scaling

__all__ = ['eager_standard_attention']


def eager_standard_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
    use_ssmax: bool,
    ssmax_scale: Optional[torch.nn.Parameter] = None,
) -> torch.Tensor:
    """
    Standard softmax attention.
    
    Args:
        query_states: (batch, num_heads, q_len, head_dim)
        key_states: (batch, num_heads, kv_len, head_dim)
        value_states: (batch, num_heads, kv_len, head_dim)
        attention_mask: Optional mask
        head_dim: Head dimension
        use_ssmax: Apply SSMax scaling
        ssmax_scale: SSMax scale param (required if use_ssmax=True)
    
    Returns:
        Attention output
    """
    kv_len = key_states.shape[-2]
    if use_ssmax:
        query_states = apply_ssmax_query_scaling(query_states, kv_len, ssmax_scale)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    return attn_output
