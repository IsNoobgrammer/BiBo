"""Attention utilities"""
import torch

__all__ = ['repeat_kv']


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
