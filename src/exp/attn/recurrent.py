"""Recurrent attention (linear/GDN/KDA)"""
import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

__all__ = ['eager_recurrent_attention']


def _linear_feature_map(states: torch.Tensor, feature_map: str, eps: float) -> torch.Tensor:
    """Apply feature map for linear attention"""
    if feature_map == "relu":
        return F.relu(states) + eps
    return F.elu(states) + 1.0 + eps


def _source_token_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Extract source token mask from attention mask"""
    if attention_mask is None:
        return None
    mask_slice = attention_mask[:, :, -1, :seq_len]
    min_value = torch.finfo(mask_slice.dtype).min
    source_mask = (mask_slice > min_value / 2).to(dtype)
    return source_mask.view(batch_size, 1, seq_len, 1)


def eager_recurrent_attention(
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    attention_type: str,
    feature_map: str,
    eps: float,
    num_heads: int,
    head_dim: int,
    delta_beta_proj: Optional[torch.nn.Module] = None,
    delta_gate_proj: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Recurrent attention family (linear/GDN/KDA).
    
    Args:
        hidden_states: Original hidden states (for delta proj)
        query_states: (batch, heads, seq, head_dim)
        key_states: (batch, heads, seq, head_dim)
        value_states: (batch, heads, seq, head_dim)
        attention_mask: Optional mask
        attention_type: "linear", "gdn", or "kda"
        feature_map: "elu" or "relu"
        eps: Epsilon for stability
        num_heads: Num attention heads
        head_dim: Head dimension
        delta_beta_proj: Beta projection (for GDN/KDA)
        delta_gate_proj: Gate projection (for GDN/KDA)
    
    Returns:
        Attention output
    """
    batch_size, num_heads, seq_len, _ = query_states.shape
    q = _linear_feature_map(query_states, feature_map, eps)
    k = _linear_feature_map(key_states, feature_map, eps)
    v = value_states

    token_mask = _source_token_mask(attention_mask, batch_size, seq_len, q.dtype)
    if token_mask is not None:
        k = k * token_mask
        v = v * token_mask

    if attention_type == "linear":
        kv_state = torch.cumsum(
            torch.einsum("bhtd,bhte->bhtde", k, v.to(k.dtype)),
            dim=2,
        )
        k_state = torch.cumsum(k, dim=2)
        numerator = torch.einsum("bhtd,bhtde->bhte", q, kv_state)
        denominator = torch.einsum("bhtd,bhtd->bht", q, k_state).unsqueeze(-1)
        return (numerator / denominator.clamp_min(eps)).to(value_states.dtype)

    beta = torch.sigmoid(delta_beta_proj(hidden_states))
    beta = rearrange(beta, "b t h -> b h t")

    if attention_type == "kda":
        gate = torch.sigmoid(delta_gate_proj(hidden_states))
        gate = rearrange(gate, "b t (h d) -> b h t d", h=num_heads)
    else:
        gate = torch.sigmoid(delta_gate_proj(hidden_states))
        gate = rearrange(gate, "b t h -> b h t")

    state = query_states.new_zeros(batch_size, num_heads, head_dim, head_dim)
    outputs = []
    for idx in range(seq_len):
        k_t = k[:, :, idx, :]
        v_t = v[:, :, idx, :]
        q_t = q[:, :, idx, :]

        if attention_type == "kda":
            state = state * gate[:, :, idx, :].unsqueeze(-1)
        else:
            state = state * gate[:, :, idx].unsqueeze(-1).unsqueeze(-1)

        old_v = torch.einsum("bhd,bhde->bhe", k_t, state)
        error = v_t - old_v
        delta = torch.einsum("bhd,bhe->bhde", k_t, error)
        state = state + beta[:, :, idx].unsqueeze(-1).unsqueeze(-1) * delta
        outputs.append(torch.einsum("bhd,bhde->bhe", q_t, state))

    return torch.stack(outputs, dim=2).to(value_states.dtype)
