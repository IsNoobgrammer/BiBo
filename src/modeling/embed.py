"""Positional embeddings"""
import torch
from torch import nn

__all__ = ['BiBoRotaryEmbedding', 'apply_rotary_pos_emb', '_rotate_half']


class BiBoRotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings.
    
    Args:
        dim (int): Embedding dim
        max_position_embeddings (int): Max seq len. Default: 2048
        base (int): Base for inv freq. Default: 10000
        device: Device for tensors
    
    Returns:
        Tuple[Tensor, Tensor]: (cos, sin) embeddings for seq len
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        """
        Args:
            x: Input tensor (for device/dtype)
            seq_len: Seq len (int or tensor) - if tensor, use max value
        Returns:
            (cos, sin) with shape (seq_len, head_dim)
        """
        # Handle tensor seq_len (e.g., cache_position)
        if isinstance(seq_len, torch.Tensor):
            # Use max position + 1 for length
            seq_len = seq_len.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def _rotate_half(x):
    """Split last dim in half, rotate by concat(-x2, x1)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply RoPE to q, k.
    
    Args:
        q: Query (batch, num_heads, seq_len, head_dim)
        k: Key (batch, num_heads, seq_len, head_dim)
        cos: Cos embeddings (max_seq_len, head_dim) or (seq_len, head_dim)
        sin: Sin embeddings (max_seq_len, head_dim) or (seq_len, head_dim)
        unsqueeze_dim: Dim to unsqueeze (default=1 for num_heads)
    
    Returns:
        (q_embed, k_embed) with RoPE applied
    """
    # Get seq_len from q
    seq_len = q.shape[2]
    # Slice cos/sin to match seq_len
    cos = cos[:seq_len, :]
    sin = sin[:seq_len, :]
    # Unsqueeze: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
