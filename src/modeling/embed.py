"""Positional embeddings - Qwen3MoE compatible."""
import torch
from torch import nn

__all__ = ["BiBoRotaryEmbedding", "apply_rotary_pos_emb", "rotate_half", "_rotate_half"]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


_rotate_half = rotate_half


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BiBoRotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings - Qwen3MoE compatible.

    The model path passes 2D ``position_ids`` and receives Qwen-style
    ``(batch, seq, dim)`` cos/sin tensors. Older unit helpers can still pass
    ``seq_len=...`` or 1D position tensors and receive ``(seq, dim)`` tensors.
    """
    inv_freq: torch.Tensor

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids=None, seq_len=None):
        """
        Args:
            x: Input tensor, used for device and dtype.
            position_ids: Position indices, usually ``(batch, seq_len)``.
            seq_len: Compatibility path for callers that request positions
                ``0..seq_len-1`` or pass a 1D cache-position tensor.

        Returns:
            ``(cos, sin)`` with shape ``(batch, seq_len, dim)`` for 2D
            position ids, otherwise ``(seq_len, dim)``.
        """
        if position_ids is None:
            if seq_len is None:
                seq_len = x.shape[-2]
            if isinstance(seq_len, torch.Tensor):
                position_ids = seq_len
            else:
                position_ids = torch.arange(seq_len, device=x.device)

        if not isinstance(position_ids, torch.Tensor):
            position_ids = torch.arange(position_ids, device=x.device)

        squeeze_batch = position_ids.ndim == 1
        if squeeze_batch:
            position_ids = position_ids.unsqueeze(0)

        max_position = int(position_ids.max().item()) + 1
        self.max_seq_len_cached = max(self.max_seq_len_cached, max_position)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].to(device=x.device, dtype=torch.float)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        if squeeze_batch:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
        return cos, sin
