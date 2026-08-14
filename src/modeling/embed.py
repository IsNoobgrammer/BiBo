"""Positional embeddings - Qwen3MoE compatible"""
import torch
from torch import nn
from transformers.utils.generic import maybe_autocast

__all__ = ['BiBoRotaryEmbedding', 'apply_rotary_pos_emb', 'rotate_half']


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BiBoRotaryEmbedding(nn.Module):
    """RoPE over the FULL head_dim, for the sliding-window layers only.

    The architecture is fixed (Aug 14 2026): full-attention layers are NoPE and windowed layers get
    full RoPE, so there is nothing to configure -- no partial-rotary fraction, no per-layer-type
    width, no second base. Attention applies this on `is_swa` layers and skips it everywhere else.

    Dynamic-NTK scaling is gone with it. It keyed off max_position_embeddings and silently rescaled
    the base at eval (seq 4095 -> base 10000 * 2^(dim/(dim-2))), so ctx1024/ctx2048 and ctx4095 were
    measured on two different models, and it stretched the windowed layers, which never see past
    `sliding_window` tokens.
    """

    inv_freq: torch.Tensor

    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                                   .to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """(cos, sin), each (batch, seq, dim)."""
        inv_freq = self.inv_freq
        inv_freq_expanded = (inv_freq[None, :, None].float()
                             .expand(position_ids.shape[0], -1, 1).to(x.device))
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (x.device.type
                       if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu")
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
