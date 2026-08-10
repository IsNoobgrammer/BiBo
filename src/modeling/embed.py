"""Positional embeddings - Qwen3MoE compatible"""
import torch
from torch import nn
from typing import Optional
from transformers.utils.generic import maybe_autocast

__all__ = ['BiBoRotaryEmbedding', 'DualRotaryEmbedding', 'apply_rotary_pos_emb', 'rotate_half']


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
    """
    Rotary positional embeddings - Qwen3MoE compatible.
    Returns (batch, seq, dim) shaped cos/sin.
    """
    inv_freq: torch.Tensor

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None,
                 rope_type="none", scaling_factor=1.0):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.dim = dim
        self.base = base
        self.rope_type = rope_type            # "none" | "dynamic" (NTK-aware)
        self.scaling_factor = scaling_factor

        inv_freq = self._compute_inv_freq(base, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling = 1.0

    def _compute_inv_freq(self, base, device):
        return 1.0 / (
            base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / self.dim)
        )

    def _ntk_base(self, seq_len):
        """Dynamic NTK base. Clamp seq_len to the trained window so seq_len <= L_orig
        is exactly identity (scale=1 -> base unchanged)."""
        seq_len = max(seq_len, self.original_max_seq_len)
        f = self.scaling_factor
        scale = (f * seq_len / self.original_max_seq_len) - (f - 1)
        return self.base * scale ** (self.dim / (self.dim - 2))

    @torch.no_grad()
    def _inv_freq_for(self, position_ids, device, seq_len=None):
        """STATELESS dynamic-NTK inv_freq for the CURRENT sequence length — order-independent:
        identical inputs always yield identical frequencies (no grow/reset history that made the
        result depend on prior batch lengths). In-window (seq_len <= original_max) returns the base
        inv_freq unchanged (a no-op — no recompute); only out-of-window pays one small recompute.
        `seq_len` is a host int (past + current tokens) supplied by the model so the extent is known
        WITHOUT a CPU<-GPU sync / torch.compile graph break; the int(position_ids.max()) fallback
        (one sync per dynamic forward, matches HF's dynamic-NTK) only runs for external callers."""
        if self.rope_type != "dynamic":
            return self.inv_freq
        if seq_len is None:
            seq_len = int(position_ids.max()) + 1
        if seq_len <= self.original_max_seq_len:
            return self.original_inv_freq.to(device)
        return self._compute_inv_freq(self._ntk_base(seq_len), device)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Args:
            x: Input tensor (for device/dtype)
            position_ids: (batch, seq_len) position indices
            seq_len: optional host int = total positions (past + current); avoids a GPU sync
                on the dynamic-NTK path
        Returns:
            (cos, sin) with shape (batch, seq_len, dim)
        """
        inv_freq = self._inv_freq_for(position_ids, x.device, seq_len)

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DualRotaryEmbedding(nn.Module):
    """cos/sin for FULL-attention and SLIDING-window layers, which need different spectra.

    A window-W layer only ever resolves relative distances <= W; a full-attention layer resolves
    the whole context. Every shipped SWA model splits this (MiMo V2.5 Pro 10k/10M at window 128,
    Gemma 3 10k/1M at window 1024, Muse Glimmer 500k local and 0 = NoPE global), so the two layer
    types get their own base AND their own rotary width.

    forward() returns ``(global_pair, local_pair)``; each entry is a ``(cos, sin)`` tuple, or
    ``None`` when that layer type is NoPE (rope_dim 0). Consumers index with ``is_swa``.
    The two share one underlying module when base and width match, so the common case costs
    exactly what a single BiBoRotaryEmbedding cost before.
    """

    def __init__(self, config):
        super().__init__()
        rt = config.rope_scaling["type"]
        sf = config.rope_scaling.get("factor", 1.0)
        mp = config.max_position_embeddings

        def _mk(dim, base):
            if dim == 0:                       # NoPE for this layer type
                return None
            return BiBoRotaryEmbedding(dim, max_position_embeddings=mp, base=base,
                                       rope_type=rt, scaling_factor=sf)

        self.glob = _mk(config.rope_dim, config.rope_theta)
        self._shared = (config.swa_rope_dim == config.rope_dim
                        and config.swa_rope_theta == config.rope_theta)
        self.local = None if self._shared else _mk(config.swa_rope_dim, config.swa_rope_theta)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        g = None if self.glob is None else self.glob(x, position_ids, seq_len=seq_len)
        if self._shared:
            return (g, g)
        l = None if self.local is None else self.local(x, position_ids, seq_len=seq_len)
        return (g, l)
