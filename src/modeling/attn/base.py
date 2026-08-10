"""BiBoAttention — minimal shared shell: projections, QK-norm, partial RoPE, KV cache,
per-layer dispatch to the SWA or full-attention flavor module, XSA, output projection.
The attention flavors themselves live in swa.py (flex band + eager reference) and
full_attention.py (SDPA fast path / mask path)."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from src.configuration_bibo import BiBoConfig
from ..norm import BiBoRMSNorm
from ..embed import apply_rotary_pos_emb
from .xsa import apply_xsa
from .swa import swa_attention
from .full_attention import full_attention

__all__ = ['BiBoAttention']


class BiBoAttention(nn.Module):
    """GQA attention with learnable XSA and per-layer SWA/global dispatch."""
    def __init__(self, config: BiBoConfig, layer_idx: int, **kwargs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim   # derived + validated in BiBoConfig
        self.layer_idx = layer_idx
        self.use_xsa = config.use_xsa
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5
        pattern = getattr(config, "hybrid_layer_pattern", None)
        self.is_swa = bool(pattern[layer_idx]) if pattern is not None else False

        # dim-wise partial RoPE: the first rope_dim of each head rotates, the rest passes through.
        # PER LAYER TYPE -- a window-W layer resolves distances <= W and a full-attention layer
        # resolves the whole context, so they get separate widths (and separate bases, handled by
        # DualRotaryEmbedding). 0 means NoPE for this layer: no rotation at all.
        self.rope_dim = (getattr(config, "swa_rope_dim", config.rope_dim) if self.is_swa
                         else config.rope_dim)
        # Hierarchical SWA reads its per-layer window off `sliding_window_per_layer`; plain
        # `sliding_window` stays scalar because transformers' cache indexes with it directly.
        _per = getattr(config, "sliding_window_per_layer", None)
        _sw = _per[layer_idx] if _per is not None else config.sliding_window
        self.sliding_window = _sw if self.is_swa else None

        # XSA rejection strength, one LEARNABLE logit per head; the applied strength is
        # tanh(xsa_alpha). Default init 0 -> tanh(0) = 0 -> XSA starts OFF and the model has to
        # switch it on, which is the configuration that won the 524M A/B. Not optional when
        # use_xsa: XSA and its learnable strength ship together.
        self.xsa_alpha = (nn.Parameter(torch.full((self.num_heads,), float(config.xsa_alpha_init)))
                          if self.use_xsa else None)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        if config.layer_norm_type != "rms":
            raise ValueError("Only 'rms' layer_norm_type supported")
        # QK-norm is unconditional on GLOBAL layers. On windowed layers it is an ablation axis:
        # MiMo-V2.5-Pro ships none anywhere, Gemma 4 applies it everywhere, and the argument for
        # dropping it here is that a 128-token span already bounds the logit range that QK-norm
        # exists to control. Identity keeps the forward path byte-identical and allocates no
        # parameters, so the arm shows up as a real param-count delta rather than a dead flag.
        _qk_norm = (not self.is_swa) or getattr(config, "swa_qk_norm", True)
        self.q_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps) if _qk_norm else nn.Identity()
        self.k_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps) if _qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # cos/sin are sized rope_dim (built at model level); the tail passes through as NoPE.
        # DualRotaryEmbedding hands down (global_pair, local_pair), either of which is None when
        # that layer type is NoPE. A bare (cos, sin) is still accepted so an external caller with
        # a plain BiBoRotaryEmbedding keeps working -- discriminated on whether entry 0 is a
        # tensor or a nested pair/None.
        _pe = position_embeddings
        _dual = _pe[0] is None or isinstance(_pe[0], (tuple, list))
        _pair = (_pe[1] if self.is_swa else _pe[0]) if _dual else _pe

        rd = self.rope_dim
        if _pair is None or rd == 0:
            cos = sin = None                      # NoPE layer: no rotation, heads pass through
        elif rd < self.head_dim:
            cos, sin = _pair
            q_rot, q_pass = query_states[..., :rd], query_states[..., rd:]
            k_rot, k_pass = key_states[..., :rd], key_states[..., rd:]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)
        else:
            cos, sin = _pair
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # All masking (band/causal/padding) lives inside the flavor modules;
        # `attention_mask` here is the raw 2D (B, K_total) padding mask (1=real, 0=pad) or None.
        # value_states stays GROUPED so XSA's enable_gqa broadcast is consistent on every path.
        if self.is_swa:
            # SWA's fast path is FlexAttention's block-sparse band, not SDPA; the eager core
            # in utils.py is that path's exact numerics target.
            attn_output, probs = swa_attention(
                query_states, key_states, value_states,
                sliding_window=self.sliding_window,
                num_key_value_groups=self.num_key_value_groups, scaling=self.scaling,
                padding_mask=attention_mask,
                dropout=self.attention_dropout, training=self.training)
            attn_weights = probs if output_attentions else None
        else:
            attn_output, attn_weights = full_attention(
                query_states, key_states, value_states,
                num_key_value_groups=self.num_key_value_groups, scaling=self.scaling,
                padding_mask=attention_mask,
                dropout=self.attention_dropout, training=self.training,
                output_attentions=output_attentions)

        # XSA: enable_gqa broadcasts V across the query group (no repeat_kv copy).
        if self.use_xsa:
            attn_output = apply_xsa(attn_output, value_states, enable_gqa=True,
                                    alpha=self.xsa_alpha)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        # No cache returned: past_key_value is mutated in place by .update().
        return attn_output, attn_weights
