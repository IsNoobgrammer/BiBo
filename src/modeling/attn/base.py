"""BiBoAttention — minimal shared shell: projections, QK-norm, partial RoPE, KV cache, SSMax,
per-layer dispatch to the SWA or full-attention flavor module, XSA, output projection.
The attention flavors themselves live in swa.py (eager-only band + sink) and
full_attention.py (SDPA fast path / mask path)."""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from src.configuration_bibo import BiBoConfig
from ..norm import BiBoRMSNorm
from ..embed import apply_rotary_pos_emb
from .ssmax import apply_ssmax_query_scaling
from .xsa import apply_xsa
from .swa import swa_attention
from .full_attention import full_attention

__all__ = ['BiBoAttention']


class BiBoAttention(nn.Module):
    """
    GQA attention with optional SSMax scaling.
    
    Args:
        config: Model config
        layer_idx: Layer index
    """
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
        self.scaling = self.head_dim ** -0.5     # 1/sqrt(head_dim); used by both SDPA and eager paths
        # Dim-wise partial RoPE: first rope_dim of EVERY head rotates, the rest is NoPE.
        self.rope_dim = config.rope_dim

        # Hybrid SWA vs global (docs/attention_layers.md §2). hybrid_layer_pattern[idx]==1 => sliding.
        pattern = getattr(config, "hybrid_layer_pattern", None)
        self.is_swa = bool(pattern[layer_idx]) if pattern is not None else False
        self.sliding_window = config.sliding_window if self.is_swa else None
        # SSMax forced OFF on SWA (n capped by window => redundant constant temperature).
        self.use_ssmax = config.use_ssmax and not self.is_swa
        # Attention sink: SWA gets it by the norm; global only if configured (G1/G3). Single
        # learnable scalar per head, appended as a value-less softmax column (GPT-OSS / MiMo style).
        use_sink = ((self.is_swa and config.add_swa_attention_sink_bias)
                    or (not self.is_swa and config.add_full_attention_sink_bias))
        self.attention_sink_bias = nn.Parameter(torch.zeros(self.num_heads)) if use_sink else None

        if self.use_ssmax:
            # SSMax init: scale * log(typical_kv_len) ≈ 1.0 at step 0
            # Prevents attention collapse early
            typical_log = math.log(max(config.max_position_embeddings / 2, 2.0))
            init_val = 1.0 / typical_log
            self.ssmax_scale = nn.Parameter(
                torch.full((1, self.num_heads, 1, 1), init_val),
                requires_grad=True
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        if config.layer_norm_type != "rms":
            raise ValueError("Only 'rms' layer_norm_type supported")
        self.q_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)

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

        # Dim-wise partial RoPE: rotate the first rope_dim of EVERY head, pass the rest through (NoPE).
        # cos/sin are sized rope_dim (built at model level).
        cos, sin = position_embeddings
        rd = self.rope_dim
        if rd < self.head_dim:
            q_rot, q_pass = query_states[..., :rd], query_states[..., rd:]
            k_rot, k_pass = key_states[..., :rd], key_states[..., rd:]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # SSMax query scaling. With a padding mask, n must count REAL keys only (grid positions
        # over-count by the pad width): mask.cumsum at each query's grid position == real causal
        # context length. SSMax runs only on global layers, whose KV is never window-cropped, so
        # the mask always covers the full kv_len.
        kv_len = key_states.shape[-2]
        if self.use_ssmax:
            context_lens = (attention_mask.cumsum(-1)[:, -query_states.shape[-2]:]
                            if attention_mask is not None else None)
            query_states = apply_ssmax_query_scaling(query_states, kv_len, self.ssmax_scale, context_lens)

        # Dispatch to the per-layer attention flavor. All masking (band/causal/padding) and sink
        # handling lives inside the flavor modules; attention_mask here is the raw 2D (B, K_total)
        # padding mask (1=real, 0=pad) or None. value_states stays GROUPED (un-repeated) so XSA's
        # enable_gqa broadcast below is consistent across all paths.
        if self.is_swa:
            # Eager only, by design — the fast path for SWA is a dedicated sink-aware banded
            # kernel (not SDPA); this eager core is its exact numerics target.
            attn_output, probs = swa_attention(
                query_states, key_states, value_states, self.attention_sink_bias,
                sliding_window=self.sliding_window,
                num_key_value_groups=self.num_key_value_groups, scaling=self.scaling,
                padding_mask=attention_mask,
                dropout=self.attention_dropout, training=self.training)
            attn_weights = probs if output_attentions else None
        else:
            attn_output, attn_weights = full_attention(
                query_states, key_states, value_states, self.attention_sink_bias,
                num_key_value_groups=self.num_key_value_groups, scaling=self.scaling,
                padding_mask=attention_mask,
                dropout=self.attention_dropout, training=self.training,
                output_attentions=output_attentions)

        # XSA: enable_gqa broadcasts V across the query group (no repeat_kv copy).
        if self.use_xsa:
            attn_output = apply_xsa(attn_output, value_states, enable_gqa=True)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        # No cache in the return: past_key_value is a shared object mutated in place by
        # .update() — threading it back through every layer return is HF-legacy ritual.
        return attn_output, attn_weights
