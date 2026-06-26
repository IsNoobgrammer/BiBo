"""BiBoAttention — Standard softmax + SSMax"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from src.configuration_bibo import BiBoConfig
from ..norm import BiBoRMSNorm
from ..embed import BiBoRotaryEmbedding, apply_rotary_pos_emb
from .utils import repeat_kv
from .ssmax import apply_ssmax_query_scaling
from .xsa import apply_xsa

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
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx
        self.use_ssmax = config.use_ssmax
        self.use_xsa = config.use_xsa
        self.attention_dropout = config.attention_dropout
        # Partial RoPE: first num_rope_heads get RoPE, rest are NoPE (position-invariant content heads)
        self.num_rope_heads = self.num_heads - round(self.num_heads * getattr(config, 'rope_nope_ratio', 0.0))
        self.num_rope_kv_heads = self.num_rope_heads // self.num_key_value_groups

        if self.use_ssmax:
            # SSMax init: scale * log(typical_kv_len) ≈ 1.0 at step 0
            # Prevents attention collapse early
            typical_log = math.log(max(config.max_position_embeddings / 2, 2.0))
            init_val = 1.0 / typical_log
            self.ssmax_scale = nn.Parameter(
                torch.full((1, self.num_heads, 1, 1), init_val),
                requires_grad=True
            )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = BiBoRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta,
        )

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.num_rope_heads < self.num_heads:
            q_r, k_r = apply_rotary_pos_emb(
                query_states[:, :self.num_rope_heads],
                key_states[:, :self.num_rope_kv_heads],
                cos, sin)
            query_states = torch.cat([q_r, query_states[:, self.num_rope_heads:]], dim=1)
            key_states = torch.cat([k_r, key_states[:, self.num_rope_kv_heads:]], dim=1)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # SSMax query scaling
        kv_len = key_states.shape[-2]
        q = query_states
        if self.use_ssmax:
            q = apply_ssmax_query_scaling(q, kv_len, self.ssmax_scale)

        # Scaled dot-product attention. Materialize repeat_kv + DROP enable_gqa:
        # enable_gqa silently forces SDPA onto the MATH backend (materializes the O(n^2) scores,
        # ~10x slower + far more memory) because the mem-efficient/flash backends don't accept the
        # GQA broadcast on our torch/HW (and T4 sm_75 has no flash at all -> mem-efficient is the
        # production path). Full MHA via repeat_kv lets the dispatcher pick mem-efficient (T4) /
        # flash (Ampere+). The KV copy is O(B*H*S*D), not O(n^2). value_states stays grouped below
        # so the XSA call is unchanged.
        if not output_attentions:
            k_rep = repeat_kv(key_states, self.num_key_value_groups)
            v_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = nn.functional.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attention_mask[:, :, :, :kv_len] if attention_mask is not None else None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            attn_weights = None
        else:
            # Fallback to manual for output_attentions=True — repeat_kv needed here
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(q, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, :kv_len]
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        # XSA: enable_gqa broadcasts V across the query group (no repeat_kv copy).
        if self.use_xsa:
            attn_output = apply_xsa(attn_output, value_states, enable_gqa=True)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
