"""BiBoAttention base class"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from src.configuration_bibo import BiBoConfig
from ..norm import BiBoRMSNorm
from ..embed import BiBoRotaryEmbedding, apply_rotary_pos_emb
from .utils import repeat_kv
from .standard import eager_standard_attention
from .sliding import eager_sliding_window_attention
from .recurrent import eager_recurrent_attention
from .ssmax import apply_ssmax_query_scaling

__all__ = ['BiBoAttention']


class BiBoAttention(nn.Module):
    """
    Multi-Layer KV Sharing + GQA.
    
    Supports:
    1. GQA: Multiple Q heads, fewer K/V heads per layer
    2. MLKV: One set of K/V proj for every group of layers
    
    Args:
        config: Model config
        layer_idx: Layer index
        use_sliding_window: Use sliding window attn
    """
    def __init__(self, config: BiBoConfig, layer_idx: int, use_sliding_window: Optional[bool] = False):
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
        self.attention_type = getattr(config, "attention_type", "softmax")
        if self.attention_type == "sliding_window":
            use_sliding_window = True
        self.linear_attention_feature_map = getattr(config, "linear_attention_feature_map", "elu")
        self.linear_attention_eps = getattr(config, "linear_attention_eps", 1e-6)

        if self.use_ssmax:
            # SSMax init: scale * log(typical_kv_len) ≈ 1.0 at step 0
            # Prevents attention collapse early
            typical_log = math.log(max(config.max_position_embeddings / 2, 2.0))
            init_val = 1.0 / typical_log  # ≈ 0.13 for max_pos_emb=2048
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

        self.sliding_window = config.sliding_window if use_sliding_window else None
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        if self.attention_type in {"gdn", "kda"}:
            self.delta_beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        if self.attention_type == "gdn":
            self.delta_gate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        elif self.attention_type == "kda":
            self.delta_gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)

        self.rotary_emb = BiBoRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta,
        )

        if config.layer_norm_type != "rms":
            raise ValueError("Only 'rms' layer_norm_type supported")
        self.q_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _apply_ssmax_query_scaling(self, query_states: torch.Tensor, kv_len: int) -> torch.Tensor:
        """Apply SSMax query scaling (for backward compat with tests)"""
        if not self.use_ssmax:
            return query_states
        return apply_ssmax_query_scaling(query_states, kv_len, self.ssmax_scale)
    
    def eager_recurrent_attention(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper for recurrent attention (backward compat)"""
        return eager_recurrent_attention(
            hidden_states,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.attention_type,
            self.linear_attention_feature_map,
            self.linear_attention_eps,
            self.num_heads,
            self.head_dim,
            getattr(self, 'delta_beta_proj', None),
            getattr(self, 'delta_gate_proj', None),
        )
    
    def eager_sliding_window_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sliding_window: int,
        stride: Optional[int] = None,
    ) -> torch.Tensor:
        """Wrapper for sliding window attention (backward compat)"""
        return eager_sliding_window_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            sliding_window,
            self.head_dim,
            self.attention_dropout,
            self.training,
            self.use_ssmax,
            getattr(self, 'ssmax_scale', None),
            stride,
        )
    
    def eager_standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Wrapper for standard attention (backward compat)"""
        return eager_standard_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.head_dim,
            self.use_ssmax,
            getattr(self, 'ssmax_scale', None),
        )
    
    def _source_token_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Extract source token mask from attention mask (backward compat)"""
        if attention_mask is None:
            return None
        mask_slice = attention_mask[:, :, -1, :seq_len]
        min_value = torch.finfo(mask_slice.dtype).min
        source_mask = (mask_slice > min_value / 2).to(dtype)
        return source_mask.view(batch_size, 1, seq_len, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        recurrent_attention_types = {"linear", "gdn", "kda"}
        if past_key_value is not None and self.attention_type not in recurrent_attention_types:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        batch_size, num_heads, q_len, head_dim = query_states.shape
        
        if self.attention_type in {"linear", "gdn", "kda"}:
            attn_output = eager_recurrent_attention(
                hidden_states,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.attention_type,
                self.linear_attention_feature_map,
                self.linear_attention_eps,
                self.num_heads,
                self.head_dim,
                getattr(self, 'delta_beta_proj', None),
                getattr(self, 'delta_gate_proj', None),
            )
        elif self.sliding_window is not None:
            stride = getattr(self.config, 'sliding_window_stride', None)
            attn_output = eager_sliding_window_attention(
                query_states, 
                key_states, 
                value_states, 
                attention_mask,
                self.sliding_window,
                self.head_dim,
                self.attention_dropout,
                self.training,
                self.use_ssmax,
                getattr(self, 'ssmax_scale', None),
                stride
            )
        else:
            attn_output = eager_standard_attention(
                query_states, 
                key_states, 
                value_states, 
                attention_mask,
                self.head_dim,
                self.use_ssmax,
                getattr(self, 'ssmax_scale', None),
            )
        
        # Qwen-style: transpose BEFORE reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
