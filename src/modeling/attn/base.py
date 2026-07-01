"""BiBoAttention — Standard softmax + SSMax + hybrid SWA / attention sink"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from src.configuration_bibo import BiBoConfig
from ..norm import BiBoRMSNorm
from ..embed import apply_rotary_pos_emb
from .utils import repeat_kv
from .ssmax import apply_ssmax_query_scaling
from .xsa import apply_xsa

__all__ = ['BiBoAttention', 'eager_attention_forward']


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, sinks=None):
    """Eager attention core — faithful to MiMo-V2.5 `eager_attention_forward` (GPT-OSS-style sink).

    Verbatim except the final `.transpose(1,2)` is deferred to the caller (BiBo runs XSA on the
    (B,H,q,d) output first, then the shared tail transposes). `key`/`value` are GROUPED (GQA) and
    repeated here. `attention_mask` is additive (0 / -inf), broadcast over (B,H). `sinks` is
    `module.attention_sink_bias` (or None): when set, one value-less per-head sink column is
    concatenated AFTER the mask, included in the softmax denominator, then dropped before the V
    matmul. Returns (attn_output (B,H,q,d), probs (real weights, sink dropped))."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    if sinks is not None:
        s = module.attention_sink_bias.reshape(1, -1, 1, 1).to(attn_weights.dtype).expand(
            query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, s], dim=-1)
    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if sinks is not None:
        probs = probs[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training)
    attn_output = torch.matmul(probs, value_states)
    return attn_output, probs


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

        if config.layer_norm_type != "rms":
            raise ValueError("Only 'rms' layer_norm_type supported")
        self.q_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _attn_bias_mask(self, q_len, kv_len, dtype, device):
        """Additive attention mask (0 / -inf), shape (1, 1, q_len, kv_len) — broadcasts over (B,H).
        Causal, plus a sliding-window band when this is an SWA layer. Bottom-right aligned via
        absolute query positions, so it's correct for packed training (q_len==kv_len) and cached
        decode (q_len==1, kv_len large) alike."""
        i = torch.arange(q_len, device=device).unsqueeze(1) + (kv_len - q_len)   # abs query pos
        j = torch.arange(kv_len, device=device).unsqueeze(0)
        allow = j <= i                                                            # causal
        if self.sliding_window is not None:
            allow = allow & ((i - j) < self.sliding_window)                       # window band
        mask = torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), float("-inf"), dtype=dtype, device=device))
        return mask[None, None]                                                    # (1,1,q,kv)

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

        # SSMax query scaling
        kv_len = key_states.shape[-2]
        q = query_states
        if self.use_ssmax:
            q = apply_ssmax_query_scaling(q, kv_len, self.ssmax_scale)

        # Attention. Global + no-sink + no-attn-weights -> SDPA fast path (is_causal lets the backend
        # SKIP the upper triangle; see AGENTS.md decision 9b). Anything needing a sink column, a
        # sliding-window band, or attention weights -> the eager core (matches ssmax_sink_ref.py /
        # MiMo eager_attention_forward). value_states stays GROUPED (un-repeated) so XSA's enable_gqa
        # broadcast below is consistent across both paths.
        q_len = q.shape[-2]
        need_sink = self.attention_sink_bias is not None
        use_eager = need_sink or self.is_swa or output_attentions
        if not use_eager:
            # Global + no-sink fast path. value_states stays grouped for XSA's enable_gqa below.
            k_rep = repeat_kv(key_states, self.num_key_value_groups)
            v_rep = repeat_kv(value_states, self.num_key_value_groups)
            if q_len > 1 and kv_len > q_len:
                # Cached multi-token prefill: SDPA is_causal is TOP-LEFT aligned (wrong when a cache
                # is present) — pass an explicit bottom-right causal mask instead (F9).
                sdpa_mask = self._attn_bias_mask(q_len, kv_len, q.dtype, q.device)
                attn_output = nn.functional.scaled_dot_product_attention(
                    q, k_rep, v_rep, attn_mask=sdpa_mask, is_causal=False,
                    dropout_p=self.attention_dropout if self.training else 0.0, scale=self.scaling)
            else:
                # Training (q_len==kv_len) and single-token decode: is_causal is correct and lets the
                # backend SKIP the upper triangle (AGENTS.md decision 9b).
                attn_output = nn.functional.scaled_dot_product_attention(
                    q, k_rep, v_rep, attn_mask=None, is_causal=q_len > 1,
                    dropout_p=self.attention_dropout if self.training else 0.0, scale=self.scaling)
            attn_weights = None
        else:
            # Eager core (MiMo eager_attention_forward): causal-or-sliding mask + optional per-head
            # sink. SWA sink is UNSCALED (no SSMax on SWA — docs/attention_layers.md §3); the sink is
            # appended AFTER the mask so it's never masked out. Pass grouped key/value (repeated inside).
            # Build the mask whenever q_len>1 OR this is a windowed layer — the latter so SWA enforces
            # its band even at q_len==1 single-token decode (F5); a global no-sink q_len==1 never lands
            # here, and would need no mask anyway.
            attn_mask = (self._attn_bias_mask(q_len, kv_len, q.dtype, q.device)
                         if (q_len > 1 or self.sliding_window is not None) else None)
            attn_output, probs = eager_attention_forward(
                self, q, key_states, value_states, attn_mask, self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                sinks=self.attention_sink_bias,
            )
            attn_weights = probs if output_attentions else None

        # XSA: enable_gqa broadcasts V across the query group (no repeat_kv copy).
        if self.use_xsa:
            attn_output = apply_xsa(attn_output, value_states, enable_gqa=True)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
