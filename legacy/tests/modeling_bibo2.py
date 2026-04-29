# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BiBo model (Simplified)."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeModelOutputWithPast,       # Using MoE specific outputs
    MoeCausalLMOutputWithPast,    # Using MoE specific outputs
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from sample import BiBoConfig

logger = logging.get_logger(__name__)

# --- Simple RMSNorm ---
class BiBoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

# --- Simple RoPE ---
class BiBoRotaryEmbedding(nn.Module):
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
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

# --- Simple MLP (SwiGLU) ---
class BiBoMLP(nn.Module):
    def __init__(self, config: BiBoConfig, is_expert=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if is_expert else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# --- Simple MHA ---
class BiBoAttention(nn.Module):
    def __init__(self, config: BiBoConfig, layer_idx: Optional[int] = None): # layer_idx unused in simple version but kept for signature
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # In simple MHA, num_key_value_heads is always num_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.use_bias = config.attention_bias

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.use_bias)
        self.rotary_emb = BiBoRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs, # Consume potential extra args like padding_mask
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, q_len, bsz)
        # Standard MHA: key and value have same shape as query before RoPE
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # --- Standard Scaled Dot Product Attention ---
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(f"Attention weights shape error! Expected {(bsz, self.num_heads, q_len, kv_seq_len)}, got {attn_weights.size()}")

        if attention_mask is not None:
             if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                 raise ValueError(f"Attention mask shape error! Expected {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}")
             attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        # --- End Standard SDPA ---

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"Attention output shape error! Expected {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# --- Simple MoE Router ---
class BiBoTopkRouter(nn.Module):
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts # Total experts available
        self.num_routed_experts = config.num_routed_experts # Number router routes to
        self.router_temperature = config.router_temperature
        self.gate = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: [*, hidden_dim] where * = batch_size * seq_len
        router_logits = self.gate(hidden_states)
        router_logits = router_logits / self.router_temperature # Apply temperature

        # Get Top-K scores and indices for the ROUTED experts
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # router_logits returned should be for aux loss calculation (before softmax/topk)
        return router_logits, routing_weights, selected_experts

# --- Simple MoE Block ---
class BiBoSparseMoeBlock(nn.Module):
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = BiBoTopkRouter(config)
        self.experts = nn.ModuleList([BiBoMLP(config, is_expert=True) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim) # [bs*seq_len, hidden_dim]
        num_tokens = hidden_states_reshaped.shape[0]

        # router_logits: [num_tokens, num_routed_experts] (raw for aux loss)
        # routing_weights: [num_tokens, top_k] (softmaxed weights for selected experts)
        # selected_experts: [num_tokens, top_k] (indices of selected experts)
        router_logits, routing_weights, selected_experts = self.router(hidden_states_reshaped)

        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        # Simple loop-based dispatch and combine (less efficient but clear)
        # Iterate over each token and its selected top-k experts
        for i in range(num_tokens):
            token_experts = selected_experts[i]
            token_weights = routing_weights[i]
            token_input = hidden_states_reshaped[i].unsqueeze(0) # [1, hidden_dim]

            expert_outputs = []
            for j in range(self.top_k):
                expert_idx = token_experts[j].item()
                # Check if expert_idx is within the bounds of actual experts list
                if expert_idx < self.num_experts:
                     expert_output = self.experts[expert_idx](token_input) # [1, hidden_dim]
                     expert_outputs.append(expert_output * token_weights[j])
                else:
                    # This shouldn't happen if num_routed_experts <= num_experts
                    logger.warning_once(f"Selected expert index {expert_idx} is out of bounds for {self.num_experts} experts.")
                    # Handle gracefully, e.g., skip or add zeros? Add zero contribution.
                    expert_outputs.append(torch.zeros_like(token_input))


            # Sum the weighted outputs of the top-k experts for this token
            if expert_outputs: # Avoid error if list is empty (should not happen with top_k > 0)
                final_hidden_states[i] = torch.stack(expert_outputs).sum(dim=0)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits # Return raw logits for aux loss

# --- Transformer Decoder Layer ---
class BiBoDecoderLayer(nn.Module):
    def __init__(self, config: BiBoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = BiBoAttention(config=config, layer_idx=layer_idx) # Simple MHA now

        # Determine if this layer uses MoE or Dense MLP
        self.is_moe_layer = layer_idx not in config.mlp_only_layers
        if self.is_moe_layer:
             self.mlp = BiBoSparseMoeBlock(config)
        else:
             # Use standard MLP for layers specified in mlp_only_layers
             self.mlp = BiBoMLP(config, is_expert=False) # is_expert=False uses intermediate_size

        self.input_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False, # Propagate flag
        use_cache: Optional[bool] = False,
        **kwargs, # Consume potential extra args like padding_mask
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # MLP or MoE Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_logits = None # Initialize router_logits to None
        if self.is_moe_layer:
            hidden_states, router_logits = self.mlp(hidden_states) # MoE block returns logits
        else:
            hidden_states = self.mlp(hidden_states) # Dense MLP block

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        # Conditionally add router logits if they exist and are requested
        if output_router_logits and router_logits is not None:
            outputs += (router_logits,)

        return outputs # Shape: (hidden_states, Optional[attn_weights], Optional[past_kv], Optional[router_logits])


# --- Base BiBo Model (Embeddings + Layers) ---
class BiBoPreTrainedModel(PreTrainedModel):
    config_class = BiBoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiBoDecoderLayer", "BiBoAttention", "BiBoSparseMoeBlock"] # Add MoE block
    _supports_flash_attn_2 = False # Disabled for simple version
    _supports_sdpa = False # Disabled for simple version
    _supports_cache_class = True # Can use HF cache mechanism

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# --- Main BiBoModel ---
# Copied from Llama/Mistral structure
class BiBoModel(BiBoPreTrainedModel):
    def __init__(self, config: BiBoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BiBoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Minimal _prepare_decoder_attention_mask for causal LM
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None, # Add MoE specific output flag
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]: # Use MoE output type

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Add output_router_logits flag based on config or input
        output_router_logits = (
             output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            if past_key_values is None: # Handle the case where past_key_values is None initally
                past_key_values = [None] * len(self.layers)
            elif len(past_key_values) != len(self.layers):
                 raise ValueError(f"Expected past_key_values to have length {len(self.layers)}, but got {len(past_key_values)}")
            # Determine past length from the first non-None past_key_value tuple
            for pkv in past_key_values:
                if pkv is not None and pkv[0] is not None:
                     past_key_values_length = pkv[0].shape[2]
                     break

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Simplified causal mask preparation for eager attention
        if attention_mask is None and seq_length > 1 :
             attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=inputs_embeds.device) # Use full mask if None provided? Or default to causal? Let's assume causal intent.
             # Need a proper causal mask function here. Using a placeholder.
             # This requires helper functions _make_causal_mask, _expand_mask usually.
             # Let's assume the user provides a correct mask or seq_len=1 for generation.
             # For training (seq_len > 1), a causal mask is essential.
             # Simplification: Assume user handles mask, or rely on attention layer internal logic if it exists.
             # The HF Llama implementation uses `_prepare_decoder_attention_mask`. Let's use a basic version.
             attention_mask_4d = _prepare_4d_causal_attention_mask(
                 attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
             )
        elif attention_mask is not None:
            # Expand the provided mask to 4D
             attention_mask_4d = _prepare_4d_causal_attention_mask(
                 attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
             )
        else: # seq_len == 1 or attention_mask is explicitly None
            attention_mask_4d = None


        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None # Collect router logits
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    position_ids,
                    None, # past_key_value must be None for checkpointing
                    output_attentions,
                    output_router_logits, # Pass flag for checkpointing
                    False, # use_cache must be False for checkpointing
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits, # Pass flag to layer
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],) # Index depends on attention output presence

            if output_attentions:
                 all_self_attns += (layer_outputs[1],)

            # Collect router logits if produced and requested
            # Check the number of outputs from the layer to see if logits are present
            if output_router_logits:
                 # The router logits are the last element if present
                 # Check if the layer is MoE and actually returned logits
                 if decoder_layer.is_moe_layer and len(layer_outputs) > (2 if output_attentions else 1) + (1 if use_cache else 0):
                      all_router_logits += (layer_outputs[-1],)
                 # else: # Dense layer or MoE layer didn't output logits (shouldn't happen if flag is true)
                 #     all_router_logits += (None,) # Add placeholder? Or skip? Let's skip.


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        # Decide which output class to use based on whether router logits were collected
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                 outputs += (all_hidden_states,)
            if output_attentions:
                 outputs += (all_self_attns,)
            if use_cache: # Should be next_cache
                 outputs += (next_cache,)
            if output_router_logits and all_router_logits: # Check if list is not empty
                 outputs += (all_router_logits,)
            return outputs

        # Return MoeModelOutputWithPast if router logits are present, otherwise BaseModelOutputWithPast
        if output_router_logits and all_router_logits:
            return MoeModelOutputWithPast(
                 last_hidden_state=hidden_states,
                 past_key_values=next_cache,
                 hidden_states=all_hidden_states,
                 attentions=all_self_attns,
                 router_logits=all_router_logits, # Include router logits
            )
        else:
             return BaseModelOutputWithPast( # Fallback to base output if no router logits
                 last_hidden_state=hidden_states,
                 past_key_values=next_cache,
                 hidden_states=all_hidden_states,
                 attentions=all_self_attns,
            )


# --- Causal LM Head Model ---
class BiBoForCausalLM(BiBoPreTrainedModel):
    # Define _tied_weights_keys if applicable, e.g., ["lm_head.weight"] if tying
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BiBoConfig):
        super().__init__(config)
        self.model = BiBoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MoE Aux Loss calculation parameters
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok # k

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # --- MoE Aux Loss Calculation (simplified from Mixtral) ---
    def _calculate_aux_loss(self, all_router_logits):
        if all_router_logits is None or self.router_aux_loss_coef == 0:
             return torch.tensor(0.0, device=self.device) # No loss if no logits or coef is 0

        # all_router_logits is a tuple of tensors, shape [num_tokens, num_routed_experts]
        # Concatenate logits across layers if needed, or calculate per layer and average?
        # Mixtral concatenates first. Let's assume that's desired.
        # Each element in tuple might have shape [batch * seq_len, num_routed_experts]
        concatenated_router_logits = torch.cat([l for l in all_router_logits if l is not None], dim=0)

        if concatenated_router_logits.numel() == 0:
             return torch.tensor(0.0, device=self.device)

        num_tokens, num_routed_experts = concatenated_router_logits.shape

        # Calculate load balancing loss
        # Need probabilities and expert indices per token (router doesn't return indices easily here)
        # Let's compute based on the definition using logits directly (as in Mixtral source):
        #   loss = sum(mean(prob_per_expert) * mean(router_output_per_expert))
        # where prob_per_expert = fraction of tokens routed to expert
        # and router_output_per_expert = mean router output value for tokens routed to expert

        # Compute softmax over routed experts for each token
        routing_probabilities = F.softmax(concatenated_router_logits, dim=-1, dtype=torch.float32) # [num_tokens, num_routed_experts]

        # Calculate expert load: Fraction of tokens where this expert is in the top-k
        # This requires knowing the top-k indices, which we only had inside the MoE block.
        # Alternative aux loss formulation (simpler, used in some papers):
        # Encourage router probability distribution to be uniform.
        # Maximize entropy of mean probabilities per expert?
        # Minimize variance of load?

        # Let's use the Mixtral aux loss formulation as implemented in HF:
        # It calculates P = mean(softmax(logits)) over tokens, for each expert
        # and f = mean(chosen_expert_mask) over tokens, for each expert
        # Loss = coef * N * sum(P * f) where N = num_experts
        # This still requires the expert mask (which experts were chosen).

        # --- Simpler Aux Loss (Load Balancing based on probabilities) ---
        # Calculate the average probability assigned to each expert across all tokens.
        avg_prob_per_expert = torch.mean(routing_probabilities, dim=0) # [num_routed_experts]
        # Calculate the squared sum of average probabilities, encourages uniformity.
        # Minimize sum(avg_prob^2) is similar to minimizing variance.
        # Scaling factor: num_routed_experts
        load_balancing_loss = num_routed_experts * torch.sum(avg_prob_per_expert * avg_prob_per_expert)


        # Scale by coefficient
        aux_loss = self.router_aux_loss_coef * load_balancing_loss

        return aux_loss.to(self.device)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None, # Need this True for aux loss
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]: # Use MoE Causal LM output

        # Ensure router logits are requested if aux loss coef > 0 for training
        if self.training and self.router_aux_loss_coef > 0 and output_router_logits is not True:
            # Force output_router_logits to True during training if aux loss is enabled
            # logger.warning_once("Training with router_aux_loss_coef > 0 requires output_router_logits=True. Setting it automatically.")
            output_router_logits = True # Set it automatically
        elif output_router_logits is None:
            # Default to config if not specified
             output_router_logits = self.config.output_router_logits


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Decoder output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits, # Pass the flag
            return_dict=True, # Force return_dict=True from base model for easier handling
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float() # Cast to float32 for stability

        loss = None
        aux_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # Calculate MoE auxiliary loss if applicable
            if output_router_logits and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
                aux_loss = self._calculate_aux_loss(outputs.router_logits)
                if loss is not None and aux_loss is not None: # Add aux loss to main loss if both exist
                    loss += aux_loss
            elif self.router_aux_loss_coef > 0:
                # If coef > 0 but logits weren't returned (e.g., during eval with output_router_logits=False)
                # aux_loss should be considered 0.
                 pass # aux_loss remains None or 0

        if not return_dict:
            output = (logits,) + outputs[1:] # Skip last_hidden_state from base model output tuple
            if loss is not None:
                 output = (loss,) + output
            # Add aux_loss to tuple if calculated
            if aux_loss is not None: # Only include if calculated
                output = output + (aux_loss,) # Append aux_loss
            return output

        # Decide which output class to use
        if output_router_logits and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            return MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=aux_loss, # Include aux_loss
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=outputs.router_logits,
            )
        else:
             # If router logits weren't requested/returned, use standard output type
             return CausalLMOutputWithPast(
                 loss=loss,
                 logits=logits,
                 past_key_values=outputs.past_key_values,
                 hidden_states=outputs.hidden_states,
                 attentions=outputs.attentions,
            )


# --- Helper functions for mask creation (minimal version) ---
# Simplified versions from transformers.modeling_attn_mask_utils
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """Make causal mask used for self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
):
    """Creates a causal 4D mask based on `attention_mask` and `past_key_values_length`."""
    bsz, seq_len = input_shape
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    causal_mask = _make_causal_mask(
        (bsz, seq_len),
        inputs_embeds.dtype,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
    )
    if attention_mask is not None:
         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
         expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len).to(
             inputs_embeds.device
         )
         # Combine masks: 0 where we attend, -inf where we don't
         # Causal mask has 0 for allowed past positions, -inf for future
         # Expanded mask has 0 for non-padding, -inf for padding
         # Adding them keeps -inf if either mask forbids attention
         causal_mask = causal_mask + expanded_attn_mask # Note the order might matter if expand_mask uses 0 for padding
         # Ensure the final mask uses -inf correctly
         causal_mask = causal_mask.masked_fill(causal_mask > -1.0, 0.0) # Make sure valid positions are 0

    return causal_mask


if __name__ == "__main__":
    config = BiBoConfig(
        num_hidden_layers=2,  # Reduced layers for speed
        hidden_size=64,  # Reduced hidden size
        intermediate_size=128,  # Reduced MLP intermediate
        moe_intermediate_size=32,  # Reduced MoE intermediate
        num_attention_heads=4,  # Reduced heads
        num_routed_experts=4,  # Reduced experts
        num_experts_per_tok=2,
        vocab_size=1000,  # Smaller vocab
        max_position_embeddings=128  # Shorter sequence length capacity
    )
    print("Created BiBoConfig instance:")
    print(config)
    print("-" * 30)

    model = BiBoForCausalLM(config)

    model.eval()

    batch_size = 2
    seq_length = 15  # Should be <= config.max_position_embeddings
    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask
        )
    print("Forward pass successful.")
    print("Output object type:", type(outputs))