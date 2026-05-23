"""Decoder layer"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.configuration_bibo import BiBoConfig
from .norm import BiBoRMSNorm
from .attn import BiBoAttention
from .ffn import BiBoMLP, BiBoMoELayer

__all__ = ['BiBoDecoderLayer']


class BiBoDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    
    Args:
        config: Model config
        layer_idx: Layer index
    """
    def __init__(self, config: BiBoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = BiBoAttention(config=config, layer_idx=layer_idx)

        # MoE or dense MLP
        self.is_moe_layer = layer_idx not in config.mlp_only_layers
        if self.is_moe_layer:
            self.mlp = BiBoMoELayer(config)
        else:
            self.mlp = BiBoMLP(config, is_expert=False)

        self.input_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
