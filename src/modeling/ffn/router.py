"""MoE router"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoMoERouter']


class BiBoMoERouter(nn.Module):
    """
    MoE router. Supports MLP or conv routing.
    
    Args:
        config: Model config
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.temperature = config.router_temperature
        self.router_noise = config.router_noise
        self.router_type = config.router_type
        self.kernel_size = config.kernel_size
        self.causal_padding = self.kernel_size - 1
        self.router_lambda = getattr(config, 'router_lambda', 1.0)

        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts))        
        if self.router_type == "mlp":
            self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
        elif self.router_type == "conv":
            self.gate_conv = nn.Conv1d(config.hidden_size, self.num_routed_experts, self.kernel_size, padding=0, bias=False)
        else:
            raise ValueError(f"Unknown router type: {self.router_type}. Expected 'mlp' or 'conv'.")

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            top_k_indices: (batch, seq_len, top_k)
            norm_weights: (batch, seq_len, top_k)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if self.router_type == "mlp":
            flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
            router_logits = self.gate_proj(flat_hidden).float()
        else:  
            x_perm = rearrange(hidden_states, 'b s h -> b h s')
            x_padded = F.pad(x_perm, (self.causal_padding, 0))
            conv_out = self.gate_conv(x_padded)
            router_logits = rearrange(conv_out, 'b e s -> (b s) e').float()

        if self.training and self.router_noise > 0:
            noise_stddev = math.sqrt(self.router_noise)
            noise = torch.randn_like(router_logits) * noise_stddev
            router_logits = router_logits + noise.detach()  

        router_logits = router_logits + self.bias
        # z = lambda * (z - mean) / std (Skywork-MoE)
        mean = router_logits.mean(dim=1, keepdim=True)
        std = router_logits.std(dim=1, keepdim=True) + 1e-6
        router_logits_norm = (router_logits - mean) / std
        router_logits_scaled = self.router_lambda * router_logits_norm

        routing_weights = F.softmax(router_logits_scaled, dim=1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        norm_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.to(hidden_states.dtype)
