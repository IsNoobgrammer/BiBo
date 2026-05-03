"""MoE layer"""
import torch
import torch.nn as nn
from einops import rearrange
from src.configuration_bibo import BiBoConfig
from .mlp import BiBoMLP
from .experts import BiBoIdentityExpert, BiBoReLUExpert, BiBoZeroExpert, BiBoNoiseExpert, BiBoCausalConv1D
from .router import BiBoMoERouter

__all__ = ['BiBoMoELayer']


class BiBoMoELayer(nn.Module):
    """
    MoE layer.
    
    Args:
        config: Model config
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.bias_update_factor = config.bias_update_factor 
        self.bias_update_threshold = config.bias_update_threshold
        self.moe_shared_scaling = getattr(config, 'moe_shared_scaling', 1.0)

        # Token counter + accumulated TPE for threshold-based bias updates
        self.register_buffer("tokens_processed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        
        self.routed_experts = nn.ModuleList()
        n = config.num_routed_experts
        if n < 5:
            raise ValueError("num_routed_experts must be >= 5 (MLPs + identity + zero + noise + relu)")
        # (n - 4) MLP experts
        for _ in range(n - 4):
            self.routed_experts.append(BiBoMLP(config, is_expert=True))
        # 1 identity, 1 zero, 1 noise, 1 relu
        self.routed_experts.append(BiBoIdentityExpert(config))
        self.routed_experts.append(BiBoZeroExpert(config))
        self.routed_experts.append(BiBoNoiseExpert(config))
        self.routed_experts.append(BiBoReLUExpert(config))
        if len(self.routed_experts) != n:
            raise ValueError(f"Mismatch: Created {len(self.routed_experts)} routed experts, expected {n}")

        self.shared_experts_list = nn.ModuleList()
        # 1 shared conv expert
        self.shared_experts_list.append(BiBoCausalConv1D(config))
        self.gate = BiBoMoERouter(config)

    @torch.no_grad()
    def update_bias(self, tokens_per_expert: torch.Tensor):
        """
        Update router bias based on token distribution.
        Heuristic load balancing: increase bias for under-utilized experts.
        
        Args:
            tokens_per_expert: Count of tokens routed to each expert
        """
        if not hasattr(self.gate, 'bias') or self.bias_update_factor <= 0:
            return

        tpe = tokens_per_expert.detach().float() 
        if self.num_routed_experts > 0:
             mean_tpe = tpe.mean()
             deviation = mean_tpe - tpe 
        else:
             deviation = torch.zeros_like(tpe) 

        # bias += factor * sign(deviation)
        # bias ↑ if deviation > 0 (expert under-utilized)
        # bias ↓ if deviation < 0 (expert over-utilized)
        self.gate.bias.add_(self.bias_update_factor * deviation.sign())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing decisions
        top_k_indices, top_k_weights = self.gate(hidden_states)
        # top_k_indices: [bsz, seq_len, top_k], top_k_weights: [bsz, seq_len, top_k]

        # Global context for heuristic update
        tokens_per_expert = None
        if self.training and hasattr(self.gate, 'bias') and self.bias_update_factor > 0:
            current_tpe = torch.bincount(
                rearrange(top_k_indices, 'b s k -> (b s k)'), 
                minlength=self.num_routed_experts
            )
            
            batch_tokens = bsz * seq_len
            self.tokens_processed += batch_tokens
            self.accumulated_tpe += current_tpe.float()
            
            if self.tokens_processed >= self.bias_update_threshold:
                tokens_per_expert = self.accumulated_tpe.clone()
                self.tokens_processed.zero_()
                self.accumulated_tpe.zero_()

        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h') 
        final_routed = torch.zeros_like(flat_hidden) 
        flat_expert_indices = rearrange(top_k_indices, 'b s k -> (b s k)')
        flat_weights = rearrange(top_k_weights, 'b s k -> (b s k)')

        flat_token_indices = torch.arange(
            bsz * seq_len, device=hidden_states.device
        ).repeat_interleave(self.num_experts_per_tok)

        for i, expert in enumerate(self.routed_experts):
            mask = (flat_expert_indices == i)

            if mask.any():
                tokens_idx_for_expert = flat_token_indices[mask]
                weights_for_expert = flat_weights[mask].unsqueeze(1)

                # Optimization: process unique tokens
                unique_tokens, inverse_indices = torch.unique(tokens_idx_for_expert, return_inverse=True)
                inputs_for_expert = flat_hidden[unique_tokens]
                outputs_for_expert_unique = expert(inputs_for_expert)
                outputs_for_expert = outputs_for_expert_unique[inverse_indices]
                weighted_output = outputs_for_expert * weights_for_expert

                final_routed.scatter_add_(
                    0, 
                    tokens_idx_for_expert.unsqueeze(1).expand(-1, hidden_dim), 
                    weighted_output
                )

        final_routed = rearrange(final_routed, '(b s) h -> b s h', b=bsz)
        shared_combined = torch.zeros_like(hidden_states) 
        if self.shared_experts_list:
            shared_combined = self.shared_experts_list[0](hidden_states)
        # Scale shared expert (DeepSeek-V2/V3, Muon)
        final_output = final_routed + (getattr(self, 'moe_shared_scaling', 1.0) * shared_combined)

        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output
