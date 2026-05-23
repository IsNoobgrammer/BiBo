"""MoE layer — fused expert dispatch"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig
from .experts import BiBoCausalConv1D
from .mlp import BiBoMLP
from .router import BiBoMoERouter

__all__ = ['BiBoMoELayer']


# PolyGLU activation cycle: each group of 3 experts gets one of each
_POLYGLU_ACTIVATIONS = ("silu", "relu2", "tanh")


class BiBoFusedExperts(nn.Module):
    """
    Fused PolyGLU experts — Qwen-style 3D weight tensors with per-expert activation.
    
    All PolyGLU expert weights stored as:
      - gate_up_proj: (num_polyglu_experts, 2 * intermediate, hidden)
      - down_proj: (num_polyglu_experts, hidden, intermediate)
    
    Dispatch: loop over active experts (skip empty), one fused F.linear per expert.
    No torch.unique — direct gather + index_add_ (Qwen pattern).
    
    Identity/Zero experts handled with zero compute.
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.use_uniform_experts = getattr(config, 'use_uniform_experts', False)
        self.num_polyglu_experts = config.polyglu_expert_multiplier * 3
        self.special_expert_pairs = config.special_expert_pairs
        self.num_routed_experts = config.num_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        
        if self.use_uniform_experts:
            # All experts are SiLU GLU (matches Qwen3MoE exactly)
            # No Identity/Zero — all num_routed_experts are fused SiLU experts
            num_fused = self.num_routed_experts
        else:
            num_fused = self.num_polyglu_experts
        
        self.num_fused_experts = num_fused
        
        # Fused weight tensors for all fused experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_fused, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_fused, self.hidden_size, self.intermediate_size)
        )
        
        # Initialize — normal_(0, std) like Qwen3MoEExperts
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)
        
        # Pre-compute activation name for each expert index
        # Layout: [SiLU_0, ReLU²_0, Tanh_0, SiLU_1, ReLU²_1, Tanh_1, ...]
        # If use_uniform_experts=True, all experts use SiLU (matches Qwen3MoE)
        self._expert_activations = []
        if self.use_uniform_experts:
            for _ in range(self.num_routed_experts):
                self._expert_activations.append("silu")
        else:
            for _ in range(config.polyglu_expert_multiplier):
                for act in _POLYGLU_ACTIVATIONS:
                    self._expert_activations.append(act)
        
        # Identity indices: [num_polyglu, num_polyglu + pairs)
        # Zero indices: [num_polyglu + pairs, num_routed)
        # When use_uniform_experts=True, these ranges are empty (no special experts)
        if self.use_uniform_experts:
            self.identity_start = self.num_routed_experts  # empty range
            self.identity_end = self.num_routed_experts
            self.zero_start = self.num_routed_experts
            self.zero_end = self.num_routed_experts
        else:
            self.identity_start = self.num_polyglu_experts
            self.identity_end = self.num_polyglu_experts + self.special_expert_pairs
            self.zero_start = self.identity_end
            self.zero_end = self.num_routed_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (num_tokens, hidden_size)
            top_k_indices: (num_tokens, top_k)
            top_k_weights: (num_tokens, top_k)
        Returns:
            output: (num_tokens, hidden_size)
        """
        output = torch.zeros_like(hidden_states)
        
        # One-hot → expert_mask[expert_idx] tells us which (top_k_pos, token_idx) pairs route there
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_routed_experts)
        # expert_mask: (num_tokens, top_k, num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        # expert_mask: (num_experts, top_k, num_tokens)
        
        # Find which experts actually have tokens (skip empty ones)
        with torch.no_grad():
            expert_hit = (expert_mask.sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0]
        
        for expert_idx in expert_hit:
            eidx = expert_idx.item()
            
            # Get (top_k_position, token_index) pairs for this expert
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            current_state = hidden_states[token_idx]  # (n_tokens_for_expert, hidden)
            current_weights = top_k_weights[token_idx, top_k_pos, None]  # (n, 1)
            
            if eidx < self.num_fused_experts:
                # PolyGLU expert: fused gate+up → activation → down
                gate_up = F.linear(current_state, self.gate_up_proj[eidx])  # (n, 2*inter)
                gate, up = gate_up.chunk(2, dim=-1)
                
                # Per-expert activation
                act_name = self._expert_activations[eidx]
                if act_name == "silu":
                    activated = F.silu(gate)
                elif act_name == "relu2":
                    activated = F.relu(gate).square()
                else:  # tanh
                    activated = torch.tanh(gate)
                
                current_hidden = F.linear(activated * up, self.down_proj[eidx])  # (n, hidden)
                current_hidden = current_hidden * current_weights
                output.index_add_(0, token_idx, current_hidden)
                
            elif eidx < self.identity_end:
                # Identity expert: pass through with weight
                output.index_add_(0, token_idx, current_state * current_weights)
                
            else:
                # Zero expert: multiply by 0 to keep graph alive for router gradients.
                # Without this, routing weights for Zero get no gradient when top_k=1.
                # Cost: one multiply + index_add of zeros (trivial).
                output.index_add_(0, token_idx, current_state * (current_weights * 0))

        return output


class BiBoMoELayer(nn.Module):
    """
    MoE layer with fused PolyGLU expert dispatch.
    
    Expert layout:
      - polyglu_expert_multiplier groups × 3 activations (SiLU, ReLU², Tanh) GLU experts
      - special_expert_pairs × (Identity + Zero) experts
    
    Total routed = polyglu_expert_multiplier * 3 + special_expert_pairs * 2
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.bias_update_factor = config.bias_update_factor
        self.bias_update_threshold = config.bias_update_threshold
        self.moe_shared_scaling = getattr(config, 'moe_shared_scaling', 1.0)
        self.load_balance_strategy = getattr(config, 'load_balance_strategy', 'none')
        self.aux_loss_coef = getattr(config, 'aux_loss_coef', 0.001)

        # Token counter + accumulated TPE for threshold-based bias updates (only for strategy="bias")
        self.register_buffer("tokens_processed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        
        # Fused experts
        self.experts = BiBoFusedExperts(config)

        # Shared expert (always active) — only if enabled
        self.use_shared_expert = getattr(config, 'use_shared_expert', True)
        self.shared_experts_list = nn.ModuleList()
        if self.use_shared_expert:
            if config.shared_expert_type == "conv":
                self.shared_experts_list.append(BiBoCausalConv1D(config))
            else:
                self.shared_experts_list.append(BiBoMLP(config, is_expert=True))
        self.gate = BiBoMoERouter(config)

    @torch.no_grad()
    def update_bias(self, tokens_per_expert: torch.Tensor):
        """Update router bias based on token distribution."""
        if not hasattr(self.gate, 'bias') or self.bias_update_factor <= 0:
            return

        tpe = tokens_per_expert.detach().float()
        if self.num_routed_experts > 0:
            mean_tpe = tpe.mean()
            deviation = mean_tpe - tpe
        else:
            deviation = torch.zeros_like(tpe)

        self.gate.bias.add_(self.bias_update_factor * deviation.sign())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        num_tokens = bsz * seq_len
        
        # Get routing decisions (now returns aux_loss and router_logits)
        top_k_indices, top_k_weights, aux_loss, router_logits = self.gate(hidden_states)

        # Bias update bookkeeping (only when load_balance_strategy="bias")
        tokens_per_expert = None
        if (self.training 
            and self.load_balance_strategy == "bias"
            and hasattr(self.gate, 'bias') 
            and self.bias_update_factor > 0):
            current_tpe = torch.bincount(
                rearrange(top_k_indices, 'b s k -> (b s k)'),
                minlength=self.num_routed_experts
            )
            
            self.tokens_processed += num_tokens
            self.accumulated_tpe += current_tpe.float()
            
            if self.tokens_processed >= self.bias_update_threshold:
                tokens_per_expert = self.accumulated_tpe.clone()
                self.tokens_processed.zero_()
                self.accumulated_tpe.zero_()

        # Flatten: (B*S, H) and (B*S, top_k)
        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
        flat_indices = rearrange(top_k_indices, 'b s k -> (b s) k')
        flat_weights = rearrange(top_k_weights, 'b s k -> (b s) k')

        # Fused expert dispatch
        final_routed = self.experts(flat_hidden, flat_indices, flat_weights)
        final_routed = rearrange(final_routed, '(b s) h -> b s h', b=bsz)
        
        # Shared expert (only if enabled)
        if self.use_shared_expert:
            shared_combined = self.shared_experts_list[0](hidden_states)
            final_output = final_routed + (self.moe_shared_scaling * shared_combined)
        else:
            final_output = final_routed

        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        # Return aux_loss and router_logits for the model to collect
        return final_output, aux_loss, router_logits
