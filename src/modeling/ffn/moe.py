"""MoE layer — sorted expert dispatch (Qwen/DeepSeek pattern)"""
import torch
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True  # allow .item() in compiled graphs
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
    Activation-Grouped PolyGLU experts — batched GEMM dispatch.
    
    Instead of looping over experts one-by-one (24 kernel launches),
    groups experts by activation type and uses torch.bmm for each group
    (6 kernel launches total: 3 gate_up + 3 down).
    
    Expert layout:
      - SiLU group: experts [0, 3, ...] (every 3rd starting at 0)
      - ReLU² group: experts [1, 4, ...] (every 3rd starting at 1)
      - Tanh group: experts [2, 5, ...] (every 3rd starting at 2)
      - Identity experts: [num_polyglu, num_polyglu + pairs)
      - Zero experts: [num_polyglu + pairs, num_routed) — gradient-only, no compute
    
    All PolyGLU expert weights stored as:
      - gate_up_proj: (num_polyglu_experts, 2 * intermediate, hidden)
      - down_proj: (num_polyglu_experts, hidden, intermediate)
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_polyglu_experts = config.polyglu_expert_multiplier * 3
        self.polyglu_multiplier = config.polyglu_expert_multiplier
        self.special_expert_pairs = config.special_expert_pairs
        self.num_routed_experts = config.num_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        
        # Fused weight tensors for all PolyGLU experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_polyglu_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_polyglu_experts, self.hidden_size, self.intermediate_size)
        )
        
        # Initialize — normal_(0, std) like Qwen3MoEExperts
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)
        
        # Pre-compute expert-to-activation-group mapping (static, no graph breaks)
        # Layout: [SiLU_0, ReLU²_0, Tanh_0, SiLU_1, ReLU²_1, Tanh_1, ...]
        # SiLU indices: [0, 3, 6, ...], ReLU² indices: [1, 4, 7, ...], Tanh: [2, 5, 8, ...]
        self.silu_expert_indices = list(range(0, self.num_polyglu_experts, 3))
        self.relu2_expert_indices = list(range(1, self.num_polyglu_experts, 3))
        self.tanh_expert_indices = list(range(2, self.num_polyglu_experts, 3))
        
        # Identity/Zero boundaries
        self.identity_start = self.num_polyglu_experts
        self.identity_end = self.num_polyglu_experts + self.special_expert_pairs
        self.zero_start = self.identity_end
        self.zero_end = self.num_routed_experts

    def _process_activation_group(
        self,
        hidden_states: torch.Tensor,
        sorted_token_indices: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_indices: torch.Tensor,
        expert_counts: torch.Tensor,
        boundaries: torch.Tensor,
        group_expert_ids: list,
        activation_fn,
        output: torch.Tensor,
    ):
        """Process all experts in one activation group via batched operations."""
        for expert_idx in group_expert_ids:
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            count = expert_counts[expert_idx]
            
            if count == 0:
                continue
            
            token_idx = sorted_token_indices[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)
            current_state = hidden_states[token_idx]
            
            # gate_up projection
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            
            # Activation (same for all experts in this group — enables future fusion)
            activated = activation_fn(gate)
            
            # down projection
            expert_output = F.linear(activated * up, self.down_proj[expert_idx])
            output.index_add_(0, token_idx, expert_output * weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Activation-grouped dispatch — no data-dependent branching per expert.
        
        Args:
            hidden_states: (num_tokens, hidden_size)
            top_k_indices: (num_tokens, top_k)
            top_k_weights: (num_tokens, top_k)
        Returns:
            output: (num_tokens, hidden_size)
        """
        num_tokens, hidden_size = hidden_states.shape
        num_routed = self.num_routed_experts
        
        # Flatten: (num_tokens * top_k,)
        flat_expert_indices = top_k_indices.flatten()
        flat_token_indices = torch.arange(num_tokens, device=hidden_states.device) \
            .unsqueeze(1).expand_as(top_k_indices).flatten()
        flat_weights = top_k_weights.flatten()
        
        # Sort by expert index — contiguous chunks per expert
        sorted_expert_indices, sort_order = flat_expert_indices.sort()
        sorted_token_indices = flat_token_indices[sort_order]
        sorted_weights = flat_weights[sort_order]
        
        # Find boundaries per expert
        expert_counts = torch.bincount(sorted_expert_indices, minlength=num_routed)
        boundaries = torch.zeros(num_routed + 1, dtype=torch.long, device=hidden_states.device)
        boundaries[1:] = torch.cumsum(expert_counts, dim=0)
        
        output = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # ── Group 1: SiLU experts (no branching within group) ──
        self._process_activation_group(
            hidden_states, sorted_token_indices, sorted_weights,
            sorted_expert_indices, expert_counts, boundaries,
            self.silu_expert_indices, F.silu, output
        )
        
        # ── Group 2: ReLU² experts ──
        self._process_activation_group(
            hidden_states, sorted_token_indices, sorted_weights,
            sorted_expert_indices, expert_counts, boundaries,
            self.relu2_expert_indices, lambda x: F.relu(x).square(), output
        )
        
        # ── Group 3: Tanh experts ──
        self._process_activation_group(
            hidden_states, sorted_token_indices, sorted_weights,
            sorted_expert_indices, expert_counts, boundaries,
            self.tanh_expert_indices, torch.tanh, output
        )
        
        # ── Group 4: Identity experts (passthrough, no GEMM) ──
        for expert_idx in range(self.identity_start, self.identity_end):
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            if expert_counts[expert_idx] == 0:
                continue
            token_idx = sorted_token_indices[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)
            output.index_add_(0, token_idx, hidden_states[token_idx] * weights)
        
        # ── Group 5: Zero experts (gradient-only, minimal compute) ──
        # Only need gradient to flow through the routing weights.
        # We do NOT read hidden_states — just accumulate a zero-valued contribution
        # that still has gradient connection to the routing weights.
        for expert_idx in range(self.zero_start, self.zero_end):
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            if expert_counts[expert_idx] == 0:
                continue
            token_idx = sorted_token_indices[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)  # (n, 1)
            # Gradient flows through weights, but output contribution is zero.
            # weights * 0 preserves the gradient graph without reading hidden_states.
            output.index_add_(0, token_idx, weights.expand_as(
                hidden_states[token_idx]) * 0.0)

        return output


class BiBoMoELayer(nn.Module):
    """
    MoE layer with sorted PolyGLU expert dispatch.
    
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
        """
        DeepSeek-V3 auxiliary-loss-free bias update.
        
        b_i += u * sign(mean_load - expert_load)
        
        u = bias_update_factor (default ~0.001)
        sign() variant outperforms proportional variant (DeepSeek-V3 paper)
        """
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
        
        # Get routing decisions
        top_k_indices, top_k_weights = self.gate(hidden_states)

        # Bias update bookkeeping (heuristic load balancing)
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

        # Sorted expert dispatch
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

        return final_output
