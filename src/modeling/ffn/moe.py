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
from .triton_moe_kernel import fused_moe_activation_group

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

    def _pytorch_dispatch_group_fast(
        self,
        hidden_states: torch.Tensor,
        sorted_token_indices: torch.Tensor,
        sorted_weights: torch.Tensor,
        boundaries_list: list,
        expert_counts_list: list,
        group_expert_ids: list,
        act_fn,
        output: torch.Tensor,
    ):
        """Fastest PyTorch dispatch — uses pre-computed Python lists, zero GPU sync."""
        for expert_idx in group_expert_ids:
            if expert_counts_list[expert_idx] == 0:
                continue
            start = boundaries_list[expert_idx]
            end = boundaries_list[expert_idx + 1]
            
            token_idx = sorted_token_indices[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)
            current_state = hidden_states[token_idx]
            
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = act_fn(gate)
            expert_output = F.linear(activated * up, self.down_proj[expert_idx])
            output.index_add_(0, token_idx, expert_output * weights)

    def _dispatch_activation_group(
        self,
        hidden_states: torch.Tensor,
        sorted_token_indices: torch.Tensor,
        sorted_weights: torch.Tensor,
        boundaries: torch.Tensor,
        boundaries_cpu: torch.Tensor,
        expert_counts: torch.Tensor,
        expert_counts_cpu: torch.Tensor,
        group_expert_ids: list,
        activation: str,
        output: torch.Tensor,
    ):
        """
        Dispatch one activation group via fused_moe_activation_group kernel.
        
        On Linux (Triton): builds contiguous buffers for the kernel.
        On Windows (PyTorch): passes boundaries directly to avoid overhead.
        """
        from .triton_moe_kernel import HAS_TRITON
        
        if HAS_TRITON and hidden_states.is_cuda:
            # Triton path: needs contiguous per-group buffers
            num_in_group = len(group_expert_ids)
            group_offsets = torch.zeros(num_in_group + 1, dtype=torch.long, device=hidden_states.device)
            
            group_token_ids_list = []
            group_weights_list = []
            
            for i, expert_idx in enumerate(group_expert_ids):
                start = boundaries_cpu[expert_idx].item()
                end = boundaries_cpu[expert_idx + 1].item()
                count = end - start
                group_offsets[i + 1] = group_offsets[i] + count
                if count > 0:
                    group_token_ids_list.append(sorted_token_indices[start:end])
                    group_weights_list.append(sorted_weights[start:end])
            
            if group_offsets[-1].item() == 0:
                return
            
            group_token_ids = torch.cat(group_token_ids_list)
            group_weights = torch.cat(group_weights_list)
            group_gate_up = self.gate_up_proj[group_expert_ids]
            group_down = self.down_proj[group_expert_ids]
            
            fused_moe_activation_group(
                hidden_states, output,
                group_gate_up, group_down,
                group_token_ids, group_weights,
                group_offsets, activation
            )
        else:
            # PyTorch fast path: use boundaries directly, no buffer allocation
            self._pytorch_dispatch_group(
                hidden_states, sorted_token_indices, sorted_weights,
                boundaries_cpu, expert_counts_cpu, group_expert_ids, activation, output
            )

    def _pytorch_dispatch_group(
        self,
        hidden_states: torch.Tensor,
        sorted_token_indices: torch.Tensor,
        sorted_weights: torch.Tensor,
        boundaries: torch.Tensor,
        expert_counts: torch.Tensor,
        group_expert_ids: list,
        activation: str,
        output: torch.Tensor,
    ):
        """Direct PyTorch dispatch — no buffer allocation, uses boundaries directly."""
        if activation == "silu":
            act_fn = F.silu
        elif activation == "relu2":
            act_fn = lambda x: F.relu(x).square()
        else:
            act_fn = torch.tanh
        
        # Move boundaries to CPU once to avoid per-expert GPU sync
        boundaries_cpu = boundaries.cpu()
        expert_counts_cpu = expert_counts.cpu()
        
        for expert_idx in group_expert_ids:
            if expert_counts_cpu[expert_idx].item() == 0:
                continue
            
            start = boundaries_cpu[expert_idx].item()
            end = boundaries_cpu[expert_idx + 1].item()
            
            token_idx = sorted_token_indices[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)
            current_state = hidden_states[token_idx]
            
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = act_fn(gate)
            expert_output = F.linear(activated * up, self.down_proj[expert_idx])
            output.index_add_(0, token_idx, expert_output * weights)

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Activation-grouped dispatch via fused kernels.
        
        NOTE: @torch.compiler.disable is required because expert dispatch has
        inherently dynamic shapes (each expert gets a different number of tokens).
        Inductor's reinplace_inplaceable_ops pass crashes on the variable-size
        index_add_ operations. Attention + dense layers are still compiled.
        
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
        
        # ── All expert dispatch — no CPU sync, no shape checks ──
        # Empty slices (when expert has 0 tokens) produce empty tensors.
        # F.linear and index_add_ handle empty tensors correctly (no-op).
        
        # ── Group 1: SiLU experts ──
        for expert_idx in self.silu_expert_indices:
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            token_idx = sorted_token_indices[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            x = hidden_states[token_idx]
            gu = F.linear(x, self.gate_up_proj[expert_idx])
            g, u = gu.chunk(2, dim=-1)
            out_e = F.linear(F.silu(g) * u, self.down_proj[expert_idx])
            output.index_add_(0, token_idx, out_e * w)
        
        # ── Group 2: ReLU² experts ──
        for expert_idx in self.relu2_expert_indices:
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            token_idx = sorted_token_indices[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            x = hidden_states[token_idx]
            gu = F.linear(x, self.gate_up_proj[expert_idx])
            g, u = gu.chunk(2, dim=-1)
            out_e = F.linear(F.relu(g).square() * u, self.down_proj[expert_idx])
            output.index_add_(0, token_idx, out_e * w)
        
        # ── Group 3: Tanh experts ──
        for expert_idx in self.tanh_expert_indices:
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            token_idx = sorted_token_indices[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            x = hidden_states[token_idx]
            gu = F.linear(x, self.gate_up_proj[expert_idx])
            g, u = gu.chunk(2, dim=-1)
            out_e = F.linear(torch.tanh(g) * u, self.down_proj[expert_idx])
            output.index_add_(0, token_idx, out_e * w)
        
        # ── Group 4: Identity experts ──
        for expert_idx in range(self.identity_start, self.identity_end):
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            token_idx = sorted_token_indices[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            output.index_add_(0, token_idx, hidden_states[token_idx] * w)
        
        # ── Group 5: Zero experts (gradient-only) ──
        for expert_idx in range(self.zero_start, self.zero_end):
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            token_idx = sorted_token_indices[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            output.index_add_(0, token_idx, w.expand_as(hidden_states[token_idx]) * 0.0)

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
