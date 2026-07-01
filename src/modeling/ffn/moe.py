"""MoE layer — sorted expert dispatch (Qwen/DeepSeek pattern)"""
import torch
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True  # allow .item() in compiled graphs
import torch.distributed as dist
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
    Fused PolyGLU experts — sorted dispatch (Qwen/DeepSeek pattern).
    
    All PolyGLU expert weights stored as:
      - gate_up_proj: (num_polyglu_experts, 2 * intermediate, hidden)
      - down_proj: (num_polyglu_experts, hidden, intermediate)
    
    Dispatch: sort tokens by expert, process contiguous chunks, index_add_ back.
    No one_hot — no permute — no torch.where.
    
    Identity/Zero experts handled with zero compute (just index_add).
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_polyglu_experts = config.polyglu_expert_multiplier * 3
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
        
        # Pre-compute activation name for each expert index
        # Layout: [SiLU_0, ReLU²_0, Tanh_0, SiLU_1, ReLU²_1, Tanh_1, ...]
        self._expert_activations = []
        for _ in range(config.polyglu_expert_multiplier):
            for act in _POLYGLU_ACTIVATIONS:
                self._expert_activations.append(act)
        
        # Identity indices: [num_polyglu, num_polyglu + pairs)
        # Zero indices: [num_polyglu + pairs, num_routed)
        self.identity_start = self.num_polyglu_experts
        self.identity_end = self.num_polyglu_experts + self.special_expert_pairs
        self.zero_start = self.identity_end
        self.zero_end = self.num_routed_experts

    @torch._dynamo.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sorted dispatch — no one_hot, no permute, no where.
        
        @torch._dynamo.disable: Expert dispatch has fundamentally dynamic shapes
        (variable tokens per expert per step). torch.compile recompiles endlessly.
        
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
        
        # Find boundaries: which chunks of sorted_indices belong to which expert
        # bincount gives us the count per expert, cumsum gives boundaries
        expert_counts = torch.bincount(sorted_expert_indices, minlength=num_routed)
        boundaries = torch.zeros(num_routed + 1, dtype=torch.long, device=hidden_states.device)
        boundaries[1:] = torch.cumsum(expert_counts, dim=0)
        
        # Process each expert's chunk. fp32 accumulator (MiMo-style): weights are fp32 and the
        # weighted index_add accumulates in fp32, cast back to hidden dtype at the end — keeps the
        # whole router→combine path fp32 for precision.
        output = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=torch.float32)
        
        for expert_idx in range(num_routed):
            # Use tensor indexing instead of .item() to avoid graph breaks
            start = boundaries[expert_idx]
            end = boundaries[expert_idx + 1]
            
            token_idx = sorted_token_indices[start:end]
            if token_idx.shape[0] == 0:
                continue
            weights = sorted_weights[start:end].unsqueeze(-1)  # (n, 1)
            current_state = hidden_states[token_idx]  # (n, hidden)
            
            if expert_idx < self.num_polyglu_experts:
                # PolyGLU expert: fused gate+up → activation → down
                gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                
                act_name = self._expert_activations[expert_idx]
                if act_name == "silu":
                    activated = F.silu(gate)
                elif act_name == "relu2":
                    activated = F.relu(gate).square()
                else:  # tanh
                    activated = torch.tanh(gate)
                
                expert_output = F.linear(activated * up, self.down_proj[expert_idx])
                output.index_add_(0, token_idx, expert_output * weights)
                
            elif expert_idx < self.zero_start:
                # Identity expert: pass through with weight
                output.index_add_(0, token_idx, current_state * weights)
                
            else:
                # Zero expert: skip (output is zero, router gradients don't depend on expert output)
                pass

        return output.to(hidden_states.dtype)


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
        self.load_balance_strategy = getattr(config, 'load_balance_strategy', 'none')

        # Token counter + accumulated TPE for threshold-based bias updates (only for strategy="bias")
        self.register_buffer("tokens_processed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        
        # Fused experts
        self.experts = BiBoFusedExperts(config)

        # Shared expert (always active) — only if enabled
        self.use_shared_expert = getattr(config, 'use_shared_expert', False)  # OFF by default (match Qwen)
        self.shared_experts_list = nn.ModuleList()
        if self.use_shared_expert:
            if config.shared_expert_type == "conv":
                self.shared_experts_list.append(BiBoCausalConv1D(config))
            else:
                self.shared_experts_list.append(BiBoMLP(config, is_expert=True))
            # moe_shared_scaling DEPRECATED — the shared expert is added directly (DeepSeek-V3/Gemma),
            # no learned or MC-estimated scalar.
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
                # DDP: each rank only counts its own data shard. All-reduce the per-expert token
                # counts so the bias balances on the GLOBAL load (all ranks). sign()-based update
                # (see update_bias) is scale-invariant, so SUM is fine. All ranks hit the threshold
                # the same step (identical per-step token count) → this collective stays in lockstep.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(self.accumulated_tpe, op=dist.ReduceOp.SUM)
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
            final_output = final_routed + shared_combined          # direct add (DeepSeek-V3/Gemma)
        else:
            final_output = final_routed

        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output
