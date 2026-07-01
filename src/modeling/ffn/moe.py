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
        
        # Activation for expert e is _POLYGLU_ACTIVATIONS[e % 3]
        # (layout: [SiLU_0, ReLU²_0, Tanh_0, SiLU_1, ReLU²_1, Tanh_1, ...]) — single source of truth.

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
        bounds = boundaries.tolist()   # ONE device->host sync; avoids a sync per expert on the slices

        # Process each expert's chunk. fp32 accumulator (MiMo-style): weights are fp32 and the
        # weighted index_add accumulates in >=fp32, cast back to hidden dtype at the end — keeps the
        # whole router→combine path fp32 for precision. promote_types keeps fp32 for fp16/bf16/fp32
        # (the training dtypes) and only widens to fp64 if the model itself runs fp64 (else index_add_
        # would hit a Double-source vs Float-buffer dtype mismatch).
        acc_dtype = torch.promote_types(torch.float32, hidden_states.dtype)
        output = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=acc_dtype)
        
        for expert_idx in range(num_routed):
            start = bounds[expert_idx]
            end = bounds[expert_idx + 1]

            token_idx = sorted_token_indices[start:end]
            if token_idx.shape[0] == 0:
                continue
            weights = sorted_weights[start:end].unsqueeze(-1)  # (n, 1)
            current_state = hidden_states[token_idx]  # (n, hidden)
            
            if expert_idx < self.num_polyglu_experts:
                # PolyGLU expert: fused gate+up → activation → down
                gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                
                act_name = _POLYGLU_ACTIVATIONS[expert_idx % 3]
                if act_name == "silu":
                    activated = F.silu(gate)
                elif act_name == "relu2":
                    r = F.relu(gate)
                    activated = (r.float() * r.float()).to(gate.dtype)  # fp32 square: fp16 overflow guard
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

        # Accumulated per-expert token counts for threshold-based bias updates (strategy="bias").
        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        # DDP-safe trigger state (plain ints, host-side — no CUDA buffer, no per-step .item() sync).
        # Counting FORWARD STEPS (identical across ranks in lockstep) instead of device tokens means
        # every rank fires update_bias on the SAME step → the all_reduce never desyncs/hangs.
        self._fwd_step = 0
        self._update_every = None   # step interval, derived once from the token threshold
        
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
            self.accumulated_tpe += current_tpe.float()   # in-place, no host sync

            # Convert the token threshold to a step interval once (num_tokens is uniform across DDP
            # ranks in the packed pipeline, so every rank derives the same interval).
            if self._update_every is None:
                self._update_every = max(1, round(self.bias_update_threshold / max(num_tokens, 1)))
            self._fwd_step += 1
            if self._fwd_step % self._update_every == 0:
                # All-reduce the per-expert counts so the bias balances GLOBAL load (sign()-based
                # update is scale-invariant → SUM is fine). Step-based trigger => all ranks are here
                # on the same step => the collective stays in lockstep (no ragged-batch deadlock).
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(self.accumulated_tpe, op=dist.ReduceOp.SUM)
                tokens_per_expert = self.accumulated_tpe.clone()
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
