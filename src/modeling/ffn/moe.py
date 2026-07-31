import torch
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig
from .experts import BiBoCausalConv1D
from .mlp import BiBoMLP
from .router import BiBoMoERouter

__all__ = ['BiBoMoELayer']


_NORMSILU_EPS = 1e-6


class BiBoFusedExperts(nn.Module):
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_glu_experts = config.num_glu_experts
        self.special_expert_pairs = config.special_expert_pairs
        self.num_routed_experts = config.num_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_glu_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_glu_experts, self.hidden_size, self.intermediate_size)
        )
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)

        self.radial_theta = nn.Parameter(torch.zeros(self.num_glu_experts))

        num_pos = getattr(config, "num_pos_identity_experts", self.special_expert_pairs)
        num_neg = getattr(config, "num_neg_identity_experts", self.special_expert_pairs)
        self.pos_start = self.num_glu_experts
        self.pos_end = self.pos_start + num_pos
        self.neg_start = self.pos_end
        self.neg_end = self.neg_start + num_neg

    @torch._dynamo.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        num_routed = self.num_routed_experts

        flat_expert_indices = top_k_indices.flatten()
        flat_token_indices = torch.arange(num_tokens, device=hidden_states.device) \
            .unsqueeze(1).expand_as(top_k_indices).flatten()
        flat_weights = top_k_weights.flatten()

        sorted_expert_indices, sort_order = flat_expert_indices.sort()
        sorted_token_indices = flat_token_indices[sort_order]
        sorted_weights = flat_weights[sort_order]

        expert_counts = torch.bincount(sorted_expert_indices, minlength=num_routed)
        boundaries = torch.zeros(num_routed + 1, dtype=torch.long, device=hidden_states.device)
        boundaries[1:] = torch.cumsum(expert_counts, dim=0)
        bounds = boundaries.tolist()

        acc_dtype = torch.promote_types(torch.float32, hidden_states.dtype)
        output = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=acc_dtype)

        for expert_idx in range(num_routed):
            start = bounds[expert_idx]
            end = bounds[expert_idx + 1]

            token_idx = sorted_token_indices[start:end]
            if token_idx.shape[0] == 0:
                continue
            weights = sorted_weights[start:end].unsqueeze(-1)
            current_state = hidden_states[token_idx]

            if expert_idx < self.num_glu_experts:
                gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)

                g = gate.float()
                r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
                act32 = F.silu(g / r) * r.pow(torch.sigmoid(self.radial_theta[expert_idx].float()))
                activated = act32.to(gate.dtype)

                expert_output = F.linear(activated * up, self.down_proj[expert_idx])
                output.index_add_(0, token_idx, expert_output * weights)

            elif expert_idx < self.neg_start:
                output.index_add_(0, token_idx, current_state * weights)

            else:
                output.index_add_(0, token_idx, current_state * -weights)

        return output.to(hidden_states.dtype)


class BiBoMoELayer(nn.Module):
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.bias_update_factor = config.bias_update_factor
        self.bias_update_threshold = config.bias_update_threshold

        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        self._fwd_step = 0
        self._update_every = None
        self.experts = BiBoFusedExperts(config)

        self.use_shared_expert = getattr(config, 'use_shared_expert', False)
        self.shared_experts_list = nn.ModuleList()
        if self.use_shared_expert:
            if config.shared_expert_type == "conv":
                self.shared_experts_list.append(BiBoCausalConv1D(config))
            else:
                _n_sh = int(getattr(config, 'num_shared_experts', 1))
                self.shared_experts_list.append(BiBoMLP(
                    config, is_expert=True,
                    intermediate_size=config.moe_intermediate_size * _n_sh))
        self.gate = BiBoMoERouter(config)


    @torch.no_grad()
    def update_bias(self, tokens_per_expert: torch.Tensor):
        if self.bias_update_factor <= 0:
            return
        share = tokens_per_expert.detach().float()
        share = share / share.sum().clamp_min(1.0)
        self.gate.bias.add_(self.bias_update_factor * (share.mean() - share))

    @torch._dynamo.disable
    def _balance_step(self, top_k_indices, num_tokens):
        current_tpe = torch.bincount(
            rearrange(top_k_indices, 'b s k -> (b s k)'),
            minlength=self.num_routed_experts
        )
        self.accumulated_tpe += current_tpe.float()
        if self._update_every is None:
            self._update_every = max(1, round(self.bias_update_threshold / max(num_tokens, 1)))
        self._fwd_step += 1
        if self._fwd_step % self._update_every == 0:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self.accumulated_tpe, op=dist.ReduceOp.SUM)
            tpe = self.accumulated_tpe.clone()
            self.accumulated_tpe.zero_()
            return tpe
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        num_tokens = bsz * seq_len

        top_k_indices, top_k_weights = self.gate(hidden_states)

        tokens_per_expert = None
        if self.training and self.bias_update_factor > 0:
            tokens_per_expert = self._balance_step(top_k_indices, num_tokens)

        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
        flat_indices = rearrange(top_k_indices, 'b s k -> (b s) k')
        flat_weights = rearrange(top_k_weights, 'b s k -> (b s) k')

        final_routed = self.experts(flat_hidden, flat_indices, flat_weights)
        final_routed = rearrange(final_routed, '(b s) h -> b s h', b=bsz)

        if self.use_shared_expert:
            shared_combined = self.shared_experts_list[0](hidden_states)
            final_output = final_routed + shared_combined
        else:
            final_output = final_routed

        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output
