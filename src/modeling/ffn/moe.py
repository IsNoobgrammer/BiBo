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


_POLYGLU_ACTIVATIONS = ("silu", "relu2", "normsilu")   # expert e uses [e % 3]
_NORMSILU_EPS = 1e-6  # MUST match _NS_EPS in the tkf kernel (kernels/sm75/moe.py)


class BiBoFusedExperts(nn.Module):
    """Fused PolyGLU experts, sorted dispatch: sort tokens by expert, process contiguous chunks,
    index_add_ back. Expert layout is GLU block, then +Identity block, then -Identity block; a disabled
    special type is simply a zero-width block. The ±Identity specials cost no GEMM.
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_polyglu_experts = config.polyglu_expert_multiplier * 3
        self.special_expert_pairs = config.special_expert_pairs
        self.num_routed_experts = config.num_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_polyglu_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_polyglu_experts, self.hidden_size, self.intermediate_size)
        )
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=config.initializer_range)

        num_pos = getattr(config, "num_pos_identity_experts", self.special_expert_pairs)
        num_neg = getattr(config, "num_neg_identity_experts", self.special_expert_pairs)
        self.pos_start = self.num_polyglu_experts
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
        """(num_tokens, hidden), (num_tokens, k), (num_tokens, k) -> (num_tokens, hidden).

        @torch._dynamo.disable: tokens-per-expert varies every step, so compiling this recompiles
        endlessly.
        """
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
        bounds = boundaries.tolist()   # ONE device->host sync; slicing per expert would sync N times

        # fp32 combine accumulator (MiMo-style), cast back at the end. promote_types keeps fp32 for
        # fp16/bf16/fp32 and only widens to fp64 for an fp64 model, where index_add_ would otherwise
        # hit a Double-source vs Float-buffer mismatch.
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

            if expert_idx < self.num_polyglu_experts:
                gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)

                act_name = _POLYGLU_ACTIVATIONS[expert_idx % 3]
                if act_name == "silu":
                    activated = F.silu(gate)
                elif act_name == "relu2":
                    r = F.relu(gate)
                    activated = (r.float() * r.float()).to(gate.dtype)  # fp32: fp16 overflows >256
                else:
                    # normsilu = SiLU(gate / rms(gate)) — DECO's intra-expert stage. The
                    # inter-expert mean stage is deliberately skipped: its gradient couples every
                    # expert's up-weights and breaks per-expert dispatch.
                    g = gate.float()
                    g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
                    activated = F.silu(g).to(gate.dtype)

                expert_output = F.linear(activated * up, self.down_proj[expert_idx])
                output.index_add_(0, token_idx, expert_output * weights)

            elif expert_idx < self.neg_start:
                output.index_add_(0, token_idx, current_state * weights)    # +Identity

            else:
                output.index_add_(0, token_idx, current_state * -weights)   # -Identity

        return output.to(hidden_states.dtype)


class BiBoMoELayer(nn.Module):
    """Routed = polyglu_expert_multiplier*3 GLU experts + special_expert_pairs*2 ±Identity."""
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.bias_update_factor = config.bias_update_factor
        self.bias_update_threshold = config.bias_update_threshold
        self.load_balance_strategy = getattr(config, 'load_balance_strategy', 'none')
        self.balance_exclude_specials = getattr(config, 'balance_exclude_specials', False)
        self.glu_token_budget = getattr(config, 'glu_token_budget', None)
        self.bias_update_mode = getattr(config, 'bias_update_mode', 'sign')

        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        # Trigger on FORWARD STEPS, not device token counts: every rank then fires update_bias on the
        # same step, so the all_reduce below can never desync. Host-side ints -> no per-step .item().
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
        """Aux-loss-free bias update: b_i += u * sign(target_i - load_i). Two target rules:

        DeepSeek-V3 (glu_token_budget=None, the default) — target is the OBSERVED MEAN of the
        balanced block. This equalizes experts WITHIN the block but says nothing about how much
        traffic the block gets in total, so with `balance_exclude_specials` the GLU-vs-special split
        is left entirely to the router.

        LongCat-Flash (glu_token_budget=r, arXiv:2509.01322 eq. for Delta b_i) — target is the
        ABSOLUTE budget share r/n_glu, measured against T_i/(k*T_all), and the specials get
        Delta b = 0 (they are the residual sink that absorbs whatever the GLU block sheds):

            Delta b_i = u * sign( r/n_glu - T_i / (k*T_all) )   for the GLU block
            Delta b_i = 0                                        for the +/-Identity specials

        So when the GLU block is collectively over budget EVERY GLU bias drops by the same u --
        a uniform shift that cannot change the ordering inside the block but does push tokens
        across to the specials. That is the knob the mean-relative rule structurally lacks:
        r is LongCat's K_e/K, the fraction of the k slots per token that should land on real FFNs.

        bias_update_mode: "sign" (DeepSeek-V3, default) | "prop" (LongCat, raw deviation).
        LongCat's equation has NO sign() -- it applies mu * (target - actual) directly. That is
        proportional control instead of bang-bang, and it fixes two defects we measured:
          1. NO DITHER / IT HAS A FIXED POINT. sign() never returns 0, so the bias oscillates +-u
             forever and u is a permanent routing-noise floor. Measured: u=0.01 hit its load target
             (spl 0.160, balance 0.9973) yet lost 0.094 of loss to the noise -- 10x the 0.009
             replicate floor. Proportional steps shrink as the deviation closes, so it settles.
          2. NO COMMON-MODE DRIFT. Raw deviations sum to EXACTLY zero over the balanced block
             (sum_i(mean - x_i) = 0), so the block's mean bias cannot move. sign()ed deviations do
             NOT sum to zero -- a right-skewed load puts most experts below the mean, so most get
             +1 and the whole block floats up. Measured with balance_exclude_specials at u=0.01:
             GLU biases at +1.28 while the frozen specials sat at 0, i.e. ~84% of the accumulated
             bias was common-mode. That silently turned u into a GLU-vs-special preference knob.
        u is NOT comparable across modes: "sign" steps by u, "prop" steps by u * deviation, and
        share deviations run ~1e-3..1e-1, so prop needs a u roughly 1-2 orders of magnitude larger.
        """
        if not hasattr(self.gate, 'bias') or self.bias_update_factor <= 0:
            return

        tpe = tokens_per_expert.detach().float()
        deviation = torch.zeros_like(tpe)
        n_glu = self.experts.num_polyglu_experts
        # Deviations are always in SHARE units (T_i / (k*T_all)), never raw counts, so the update is
        # invariant to batch size and to how many steps we accumulate over. In "sign" mode only the
        # sign survives so this rescaling is a no-op; in "prop" mode it is what makes u portable.
        share = tpe / tpe.sum().clamp_min(1.0)

        if self.glu_token_budget is not None:
            if n_glu > 0:
                deviation[:n_glu] = self.glu_token_budget / n_glu - share[:n_glu]
        else:
            n_balanced = n_glu if self.balance_exclude_specials else self.num_routed_experts
            if n_balanced > 0:
                deviation[:n_balanced] = share[:n_balanced].mean() - share[:n_balanced]

        if self.bias_update_mode == "prop":
            self.gate.bias.add_(self.bias_update_factor * deviation)
        else:
            self.gate.bias.add_(self.bias_update_factor * deviation.sign())

    @torch._dynamo.disable
    def _balance_step(self, top_k_indices, num_tokens):
        """Accumulate per-expert counts; every _update_every forwards return the global summed counts
        and reset, else None. Dynamo-disabled: the Python `_fwd_step` counter would otherwise be baked
        in as a compile-time constant and recompile the forward every step.
        """
        current_tpe = torch.bincount(
            rearrange(top_k_indices, 'b s k -> (b s k)'),
            minlength=self.num_routed_experts
        )
        self.accumulated_tpe += current_tpe.float()
        if self._update_every is None:
            self._update_every = max(1, round(self.bias_update_threshold / max(num_tokens, 1)))
        self._fwd_step += 1
        if self._fwd_step % self._update_every == 0:
            # SUM is fine because sign() is scale-invariant; the bias then balances GLOBAL load.
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
        if (self.training
            and self.load_balance_strategy == "bias"
            and hasattr(self.gate, 'bias')
            and self.bias_update_factor > 0):
            tokens_per_expert = self._balance_step(top_k_indices, num_tokens)

        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
        flat_indices = rearrange(top_k_indices, 'b s k -> (b s) k')
        flat_weights = rearrange(top_k_weights, 'b s k -> (b s) k')

        final_routed = self.experts(flat_hidden, flat_indices, flat_weights)
        final_routed = rearrange(final_routed, '(b s) h -> b s h', b=bsz)

        if self.use_shared_expert:
            shared_combined = self.shared_experts_list[0](hidden_states)
            final_output = final_routed + shared_combined   # direct add (DeepSeek-V3/Gemma)
        else:
            final_output = final_routed

        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output
