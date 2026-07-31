"""MoE layer — sorted expert dispatch (Qwen/DeepSeek pattern)"""
import torch
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True  # allow .item() in compiled graphs
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig, POLYGLU_GROUP
from .experts import BiBoCausalConv1D
from .mlp import BiBoMLP
from .router import BiBoMoERouter
from ..norm import BiBoRMSNorm

__all__ = ['BiBoMoELayer']


# RADIAL NormSiLU is the activation (Jul 31 2026). The old PolyGLU menu -- a per-expert CYCLE over
# (silu, relu2, normsilu) -- is retired: at 1B tokens radial beat every alternative on bpb
# (0.64313 vs silu-a 0.64429, normsilu 0.64646, silu 0.64768, against a 0.00037 same-seed floor)
# and the mixing rounds never found a cycle that beat the best single act. Kept as a 3-tuple so
# `[e % 3]` indexing and every caller are unchanged; all three entries are the same activation.
# The EAGER path below now implements radial too (it reads `situ_alpha` for theta), so eager and
# the Triton path agree. Eager is slow but real: it is what runs on CPU, in reference checks, and
# anywhere Triton is unavailable. ReLU^2 dropped with the rest of the retired menu.
_POLYGLU_ACTIVATIONS = ("radial", "radial", "radial")
_NORMSILU_EPS = 1e-6  # MUST match _NS_EPS in the tkf kernel (kernels/sm75/moe.py)


class BiBoFusedExperts(nn.Module):
    """Fused PolyGLU experts, sorted dispatch: sort tokens by expert, process contiguous chunks,
    index_add_ back. Expert layout is GLU block, then +Identity block, then -Identity block; a disabled
    special type is simply a zero-width block. The ±Identity specials cost no GEMM.
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_polyglu_experts = config.polyglu_expert_multiplier * POLYGLU_GROUP
        self.special_expert_pairs = config.special_expert_pairs
        self.num_routed_experts = config.num_routed_experts
        # LatentMoE: when moe_latent_dim is set the experts live in the LATENT space, so their
        # in/out width is d, not hidden_size. Everything downstream keys off self.hidden_size --
        # including the fused Triton kernel, which just sees a smaller H -- so no kernel change.
        self.hidden_size = getattr(config, "moe_latent_dim", 0) or config.hidden_size
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
                else:
                    # normsilu = SiLU(gate / rms(gate)) — DECO's intra-expert stage. The
                    # inter-expert mean stage is deliberately skipped: its gradient couples every
                    # expert's up-weights and breaks per-expert dispatch.
                    # radial  = r^p * SiLU(gate / r), r = rms(gate), p = sigmoid(theta) in (0,1).
                    #   p -> 0 IS normsilu and p -> 1 is full magnitude, so the SAME two lines serve
                    #   both codes; radial just scales the normsilu result by r^p.
                    g = gate.float()
                    r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
                    act32 = F.silu(g / r)
                    if act_name == "radial":
                        theta = getattr(self, "situ_alpha", None)
                        if theta is None:
                            raise RuntimeError(
                                "eager radial needs the per-expert exponent theta, which lives in "
                                "`situ_alpha` -- call patches.add_situ_params(model) before the "
                                "forward (train.py auto-enables it for act code 8). Without it the "
                                "layer would silently run as normsilu, radial's p->0 floor.")
                        # keep theta in the graph: this is how the exponent LEARNS on the eager path
                        act32 = act32 * r.pow(torch.sigmoid(theta[expert_idx].float()))
                    activated = act32.to(gate.dtype)

                expert_output = F.linear(activated * up, self.down_proj[expert_idx])
                output.index_add_(0, token_idx, expert_output * weights)

            elif expert_idx < self.neg_start:
                output.index_add_(0, token_idx, current_state * weights)    # +Identity

            else:
                output.index_add_(0, token_idx, current_state * -weights)   # -Identity

        return output.to(hidden_states.dtype)


class BiBoMoELayer(nn.Module):
    """Routed = polyglu_expert_multiplier*POLYGLU_GROUP GLU experts + special_expert_pairs*2 ±Identity."""
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # LatentMoE (arXiv 2601.18089; Kimi K3 ships it as routed_expert_hidden_size 3584 of 7168,
        # i.e. ratio 1/2, with latent_moe_use_norm=true). Shared W_down BEFORE dispatch and W_up
        # AFTER combine wrap the routed path, so the experts run at width d instead of hidden_size.
        # The ROUTER still reads the FULL hidden state -- the paper is explicit about that, and our
        # router has been the most sensitive part of this model.
        self.moe_latent_dim = getattr(config, 'moe_latent_dim', 0) or 0
        if self.moe_latent_dim:
            d = self.moe_latent_dim
            self.latent_down = nn.Linear(config.hidden_size, d, bias=False)
            self.latent_up = nn.Linear(d, config.hidden_size, bias=False)
            nn.init.normal_(self.latent_down.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(self.latent_up.weight, mean=0.0, std=config.initializer_range)
            # RMSNorm on the LATENT before the up-projection. Distinct from moe_out_norm, which
            # normalizes the full-width output and cost 0.010 bpb -- but that was measured with no
            # bottleneck, and K3 runs this one at 2.8T, so it is a flag defaulting to the K3 setting.
            self.latent_norm = (BiBoRMSNorm(d, eps=config.rms_norm_eps)
                                if getattr(config, 'latent_moe_use_norm', True) else None)
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

        # Per-token norm on the BLOCK OUTPUT, applied to the combined expert sum just before the
        # residual add. Top-k weight normalization bounds the WEIGHTS but not the experts' own
        # output magnitudes, so the branch magnitude still varies per token; this pins it directly.
        # "unit" is gain-free -> only the DIRECTION of the mixture reaches the residual stream.
        self.moe_out_norm = getattr(config, 'moe_out_norm', 'none')
        self.out_norm = (BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                         if self.moe_out_norm == 'rms' else None)

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

        # LatentMoE: compress AFTER the router (which read the full hidden state above), so only
        # the expert path pays the bottleneck.
        if self.moe_latent_dim:
            flat_hidden = self.latent_down(flat_hidden)

        final_routed = self.experts(flat_hidden, flat_indices, flat_weights)

        if self.moe_latent_dim:
            if self.latent_norm is not None:
                final_routed = self.latent_norm(final_routed)
            final_routed = self.latent_up(final_routed)      # back to hidden_size before the add
        final_routed = rearrange(final_routed, '(b s) h -> b s h', b=bsz)

        if self.use_shared_expert:
            shared_combined = self.shared_experts_list[0](hidden_states)
            final_output = final_routed + shared_combined   # direct add (DeepSeek-V3/Gemma)
        else:
            final_output = final_routed

        if self.moe_out_norm == 'rms':
            final_output = self.out_norm(final_output)
        elif self.moe_out_norm == 'unit':                      # gain-free: direction only
            _v = final_output.float().pow(2).mean(-1, keepdim=True)
            final_output = (final_output.float() * torch.rsqrt(_v + _NORMSILU_EPS)).to(hidden_states.dtype)

        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output
