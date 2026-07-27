"""MoE router — MiMo-V2.5 / DeepSeek-V3 auxiliary-loss-free sigmoid gating (verbatim routing)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoMoERouter']


class BiBoMoERouter(nn.Module):
    """MiMo-V2.5 / DeepSeek-V3 aux-loss-free gate (arXiv:2408.15664, arXiv:2412.19437).

    Diverges from MiMo in one place: `norm_topk_prob` normalizes the top-k weights by SOFTMAX,
    not ÷sum. ÷sum is only valid for non-negative scores — with `gate_type="situ"` the top-k sum
    can cross zero (division explodes) or go negative (all weights flip sign). Softmax has
    neither failure. Cost: it is flatter than ÷sum (sigmoid scores span <1, so the max/min
    weight ratio is bounded by e).
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok

        # getattr fallbacks must MATCH the BiBoConfig defaults, or a stale config silently flips
        # behavior (norm_topk_prob especially — its config default is True).
        self.router_activation = getattr(config, 'router_activation', 'none')
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.gate_type = getattr(config, 'gate_type', 'sigmoid')
        self.router_temperature = float(getattr(config, 'router_temperature', 1.0))
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)

        self.router_type = getattr(config, 'router_type', 'mlp')
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size

        # Heuristically updated by BiBoMoELayer, NOT optimizer-managed -> requires_grad=False.
        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts), requires_grad=False)

        self._probe_gap = False          # see forward(); harness diagnostic, zero cost when off
        self.boundary_gap = None

        if self.router_type == "mlp":
            # (num_routed_experts, hidden_size) — experts are the ROW dim.
            self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=config.initializer_range)
        elif self.router_type == "conv":
            # ┌─ WHY THIS IS STORED 2D (E, H*K) AND NOT AS AN nn.Conv1d ─────────────────────────┐
            # A Conv1d weight is (E, H, K) — 3D. Muon's Newton-Schulz treats the LEADING dim as a
            # BATCH and orthogonalizes each trailing 2D slice, iterating on the smaller Gram. With
            # (E, H, K) the Gram is (K,K), so NS decorrelates the K TAPS of each expert and leaves
            # the EXPERTS correlated — the exact opposite of what a router needs. Measured: fed
            # experts collapsed onto one direction, the 3D path returns |cos| 0.9999 -> 0.9999, i.e.
            # it CANNOT de-collapse experts, while the MLP router's 2D (E,H) path gives 0.9999 -> 0.0.
            # Over 300 real steps the 3D layout drove expert correlation up 13x faster than the MLP
            # router (dxcos +0.046 vs +0.0035).
            # Storing the weight 2D as (E, H*K) makes the Gram (E,E), so NS decorrelates EXPERTS —
            # identical semantics to the MLP router — and it does so BY CONSTRUCTION: no optimizer
            # flag, no param-group surgery, and it stays correct under the fused Muon unchanged.
            # It also keeps the weight out of any "3D => expert stack" bucket (e.g. xorth).
            # DO NOT "simplify" this back to nn.Conv1d. See ablate/common/optim.py for the full rule.
            # └──────────────────────────────────────────────────────────────────────────────────┘
            self.gate_conv = nn.Parameter(
                torch.empty(self.num_routed_experts, config.hidden_size * self.kernel_size)
            )
            # FAN-IN AWARE: a conv logit sums H*K terms vs the MLP's H, so sharing `initializer_range`
            # would start the conv router sqrt(K)x sharper (measured 1.64x at K=3) — which silently
            # confounded every historical conv-vs-mlp comparison. Divide by sqrt(K).
            nn.init.normal_(self.gate_conv, mean=0.0,
                            std=config.initializer_range / math.sqrt(self.kernel_size))
        else:
            raise ValueError(f"router_type must be 'mlp' or 'conv', got '{self.router_type}'")

    def _apply_router_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.router_activation == "relu":
            return F.relu(logits)
        elif self.router_activation == "silu":
            return F.silu(logits)
        else:  # "none"
            return logits

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(b, s, h) -> ((b s), num_routed_experts) fp32 logits, before gate_type."""
        if self.router_type == "mlp":
            return self.gate_proj(rearrange(hidden_states, 'b s h -> (b s) h')).float()
        # causal: left-pad K-1 so position t sees only t-K+1..t. view() is free (weight is contiguous).
        x = F.pad(rearrange(hidden_states, 'b s h -> b h s'), (self.kernel_size - 1, 0))
        w = self.gate_conv.view(self.num_routed_experts, self.hidden_size, self.kernel_size)
        return rearrange(F.conv1d(x, w), 'b e s -> (b s) e').float()

    def forward(self, hidden_states: torch.Tensor):
        """(b, s, h) -> top_k_indices (b, s, k), norm_weights (b, s, k). Weights are UNBIASED."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        router_logits = self._apply_router_activation(self.router_logits(hidden_states))
        # Temperature BEFORE the gate: sigma(x/T) has derivative sigma'(x/T)/T, so T>1 flattens the
        # score spread by exactly 1/T. Applied to logits, not scores, so the selection bias (added
        # to SCORES) is untouched.
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature

        if self.gate_type == "sigmoid":
            scores = torch.sigmoid(router_logits)
        elif self.gate_type == "situ":
            # SiTU = sigmoid(x)*tanh(x). Range ~(-0.2785, 1), min at x ≈ -0.78, and NON-MONOTONIC
            # in the logit: f(-5) ≈ -0.0067 > f(-0.78) ≈ -0.2785, so a strongly-rejected expert can
            # outrank a mildly-rejected one in the top-k ordering.
            scores = torch.sigmoid(router_logits) * torch.tanh(router_logits)
        elif self.gate_type == "softmax":
            scores = F.softmax(router_logits, dim=1)
        else:
            raise ValueError(
                f"gate_type must be 'sigmoid', 'situ', or 'softmax', got '{self.gate_type}'"
            )

        # Diagnostic, OFF unless a harness turns it on. Mean rank-k minus rank-(k+1) RAW score gap:
        # the SELECTION-BOUNDARY gap. This is the quantity the load-balancing bias actually competes
        # against -- b is added to `scores` and only reorders a token when it closes this gap -- so
        # it, not the top-1/top-2 split, sets how many tokens one +-u step can flip (the balancer's
        # control authority). Kept on-device, no sync; ablate's RouterTrace reads it post-forward.
        if self._probe_gap and self.top_k < self.num_routed_experts:
            with torch.no_grad():
                _tk = scores.topk(self.top_k + 1, dim=-1).values
                self.boundary_gap = (_tk[..., self.top_k - 1] - _tk[..., self.top_k]).mean()

        # Bias is SELECTION-ONLY. The combine weights below come from raw `scores`, never from
        # `selection_scores` — mixing them up silently breaks aux-loss-free balancing.
        selection_scores = scores + self.bias
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False)

        top_k_weights = scores.gather(-1, top_k_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            norm_weights = F.softmax(top_k_weights, dim=-1)
        else:
            norm_weights = top_k_weights
        norm_weights = norm_weights * self.routed_scaling_factor

        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        # fp32 end-to-end (logits -> gate -> norm -> here); the MoE combine accumulates in fp32.
        return top_k_indices.long(), norm_weights.float()
