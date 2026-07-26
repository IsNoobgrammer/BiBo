"""MoE router — MiMo-V2.5 / DeepSeek-V3 auxiliary-loss-free sigmoid gating (verbatim routing)."""
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
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)

        # Heuristically updated by BiBoMoELayer, NOT optimizer-managed -> requires_grad=False.
        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts), requires_grad=False)

        # (num_routed_experts, hidden_size) — experts are the ROW dim.
        self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=config.initializer_range)

    def _apply_router_activation(self, logits: torch.Tensor) -> torch.Tensor:
        if self.router_activation == "relu":
            return F.relu(logits)
        elif self.router_activation == "silu":
            return F.silu(logits)
        else:  # "none"
            return logits

    def forward(self, hidden_states: torch.Tensor):
        """(b, s, h) -> top_k_indices (b, s, k), norm_weights (b, s, k). Weights are UNBIASED."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
        router_logits = self.gate_proj(flat_hidden).float()
        router_logits = self._apply_router_activation(router_logits)

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
