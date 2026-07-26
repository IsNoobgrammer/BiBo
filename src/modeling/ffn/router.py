"""MoE router — MiMo-V2.5 / DeepSeek-V3 auxiliary-loss-free sigmoid gating (verbatim routing)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoMoERouter']


class BiBoMoERouter(nn.Module):
    """
    MoE router with DeepSeek-V3 auxiliary-loss-free load balancing.
    
    Key design (from arXiv:2408.15664 + arXiv:2412.19437):
    
    1. Sigmoid gating (not softmax) — independent expert scoring, no competition
    2. Bias affects ONLY top-k selection, NOT output computation weights
    3. Bias updated via: b_i += u * sign(mean_load - expert_load), u=0.001
    4. No interference gradients — bias is outside the computation graph
    
    Pipeline (MiMo-V2.5 `MiMoV2MoEGate` structure, but see step 6 — the normalizer differs):
        1. raw_logits = W @ x  (MLP or Conv)
        2. raw_logits += noise  (DEPRECATED — commented out; we use router_noise=0)
        3. scores = gate(raw_logits)  — sigmoid (default), situ, or softmax
        4. selection_scores = scores + bias  — for top-k selection ONLY
        5. top_k_indices = topk(selection_scores)
        6. top_k_weights = scores[top_k_indices]  — UNBIASED; SOFTMAX'd if norm_topk_prob;
           × routed_scaling

    ⚠️ Step 6 is a DELIBERATE DIVERGENCE from MiMo (2026-07-26). MiMo/DeepSeek-V3 normalize by
    ÷sum. BiBo uses softmax over the gathered top-k scores instead, because ÷sum is only valid
    for non-negative scores: with a signed gate (`gate_type="situ"`, range ~(-0.28, 1)) the sum can
    cross zero (division explodes) or go negative (every weight flips sign). Softmax is defined
    for signed inputs, always yields positive weights summing to 1, and has no such singularity.
    Cost: softmax over sigmoid-squashed scores is FLATTER than ÷sum (inputs are confined to a
    unit-width band, so the max weight ratio is bounded by e). `norm_topk_prob=False` keeps the
    raw gathered scores and is unchanged.
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.router_noise = config.router_noise
        self.router_type = config.router_type
        self.kernel_size = config.kernel_size
        self.causal_padding = self.kernel_size - 1

        # Configurable options — getattr fallbacks MATCH the BiBoConfig defaults so a partial/stale
        # config can't silently flip behavior (norm_topk_prob especially: config default is True).
        self.router_activation = getattr(config, 'router_activation', 'none')
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', True)
        self.gate_type = getattr(config, 'gate_type', 'sigmoid')  # 'sigmoid' or 'softmax'
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)  # MiMo/DeepSeek-V3

        # Load-balancing bias — heuristically updated (not optimizer-managed)
        # DeepSeek-V3: bias only affects selection, not output weights
        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts), requires_grad=False)

        # Router projection — normal init (matches Qwen3MoE)
        if self.router_type == "mlp":
            self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=config.initializer_range)
        elif self.router_type == "conv":
            self.gate_conv = nn.Conv1d(config.hidden_size, self.num_routed_experts, self.kernel_size, padding=0, bias=False)
            nn.init.normal_(self.gate_conv.weight, mean=0.0, std=config.initializer_range)
        else:
            raise ValueError(f"Unknown router type: {self.router_type}. Expected 'mlp' or 'conv'.")

    def _apply_router_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply optional activation to router logits before gating."""
        if self.router_activation == "relu":
            return F.relu(logits)
        elif self.router_activation == "silu":
            return F.silu(logits)
        else:  # "none"
            return logits

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            top_k_indices: (batch, seq_len, top_k)
            norm_weights: (batch, seq_len, top_k) — UNBIASED routing weights
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Step 1: raw logits
        if self.router_type == "mlp":
            flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
            router_logits = self.gate_proj(flat_hidden).float()
        else:  
            x_perm = rearrange(hidden_states, 'b s h -> b h s')
            x_padded = F.pad(x_perm, (self.causal_padding, 0))
            conv_out = self.gate_conv(x_padded)
            router_logits = rearrange(conv_out, 'b e s -> (b s) e').float()

        # Step 2: exploration noise (training only) — DEPRECATED, DO NOT REMOVE.
        # We do not use router noise (router_noise=0). Commented out because forward-time
        # randomness breaks gradient checkpointing unless RNG state is preserved on recompute.
        # Kept (not deleted) so it can be re-enabled if exploration noise is ever needed again.
        # if self.training and self.router_noise > 0:
        #     noise_stddev = math.sqrt(self.router_noise)
        #     noise = torch.randn_like(router_logits) * noise_stddev
        #     router_logits = router_logits + noise.detach()

        # Step 3: router activation (ReLU/SiLU/none)
        router_logits = self._apply_router_activation(router_logits)

        # Step 4: gating — sigmoid (independent), situ (independent, signed), softmax (competitive)
        if self.gate_type == "sigmoid":
            # Sigmoid: each expert scored independently (DeepSeek-V3 preferred)
            scores = torch.sigmoid(router_logits)
        elif self.gate_type == "situ":
            # SiTU (Sigmoid Tanh Unit) = sigmoid(x)*tanh(x) — same fn as PolyGLU act code 5
            # (`--situ`); shipped as the EXPERT activation in Kimi K3, used here as the ROUTER gate.
            # Range ~(-0.2785, 1); min at x ≈ -0.7799; NON-MONOTONIC in the logit (f(-5) ≈ -0.0067
            # > f(-0.78) ≈ -0.2785), so a strongly-rejected expert can outrank a mildly-rejected
            # one in the top-k ordering. Requires norm_topk_prob=True (softmax) — ÷sum is unsafe
            # on signed scores.
            scores = torch.sigmoid(router_logits) * torch.tanh(router_logits)
        elif self.gate_type == "softmax":
            # Softmax: competitive normalization (legacy)
            scores = F.softmax(router_logits, dim=1)
        else:
            # Explicit: the old `else -> softmax` fallthrough silently accepted typos.
            raise ValueError(
                f"gate_type must be 'sigmoid', 'situ', or 'softmax', got '{self.gate_type}'"
            )

        # Step 5: selection uses scores + bias. Bias is for load-balancing SELECTION ONLY
        # (MiMo/DeepSeek-V3) — never the combine weights (those come from raw `scores`, Step 7).
        selection_scores = scores + self.bias

        # Step 6: top-k selection. sorted=False to match MiMo's gate EXACTLY — same return order =>
        # same fp summation order in the norm divisor => bit-exact weights (vs sorted=True's 1-ULP drift).
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False)

        # Step 8: gather UNBIASED weights — normalize only if norm_topk_prob=True
        # CRITICAL: use original scores, NOT selection_scores (which includes bias)
        top_k_weights = scores.gather(-1, top_k_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            # SOFTMAX over the gathered top-k scores (NOT MiMo's ÷sum) — see the class docstring.
            # Safe for signed gates (gate_type="situ"): no zero-sum singularity, no sign flip, and
            # the weights are always positive and sum to exactly 1.
            norm_weights = F.softmax(top_k_weights, dim=-1)
        else:
            norm_weights = top_k_weights
        # MiMo/DeepSeek-V3: rescale the routed weights by a constant AFTER norm (1.0 = no-op)
        norm_weights = norm_weights * self.routed_scaling_factor

        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        # Keep weights in fp32 (like MiMo's gate) — the MoE combine accumulates in fp32 and casts
        # to hidden dtype at the end. ALL router ops stay fp32 end-to-end (logits→sigmoid→norm→here).
        return top_k_indices.long(), norm_weights.float()
