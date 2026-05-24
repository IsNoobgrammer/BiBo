"""MoE router — DeepSeek-V3 auxiliary-loss-free sigmoid gating + logit norm"""
import math
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
    
    Pipeline:
        1. raw_logits = W @ x  (MLP or Conv)
        2. raw_logits += noise  (if training and router_noise > 0)
        3. scores = sigmoid(raw_logits)  — independent per-expert scores
        4. if use_router_logit_norm: scores = lambda * (scores - mean) / std
        5. selection_scores = scores + bias  — for top-k selection ONLY
        6. top_k_indices = topk(selection_scores)
        7. top_k_weights = normalize(scores[top_k_indices])  — UNBIASED weights
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.router_noise = config.router_noise
        self.router_type = config.router_type
        self.kernel_size = config.kernel_size
        self.causal_padding = self.kernel_size - 1
        self.router_lambda = getattr(config, 'router_lambda', 1.0)
        
        # Configurable options
        self.use_router_logit_norm = getattr(config, 'use_router_logit_norm', False)
        self.router_activation = getattr(config, 'router_activation', 'none')
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)
        self.gate_type = getattr(config, 'gate_type', 'sigmoid')  # 'sigmoid' or 'softmax'

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

        # Step 2: exploration noise (training only)
        if self.training and self.router_noise > 0:
            noise_stddev = math.sqrt(self.router_noise)
            noise = torch.randn_like(router_logits) * noise_stddev
            router_logits = router_logits + noise.detach()

        # Step 3: router activation (ReLU/SiLU/none)
        router_logits = self._apply_router_activation(router_logits)

        # Step 4: gating — sigmoid (independent) or softmax (competitive)
        if self.gate_type == "sigmoid":
            # Sigmoid: each expert scored independently (DeepSeek-V3 preferred)
            scores = torch.sigmoid(router_logits)
        else:
            # Softmax: competitive normalization (legacy)
            scores = F.softmax(router_logits, dim=1)

        # Step 5: optional logit normalization (Skywork-MoE style)
        if self.use_router_logit_norm:
            mean = scores.mean(dim=1, keepdim=True)
            std = scores.std(dim=1, keepdim=True) + 1e-6
            scores = self.router_lambda * (scores - mean) / std

        # Step 6: selection uses scores + bias (bias for load balancing ONLY)
        # DeepSeek-V3: bias affects selection, NOT output weights
        selection_scores = scores + self.bias

        # Step 7: top-k selection
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1)

        # Step 8: gather UNBIASED weights — normalize only if norm_topk_prob=True
        # CRITICAL: use original scores, NOT selection_scores (which includes bias)
        top_k_weights = scores.gather(-1, top_k_indices)
        if self.norm_topk_prob:
            norm_weights = top_k_weights / (top_k_weights.sum(-1, keepdim=True) + 1e-6)
        else:
            norm_weights = top_k_weights

        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.to(hidden_states.dtype)
