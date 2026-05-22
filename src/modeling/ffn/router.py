"""MoE router — configurable logit norm, activation, load balancing"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoMoERouter']


class BiBoMoERouter(nn.Module):
    """
    MoE router with configurable behavior:
    
    - router_activation: "none" | "relu" | "silu" — applied to raw logits before softmax
    - use_router_logit_norm: bool — z-score normalize logits (Skywork-MoE style)
    - load_balance_strategy: "none" | "bias" | "aux_loss"
    - router_type: "mlp" | "conv"
    - router_noise: float — exploration noise during training
    
    Routing pipeline:
        1. raw_logits = W @ x  (MLP or Conv)
        2. raw_logits += noise  (if training and router_noise > 0)
        3. logits = activation(raw_logits)  (relu/silu/none)
        4. if use_router_logit_norm: logits = lambda * (logits - mean) / std
        5. routing_weights = softmax(logits)  — for weight computation
        6. selection_scores = logits + bias  — for top-k selection (bias strategy only)
        7. top_k_indices = topk(selection_scores)
        8. top_k_weights = normalize(routing_weights[top_k_indices])
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
        
        # New configurable options
        self.use_router_logit_norm = getattr(config, 'use_router_logit_norm', False)
        self.load_balance_strategy = getattr(config, 'load_balance_strategy', 'none')
        self.aux_loss_coef = getattr(config, 'aux_loss_coef', 0.001)
        self.router_activation = getattr(config, 'router_activation', 'none')
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)

        # Load-balancing bias — only used when load_balance_strategy="bias"
        # NOT learned via gradient, updated by heuristic threshold logic.
        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts), requires_grad=False)

        # Router projection — zero-init like Qwen (uniform routing at start)
        if self.router_type == "mlp":
            self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
            nn.init.zeros_(self.gate_proj.weight)
            self.gate_proj._is_router_gate = True  # tag so _init_weights skips re-init
        elif self.router_type == "conv":
            self.gate_conv = nn.Conv1d(config.hidden_size, self.num_routed_experts, self.kernel_size, padding=0, bias=False)
            nn.init.zeros_(self.gate_conv.weight)
        else:
            raise ValueError(f"Unknown router type: {self.router_type}. Expected 'mlp' or 'conv'.")

    def _apply_router_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply optional activation to router logits before softmax."""
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
            norm_weights: (batch, seq_len, top_k)
            aux_loss: scalar tensor or None (only when load_balance_strategy="aux_loss")
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

        # --- AUX LOSS: computed from RAW logits (before activation), matching Qwen ---
        aux_loss = None
        if self.load_balance_strategy == "aux_loss" and self.training:
            raw_routing_weights = F.softmax(router_logits, dim=-1)
            _, raw_top_k_indices = torch.topk(raw_routing_weights, self.top_k, dim=-1)
            expert_mask = F.one_hot(raw_top_k_indices, num_classes=self.num_routed_experts).float()
            tokens_per_expert = expert_mask.mean(dim=0)        # (top_k, num_experts)
            router_prob_per_expert = raw_routing_weights.mean(dim=0)  # (num_experts,)
            aux_loss = self.num_routed_experts * (tokens_per_expert * router_prob_per_expert.unsqueeze(0)).sum()

        # Step 3: router activation (ReLU/SiLU/none)
        router_logits = self._apply_router_activation(router_logits)

        # Step 4: optional logit normalization (Skywork-MoE style)
        if self.use_router_logit_norm:
            mean = router_logits.mean(dim=1, keepdim=True)
            std = router_logits.std(dim=1, keepdim=True) + 1e-6
            router_logits = self.router_lambda * (router_logits - mean) / std

        # Step 5: routing weights via softmax (always on clean logits)
        routing_weights = F.softmax(router_logits, dim=1)

        # Step 6: selection scores (bias only when strategy="bias")
        if self.load_balance_strategy == "bias":
            selection_scores = router_logits + self.bias
        else:
            selection_scores = router_logits

        # Step 7: top-k selection
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1)

        # Step 8: gather weights — normalize only if norm_topk_prob=True
        top_k_weights = routing_weights.gather(-1, top_k_indices)
        if self.norm_topk_prob:
            norm_weights = top_k_weights / (top_k_weights.sum(-1, keepdim=True) + 1e-6)
        else:
            norm_weights = top_k_weights

        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.to(hidden_states.dtype), aux_loss
