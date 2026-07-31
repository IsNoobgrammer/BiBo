import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoMoERouter']


class BiBoMoERouter(nn.Module):
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts), requires_grad=False)

        self._probe_gap = False
        self.boundary_gap = None

        self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=config.initializer_range)

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.gate_proj(rearrange(hidden_states, 'b s h -> (b s) h')).float()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        scores = torch.sigmoid(self.router_logits(hidden_states))

        if self._probe_gap and self.top_k < self.num_routed_experts:
            with torch.no_grad():
                _tk = scores.topk(self.top_k + 1, dim=-1).values
                self.boundary_gap = (_tk[..., self.top_k - 1] - _tk[..., self.top_k]).mean()

        selection_scores = scores + self.bias
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False)

        top_k_weights = scores.gather(-1, top_k_indices)
        if self.top_k > 1 and self.norm_topk_prob == "softmax":
            norm_weights = F.softmax(top_k_weights, dim=-1)
        elif self.top_k > 1 and self.norm_topk_prob:
            norm_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            norm_weights = top_k_weights

        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.float()
