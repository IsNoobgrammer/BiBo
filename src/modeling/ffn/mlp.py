"""Standard MLP"""
import torch.nn as nn
from transformers.activations import ACT2FN
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoMLP']


class BiBoMLP(nn.Module):
    """
    MLP with SwiGLU gating.
    
    Args:
        config: Model config
        is_expert: Use moe_intermediate_size if True. Default: False
    """
    def __init__(self, config: BiBoConfig, is_expert=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if is_expert else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
