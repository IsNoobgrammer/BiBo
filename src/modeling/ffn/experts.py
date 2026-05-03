"""Special expert types"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from src.configuration_bibo import BiBoConfig

__all__ = [
    'BiBoIdentityExpert',
    'BiBoReLUExpert', 
    'BiBoZeroExpert',
    'BiBoNoiseExpert',
    'BiBoCausalConv1D'
]


class BiBoIdentityExpert(nn.Module):
    """Identity/residual expert"""
    def __init__(self, config: BiBoConfig, *args, **kwargs): 
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BiBoReLUExpert(nn.Module):
    """ReLU^2 activated expert"""
    def __init__(self, config: BiBoConfig, *args, **kwargs): 
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


class BiBoZeroExpert(nn.Module):
    """Returns x * 0 (preserves shape/dtype/device/sharding)"""
    def __init__(self, config: BiBoConfig, *args, **kwargs):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x * zero


class BiBoNoiseExpert(nn.Module):
    """Adds Gaussian noise (std=0.5)"""
    def __init__(self, config: BiBoConfig, std: float = 0.5, *args, **kwargs):
        super().__init__()
        self.std = std
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if True:  # use in inference too
            with torch.no_grad():
                noise = torch.randn_like(x) * self.std
            return x + noise


class BiBoCausalConv1D(nn.Module):
    """
    1D causal conv expert.
    
    Applies causal (left-padded) 1D conv → gated activation → linear proj.
    Captures local sequential deps while preserving causality.
    
    Args:
        config: Model config
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.kernel_size_gate = config.kernel_size
        self.causal_padding_gate = self.kernel_size_gate - 1

        self.gate_conv = nn.Conv1d(self.hidden_size, self.intermediate_size, self.kernel_size_gate, padding=0, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = x.shape
        x_perm = rearrange(x, 'b s h -> b h s')
        
        # Causal pad left (k-1)
        x_padded = F.pad(x_perm, (self.causal_padding_gate, 0))
        gate_conv_out = self.gate_conv(x_padded)
        gate_output = rearrange(gate_conv_out, 'b i s -> b s i')
        output = self.down_proj(self.act_fn(gate_output) * self.up_proj(x))

        if output.shape[1] != seq_len:
             raise RuntimeError(f"Conv expert len mismatch. Expected {seq_len}, got {output.shape[1]}")
        return output
