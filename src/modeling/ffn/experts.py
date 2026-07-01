"""Special expert types and PolyGLU experts"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from src.configuration_bibo import BiBoConfig

__all__ = [
    'BiBoPolyGLUExpert',
    'BiBoCausalConv1D',
]

# Note: the Identity ("pass x through") and Zero ("x*0") experts are NOT separate classes — the
# fused MoE path (BiBoFusedExperts in moe.py) handles them inline (weighted passthrough / skip).


class BiBoPolyGLUExpert(nn.Module):
    """
    GLU expert with configurable activation function.
    
    PolyGLU idea: diverse activations across experts in the same MoE layer.
    Each expert uses a different activation in the GLU gate:
      - "silu"  → SiLU (SwiGLU, standard)
      - "relu2" → ReLU² (ReGLU², sparse + sharp)
      - "tanh"  → Tanh (TanhGLU, bounded + smooth)
    
    Architecture: down_proj( act(gate_proj(x)) * up_proj(x) )
    Same structure as BiBoMLP but with explicit activation choice.
    
    Args:
        config: Model config
        activation: One of "silu", "relu2", "tanh"
    """
    VALID_ACTIVATIONS = ("silu", "relu2", "tanh")

    def __init__(self, config: BiBoConfig, activation: str = "silu"):
        super().__init__()
        if activation not in self.VALID_ACTIVATIONS:
            raise ValueError(f"PolyGLU activation must be one of {self.VALID_ACTIVATIONS}, got '{activation}'")
        self.activation_name = activation
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "silu":
            return F.silu(x)
        elif self.activation_name == "relu2":
            r = F.relu(x)
            return (r.float() * r.float()).to(x.dtype)   # fp32 square: avoid fp16 overflow (>256 -> inf)
        elif self.activation_name == "tanh":
            return torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._activate(self.gate_proj(x)) * self.up_proj(x))


class BiBoCausalConv1D(nn.Module):
    """
    1D causal conv expert (shared, always-active).
    
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
