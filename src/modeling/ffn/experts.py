import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from src.configuration_bibo import BiBoConfig

__all__ = ['BiBoCausalConv1D']


class BiBoCausalConv1D(nn.Module):
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
        x_padded = F.pad(x_perm, (self.causal_padding_gate, 0))
        gate_conv_out = self.gate_conv(x_padded)
        gate_output = rearrange(gate_conv_out, 'b i s -> b s i')
        output = self.down_proj(self.act_fn(gate_output) * self.up_proj(x))

        if output.shape[1] != seq_len:
             raise RuntimeError(f"Conv expert len mismatch. Expected {seq_len}, got {output.shape[1]}")
        return output
