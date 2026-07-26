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

# The ±Identity specials (+w*x / -w*x) are NOT classes — BiBoFusedExperts handles them inline as a
# signed weighted passthrough. (The Zero expert was removed Jul 26 2026, see configuration_bibo.py.)


class BiBoPolyGLUExpert(nn.Module):
    """down_proj(act(gate_proj(x)) * up_proj(x)) — a BiBoMLP with an explicit activation choice.

    PolyGLU's premise is diverse activations across experts in one MoE layer. Reference
    implementation only: the shipped path is BiBoFusedExperts, which inlines the same math.
    """
    VALID_ACTIVATIONS = ("silu", "relu2", "normsilu")

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
            return (r.float() * r.float()).to(x.dtype)   # fp32 square: fp16 overflows above 256
        elif self.activation_name == "normsilu":
            # eps MUST match _NORMSILU_EPS in moe.py and _NS_EPS in the tkf kernel
            g = x.float()
            g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + 1e-6)
            return F.silu(g).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._activate(self.gate_proj(x)) * self.up_proj(x))


class BiBoCausalConv1D(nn.Module):
    """Shared, always-active expert: left-padded causal 1D conv -> gated act -> linear proj.

    This is the conv SHARED EXPERT (shared_expert_type="conv"), not the conv router — that was
    removed Jul 26 2026.
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
        x_padded = F.pad(x_perm, (self.causal_padding_gate, 0))   # left-pad k-1 => causal
        gate_conv_out = self.gate_conv(x_padded)
        gate_output = rearrange(gate_conv_out, 'b i s -> b s i')
        output = self.down_proj(self.act_fn(gate_output) * self.up_proj(x))

        if output.shape[1] != seq_len:
             raise RuntimeError(f"Conv expert len mismatch. Expected {seq_len}, got {output.shape[1]}")
        return output
