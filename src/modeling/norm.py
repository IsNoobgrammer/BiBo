"""Normalization layers"""
import torch
from torch import nn

__all__ = ['BiBoRMSNorm']


class BiBoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        Args:
            hidden_size (int): Size of last dim
            eps (float): Numerical stability. Default: 1e-5
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
