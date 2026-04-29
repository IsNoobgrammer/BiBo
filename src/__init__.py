"""
BiBo: Biased Bounded MoE Language Model

A compact, efficient MoE architecture with:
- Multi-layer KV sharing for reduced memory
- Adaptive router with bias-based load balancing
- Diverse expert types (MLP, Conv, Identity, Noise, ReLU, Zero)
- Scaling Softmax (SSMax) for long-context attention
"""

from .configuration_bibo import BiBoConfig
from .modeling_bibo import (
    BiBoModel,
    BiBoForCausalLM,
    BiBoPreTrainedModel,
)

__version__ = "0.1.0"

__all__ = [
    "BiBoConfig",
    "BiBoModel",
    "BiBoForCausalLM",
    "BiBoPreTrainedModel",
]
