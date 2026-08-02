"""Experimental BiBo variants.

This package may depend on reusable components from :mod:`src`, but ``src`` must
never import from here. Experimental architectures graduate to ``src`` only via
an explicit, later change.
"""

from .configuration_bibo import BiBoConfig
from .modeling_bibo import (
    BiBoDecoderLayer,
    BiBoForCausalLM,
    BiBoModel,
    BiBoPreTrainedModel,
    apply_attention_residual,
)

__all__ = [
    "BiBoConfig",
    "BiBoDecoderLayer",
    "BiBoPreTrainedModel",
    "BiBoModel",
    "BiBoForCausalLM",
    "apply_attention_residual",
]
