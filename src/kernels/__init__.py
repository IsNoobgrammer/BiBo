"""
BiBo Triton Kernels — Fused GPU operations via Liger-Kernel + custom Triton.

Uses linkedin/Liger-Kernel for production-grade Triton ops:
- Fused RMSNorm (8-9x faster than PyTorch eager)
- Fused RoPE (2-3x faster, eliminates intermediate tensors)

Custom Triton kernels for BiBo-specific ops:
- Fused MoE GLU activation (eliminates 3 intermediate tensors per expert)
- Fused router scoring (sigmoid + logit_norm + bias in 1 kernel)

Patching utilities:
- patch_bibo_with_triton: Monkey-patch BiBo model (RMSNorm + RoPE)
- patch_moe_with_triton: Monkey-patch MoE layer (fused GLU activation)
- patch_qwen3_with_triton: Monkey-patch Qwen3/Qwen3MoE model

Works on any CUDA GPU — Triton compiles for target arch at runtime.
Kaggle T4 (sm_75), RTX 3050 (sm_86), A100 (sm_80) all supported.
"""

from .patch import (
    patch_bibo_with_triton,
    patch_qwen3_with_triton,
    unpatch_bibo,
    unpatch_qwen3,
)
from .moe_dispatch import (
    patch_moe_with_triton,
    unpatch_moe,
)

__all__ = [
    'patch_bibo_with_triton',
    'patch_qwen3_with_triton',
    'unpatch_bibo',
    'unpatch_qwen3',
    'patch_moe_with_triton',
    'unpatch_moe',
]
