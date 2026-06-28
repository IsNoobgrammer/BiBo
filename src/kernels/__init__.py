"""
BiBo Triton Kernels — Fused GPU operations via Liger-Kernel + custom Triton.

Uses linkedin/Liger-Kernel for production-grade Triton ops:
- Fused RMSNorm (8-9x faster than PyTorch eager)
- Fused RoPE (2-3x faster, eliminates intermediate tensors)

Custom Triton kernels for BiBo-specific ops (the 3-kernel training set):
- MoE: per-expert dispatch with a manual backward + fused fp32 combine (moe_dispatch.py)
- XSA: Exclusive Self Attention rejection (xsa_fused.py)
- Fused-linear cross-entropy (fused_ce.py)

Key insight from Liger: NEVER fuse GEMMs. Fuse only the elementwise operations
(silu, multiply, activation) BETWEEN GEMMs. cuBLAS handles all matrix multiplications.

Patching utilities:
- patch_bibo_with_triton: Monkey-patch BiBo model (RMSNorm + RoPE, Liger)
- patch_moe_with_triton: Monkey-patch MoE layer (per-expert dispatch + fused combine)
- patch_xsa_with_triton: Monkey-patch XSA rejection

Works on any CUDA GPU — Triton compiles for target arch at runtime.
Kaggle T4 (sm_75), RTX 3050 (sm_86), A100 (sm_80) all supported.
"""

from .patch import (
    patch_bibo_with_liger,
    patch_qwen3_with_liger,
    patch_bibo_with_triton,   # deprecated alias of patch_bibo_with_liger
    patch_qwen3_with_triton,  # deprecated alias of patch_qwen3_with_liger
    patch_qwen3_fused_ce,
    unpatch_bibo,
    unpatch_qwen3,
)
from .moe_dispatch import (
    patch_moe_with_triton,
    unpatch_moe,
    triton_fused_weight_scatter_autograd,
    triton_batched_glu_activation,
)
from .xsa_fused import (
    fused_xsa,
    patch_xsa_with_triton,
    unpatch_xsa,
)

__all__ = [
    'patch_bibo_with_liger',
    'patch_qwen3_with_liger',
    'patch_bibo_with_triton',
    'patch_qwen3_with_triton',
    'unpatch_bibo',
    'unpatch_qwen3',
    'patch_moe_with_triton',
    'unpatch_moe',
    'triton_fused_weight_scatter_autograd',
    'fused_xsa',
    'patch_xsa_with_triton',
    'unpatch_xsa',
]
