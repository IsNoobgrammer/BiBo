"""
BiBo Triton Kernels — Fused GPU operations via Liger-Kernel + custom Triton.

Uses linkedin/Liger-Kernel for production-grade Triton ops:
- Fused RMSNorm (8-9x faster than PyTorch eager)
- Fused RoPE (2-3x faster, eliminates intermediate tensors)

Custom Triton kernels for BiBo-specific ops:
- Fused MoE GLU activation (eliminates 3 intermediate tensors per expert)
  In-place backward (Liger pattern): writes gradients to saved gate_up, no allocation
- Fused router scoring (sigmoid + logit_norm + bias in 1 kernel)
- Dense MLP SwiGLU (Triton forward + backward activation fusion)
- Conv expert: PyTorch ops (Triton kernel overhead makes it 0.41x slower — see benchmark)

Key insight from Liger: NEVER fuse GEMMs. Fuse only the elementwise operations
(silu, multiply, activation) BETWEEN GEMMs. cuBLAS handles all matrix multiplications.

Patching utilities:
- patch_bibo_with_triton: Monkey-patch BiBo model (RMSNorm + RoPE)
- patch_moe_with_triton: Monkey-patch MoE layer (fused GLU activation)
- patch_dense_mlp_with_triton: Monkey-patch dense MLP layers (fused SwiGLU)
- patch_conv_router_with_triton: Monkey-patch conv router (optimized projection)
- patch_conv_expert_with_triton: Monkey-patch conv shared expert (PyTorch ops)

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
    triton_fused_weight_scatter,
    triton_batched_glu_activation,
)
from .dense_mlp import (
    patch_dense_mlp_with_triton,
    unpatch_dense_mlp,
    patch_qwen_dense_mlp_with_triton,
    unpatch_qwen_dense_mlp,
    triton_fused_swiglu,
    _FusedSwiGLUFull,
    _TritonFusedGLUFunction,
)
from .conv_fused import (
    patch_conv_router_with_triton,
    patch_conv_expert_with_triton,
    unpatch_conv_router,
    unpatch_conv_expert,
    triton_causal_conv1d_router,
    triton_causal_conv1d_gated,
    triton_fused_conv_gate_multiply,
    triton_fused_permute_act,
)

__all__ = [
    'patch_bibo_with_triton',
    'patch_qwen3_with_triton',
    'unpatch_bibo',
    'unpatch_qwen3',
    'patch_moe_with_triton',
    'unpatch_moe',
    'patch_dense_mlp_with_triton',
    'unpatch_dense_mlp',
    'patch_qwen_dense_mlp_with_triton',
    'unpatch_qwen_dense_mlp',
    'triton_fused_swiglu',
    'patch_conv_router_with_triton',
    'patch_conv_expert_with_triton',
    'unpatch_conv_router',
    'unpatch_conv_expert',
    'triton_causal_conv1d_router',
    'triton_causal_conv1d_gated',
    'triton_fused_conv_gate_multiply',
    'triton_fused_permute_act',
]
