"""
BiBo Triton Kernels — Fused GPU operations via Liger-Kernel + custom Triton.

Uses linkedin/Liger-Kernel for production-grade Triton ops:
- Fused RMSNorm (8-9x faster than PyTorch eager)
- Fused RoPE (2-3x faster, eliminates intermediate tensors)

Custom Triton kernels for BiBo-specific ops:
- Fused MoE GLU activation (eliminates 3 intermediate tensors per expert)
- Fused router scoring (sigmoid + logit_norm + bias in 1 kernel)
- Fused conv permute + activation + gate multiply (eliminates 2 intermediates)
  → 1.34-1.41x training speedup, 8% memory reduction for conv models
- Fused dense MLP SwiGLU (fused gate_up GEMM + Triton silu*up activation)
  → Eliminates 2 intermediate tensors per dense layer forward

Patching utilities:
- patch_bibo_with_triton: Monkey-patch BiBo model (RMSNorm + RoPE)
- patch_moe_with_triton: Monkey-patch MoE layer (fused GLU activation)
- patch_dense_mlp_with_triton: Monkey-patch dense MLP layers (fused SwiGLU)
- patch_conv_router_with_triton: Monkey-patch conv router (optimized projection)
- patch_conv_expert_with_triton: Monkey-patch conv shared expert (fused gate)
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
