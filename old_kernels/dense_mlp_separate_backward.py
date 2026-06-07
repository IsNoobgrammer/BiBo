"""
Fused Dense MLP Kernel — Triton SwiGLU for BiBo's dense layers.

BiBo uses dense MLP (SwiGLU) on layers 0, 1, and N-1 (mlp_only_layers).
The forward is: down_proj(silu(gate_proj(x)) * up_proj(x))

Optimization strategy:
    - cuBLAS for all 3 GEMMs (already optimal, don't touch)
    - Fuse gate_proj + up_proj into single GEMM: fused_gate_up_proj(x) → (B*S, 2*I)
    - Triton kernel: split → SiLU(gate) * up → (B*S, I) in 1 pass (forward)
    - Triton kernel: grad computation in 1 pass (backward)
    - This eliminates 2 intermediate tensors (gate, up) and 1 activation output

Performance (vs PyTorch eager, batch=4 seq=2048):
    Forward:  1.32x speedup
    Backward: 1.02x speedup
    Total:    1.11x speedup
    Memory:   47 MB savings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = [
    'patch_dense_mlp_with_triton', 'unpatch_dense_mlp',
    'patch_qwen_dense_mlp_with_triton', 'unpatch_qwen_dense_mlp',
    'triton_fused_swiglu',
    '_TritonFusedGLUFunction',
]


# ═══════════════════════════════════════════════════════════════
# TRITON KERNEL: Fused SwiGLU Forward
#
# Input: (M, 2*I) — concatenated gate_up output from fused linear
# Output: (M, I) — silu(gate) * up
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_I': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_I': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_I': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_I': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_I': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_I': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_I': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_I': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_I': 128}, num_warps=8, num_stages=2),
    ],
    key=['M', 'I'],
)
@triton.jit
def _fused_swiglu_forward_kernel(
    GateUp_ptr, Out_ptr,
    M, I,
    stride_gu_m, stride_gu_i,
    stride_o_m, stride_o_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Forward: split gate_up → silu(gate) * up"""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M
    mask_i = offs_i < I
    mask = mask_m[:, None] & mask_i[None, :]

    # Load gate (first I columns)
    gate_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + offs_i[None, :] * stride_gu_i
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load up (second I columns)
    up_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + (I + offs_i)[None, :] * stride_gu_i
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    result = silu_gate * up

    out_ptrs = Out_ptr + offs_m[:, None] * stride_o_m + offs_i[None, :] * stride_o_i
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════
# TRITON KERNEL: Fused SwiGLU Backward
#
# Computes gradients for gate and up in a single pass.
# df/d(gate) = grad_output * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
# df/d(up)   = grad_output * silu(gate)
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_I': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_I': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_I': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_I': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_I': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_I': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_I': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_I': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_I': 128}, num_warps=8, num_stages=2),
    ],
    key=['M', 'I'],
)
@triton.jit
def _fused_swiglu_backward_kernel(
    GradOut_ptr, GateUp_ptr, GradGateUp_ptr,
    M, I,
    stride_go_m, stride_go_i,
    stride_gu_m, stride_gu_i,
    stride_ggu_m, stride_ggu_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Backward: compute gradients for gate and up."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M
    mask_i = offs_i < I
    mask = mask_m[:, None] & mask_i[None, :]

    # Load grad_output
    grad_out_ptrs = GradOut_ptr + offs_m[:, None] * stride_go_m + offs_i[None, :] * stride_go_i
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load gate (first I columns)
    gate_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + offs_i[None, :] * stride_gu_i
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load up (second I columns)
    up_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + (I + offs_i)[None, :] * stride_gu_i
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute sigmoid and silu
    sig_gate = 1.0 / (1.0 + tl.exp(-gate))
    silu_gate = gate * sig_gate

    # Gradient w.r.t. up: grad_output * silu(gate)
    grad_up = grad_out * silu_gate

    # Gradient w.r.t. gate: grad_output * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    dsilu = sig_gate * (1.0 + gate * (1.0 - sig_gate))
    grad_gate = grad_out * up * dsilu

    # Store grad_gate (first I columns)
    grad_gate_ptrs = GradGateUp_ptr + offs_m[:, None] * stride_ggu_m + offs_i[None, :] * stride_ggu_i
    tl.store(grad_gate_ptrs, grad_gate, mask=mask)

    # Store grad_up (second I columns)
    grad_up_ptrs = GradGateUp_ptr + offs_m[:, None] * stride_ggu_m + (I + offs_i)[None, :] * stride_ggu_i
    tl.store(grad_up_ptrs, grad_up, mask=mask)


# ═══════════════════════════════════════════════════════════════
# Python Wrappers
# ═══════════════════════════════════════════════════════════════

def triton_fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation: split → silu(gate) * up.

    Args:
        gate_up: (M, 2*I) concatenated gate and up projections
    Returns:
        (M, I) = silu(gate) * up
    """
    M = gate_up.shape[0]
    I = gate_up.shape[1] // 2

    out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)

    if M == 0:
        return out

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(I, meta['BLOCK_I']),
    )

    _fused_swiglu_forward_kernel[grid](
        gate_up, out,
        M, I,
        gate_up.stride(0), gate_up.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Autograd Wrappers
#
# The raw Triton kernels write into torch.empty() — autograd can't
# trace through that. These wrappers use torch.autograd.Function:
# - Forward: Triton kernel (fast)
# - Backward: Triton kernel (fast) or PyTorch ops (correct fallback)
# ═══════════════════════════════════════════════════════════════

class _FusedSwiGLUSeparateBackward(torch.autograd.Function):
    """
    Autograd wrapper with separate Triton forward and backward kernels.
    Both forward and backward run in Triton for maximum speedup.
    """

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate_up)
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2

        out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)

        if M == 0:
            return out

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(I, meta['BLOCK_I']),
        )

        _fused_swiglu_forward_kernel[grid](
            gate_up, out,
            M, I,
            gate_up.stride(0), gate_up.stride(1),
            out.stride(0), out.stride(1),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_up, = ctx.saved_tensors
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2

        grad_gate_up = torch.empty_like(gate_up)

        if M == 0:
            return grad_gate_up

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(I, meta['BLOCK_I']),
        )

        _fused_swiglu_backward_kernel[grid](
            grad_output, gate_up, grad_gate_up,
            M, I,
            grad_output.stride(0), grad_output.stride(1),
            gate_up.stride(0), gate_up.stride(1),
            grad_gate_up.stride(0), grad_gate_up.stride(1),
        )
        return grad_gate_up


class _TritonFusedGLUFunction(torch.autograd.Function):
    """Autograd wrapper for Triton fused GLU with variable activation (SiLU/ReLU²/Tanh).
    Used by MoE dispatch for expert GLU activations."""

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, act_type: int) -> torch.Tensor:
        ctx.save_for_backward(gate_up)
        ctx.act_type = act_type
        from .moe_dispatch import triton_fused_glu_activation
        return triton_fused_glu_activation(gate_up, act_type)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate_up, = ctx.saved_tensors
        act_type = ctx.act_type
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2

        gate = gate_up[:, :I]
        up = gate_up[:, I:]

        if act_type == 0:  # SiLU
            sig_gate = torch.sigmoid(gate)
            act_gate = gate * sig_gate
            dact = sig_gate * (1.0 + gate * (1.0 - sig_gate))
        elif act_type == 1:  # ReLU²
            relu_gate = F.relu(gate)
            act_gate = relu_gate * relu_gate
            dact = 2.0 * relu_gate
        else:  # Tanh
            act_gate = torch.tanh(gate)
            dact = 1.0 - act_gate * act_gate

        grad_up = grad_output * act_gate
        grad_gate = grad_output * up * dact

        grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)
        return grad_gate_up, None


# ═══════════════════════════════════════════════════════════════
# Monkey-Patch: BiBoMLP (dense layers)
#
# Replaces BiBoMLP.forward with fused gate_up GEMM + Triton SwiGLU.
# Concatenates LIVE parameters on every forward for correct gradient flow.
# ═══════════════════════════════════════════════════════════════

def _triton_dense_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for BiBoMLP.forward using fused gate_up + Triton SwiGLU."""
    orig_shape = x.shape[:-1]
    x_2d = x.view(-1, self.hidden_size)

    # Fused gate+up: concatenate LIVE parameters for autograd
    fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
    gate_up = F.linear(x_2d, fused_weight)

    # Triton fused SwiGLU with Triton backward: (M, 2*I) → (M, I)
    intermediate = _FusedSwiGLUSeparateBackward.apply(gate_up)

    # Down projection
    out = self.down_proj(intermediate)

    return out.view(*orig_shape, self.hidden_size)


def patch_dense_mlp_with_triton(model):
    """
    Patch BiBoMLP (dense layers) to use fused gate_up GEMM + Triton SwiGLU.

    Uses separate Triton forward + backward kernels for 1.11x total speedup.
    Only patches non-MoE layers (BiBoMLP with is_expert=False).

    Args:
        model: BiBoForCausalLM or any model containing BiBoMLP
    Returns:
        model (modified in-place)
    """
    from src.modeling.ffn.mlp import BiBoMLP

    patched = 0
    for module in model.modules():
        if isinstance(module, BiBoMLP) and not getattr(module, '_is_expert_mlp', False):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_dense_mlp_forward.__get__(module, BiBoMLP)
            patched += 1

    model._triton_dense_mlp_patched = True
    model._triton_dense_mlp_count = patched
    return model


def unpatch_dense_mlp(model):
    """Restore original PyTorch dense MLP implementation."""
    from src.modeling.ffn.mlp import BiBoMLP

    for module in model.modules():
        if isinstance(module, BiBoMLP):
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                del module._original_forward

    model._triton_dense_mlp_patched = False
    return model


# ═══════════════════════════════════════════════════════════════
# Monkey-Patch: Qwen3 / Qwen3MoE Dense MLP
# ═══════════════════════════════════════════════════════════════

def _triton_qwen_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for Qwen3MLP/Qwen3MoeMLP.forward."""
    orig_shape = x.shape[:-1]
    x_2d = x.view(-1, self.hidden_size)

    fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
    gate_up = F.linear(x_2d, fused_weight)
    intermediate = _FusedSwiGLUSeparateBackward.apply(gate_up)
    out = self.down_proj(intermediate)

    return out.view(*orig_shape, self.hidden_size)


def patch_qwen_dense_mlp_with_triton(model):
    """
    Patch Qwen3MLP and Qwen3MoeMLP to use fused gate_up GEMM + Triton SwiGLU.

    Args:
        model: Qwen3ForCausalLM or Qwen3MoeForCausalLM
    Returns:
        model (modified in-place)
    """
    mlp_classes = []
    try:
        from baseline.qwen3.modeling import Qwen3MLP
        mlp_classes.append(Qwen3MLP)
    except ImportError:
        pass
    try:
        from baseline.qwen3moe.modeling import Qwen3MoeMLP
        mlp_classes.append(Qwen3MoeMLP)
    except ImportError:
        pass

    if not mlp_classes:
        raise ImportError("Could not import any Qwen3 MLP class")

    mlp_classes = tuple(mlp_classes)

    patched = 0
    for module in model.modules():
        if isinstance(module, mlp_classes):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_qwen_mlp_forward.__get__(module, type(module))
            patched += 1

    model._triton_qwen_dense_mlp_patched = True
    model._triton_qwen_dense_mlp_count = patched
    return model


def unpatch_qwen_dense_mlp(model):
    """Restore original Qwen3 MLP implementations."""
    mlp_classes = []
    try:
        from baseline.qwen3.modeling import Qwen3MLP
        mlp_classes.append(Qwen3MLP)
    except ImportError:
        pass
    try:
        from baseline.qwen3moe.modeling import Qwen3MoeMLP
        mlp_classes.append(Qwen3MoeMLP)
    except ImportError:
        pass

    mlp_classes = tuple(mlp_classes) if mlp_classes else ()

    for module in model.modules():
        if mlp_classes and isinstance(module, mlp_classes):
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                del module._original_forward

    model._triton_qwen_dense_mlp_patched = False
    return model
