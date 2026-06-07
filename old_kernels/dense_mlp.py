"""
Fused Dense MLP Kernel — Optimized SwiGLU for BiBo's dense layers.

BiBo uses dense MLP (SwiGLU) on layers 0, 1, and N-1 (mlp_only_layers).
The forward is: down_proj(silu(gate_proj(x)) * up_proj(x))

Baseline (PyTorch eager):
    gate = gate_proj(x)          # (B*S, I) — cuBLAS GEMM
    up = up_proj(x)              # (B*S, I) — cuBLAS GEMM
    activated = silu(gate)       # (B*S, I) — elementwise, 1 intermediate
    gated = activated * up       # (B*S, I) — elementwise, 1 intermediate
    out = down_proj(gated)       # (B*S, H) — cuBLAS GEMM

Optimization strategy:
    - cuBLAS for all 3 GEMMs (already optimal, don't touch)
    - Fuse gate_proj + up_proj into single GEMM: fused_gate_up_proj(x) → (B*S, 2*I)
    - Triton kernel: split → SiLU(gate) * up → (B*S, I) in 1 pass
    - This eliminates 2 intermediate tensors (gate, up) and 1 activation output

This is essentially the same kernel as moe_dispatch._fused_glu_act_kernel
but specialized for the dense MLP case (always SiLU, no activation dispatch).

Memory savings per dense layer forward:
    Before: gate(B*S*I) + up(B*S*I) + silu_out(B*S*I) = 3 * B*S*I
    After:  gate_up(B*S*2I) + out(B*S*I) = 3 * B*S*I (same total, but 1 fewer kernel launch)
    Real win: fused kernel avoids materializing silu_out separately

Performance win:
    - 1 fewer kernel launch (silu is separate in eager)
    - Better cache utilization (gate and up loaded once, consumed immediately)
    - No intermediate tensor allocation/deallocation overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

__all__ = ['patch_dense_mlp_with_triton', 'unpatch_dense_mlp',
           'patch_qwen_dense_mlp_with_triton', 'unpatch_qwen_dense_mlp',
           'triton_fused_swiglu', '_TritonSwiGLUFunction', '_TritonFusedGLUFunction']


# ═══════════════════════════════════════════════════════════════
# KERNEL: Fused SwiGLU Activation
#
# Input: (M, 2*I) — concatenated gate_up output from fused linear
# Output: (M, I) — silu(gate) * up
#
# Eliminates: separate silu kernel + separate multiply kernel
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
def _fused_swiglu_kernel(
    GateUp_ptr,     # (M, 2*I) — fused gate+up from linear
    Out_ptr,        # (M, I) — silu(gate) * up
    M, I,           # M=num_tokens, I=intermediate_size
    stride_gu_m, stride_gu_i,
    stride_o_m, stride_o_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused SwiGLU: split gate_up → silu(gate) * up.
    
    Input layout: [gate_0..gate_I-1, up_0..up_I-1] per row
    Gate is first I columns, Up is second I columns.
    """
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
    
    # Load up (second I columns, offset by I)
    up_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + (I + offs_i)[None, :] * stride_gu_i
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # SiLU(gate) * up — all in registers, no intermediate materialization
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    result = silu_gate * up
    
    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_o_m + offs_i[None, :] * stride_o_i
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════
# Python Wrapper
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
    
    _fused_swiglu_kernel[grid](
        gate_up, out,
        M, I,
        gate_up.stride(0), gate_up.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Autograd-Compatible Wrapper for Triton SwiGLU
#
# The raw Triton kernel writes into torch.empty() — autograd can't
# trace through that. This wrapper uses torch.autograd.Function to:
# - Forward: run Triton kernel (fast, no intermediates)
# - Backward: recompute gate/up from saved gate_up, compute grads
#   using PyTorch ops (correct, autograd-compatible)
#
# This gives us the forward speedup + memory savings while keeping
# backward correctness for training.
# ═══════════════════════════════════════════════════════════════

class _TritonSwiGLUFunction(torch.autograd.Function):
    """Autograd wrapper for Triton fused SwiGLU: silu(gate) * up."""
    
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Forward: use Triton kernel for speed.
        Input: (M, 2*I) — concatenated gate and up
        Output: (M, I) — silu(gate) * up
        """
        ctx.save_for_backward(gate_up)
        return triton_fused_swiglu(gate_up)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward: recompute from saved gate_up using PyTorch ops.
        
        f(gate_up) = silu(gate) * up
        df/d(gate_up) = [df/d(gate), df/d(up)]
        
        df/d(gate) = grad_output * up * d(silu(gate))/d(gate)
                   = grad_output * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
                   = grad_output * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        df/d(up) = grad_output * silu(gate)
        """
        gate_up, = ctx.saved_tensors
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        
        # Compute sigmoid and silu
        sig_gate = torch.sigmoid(gate)
        silu_gate = gate * sig_gate
        
        # Gradient w.r.t. up: grad_output * silu(gate)
        grad_up = grad_output * silu_gate
        
        # Gradient w.r.t. gate: grad_output * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        # = grad_output * up * (silu(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
        # Simplified: d(silu)/d(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        dsilu = sig_gate * (1.0 + gate * (1.0 - sig_gate))
        grad_gate = grad_output * up * dsilu
        
        # Concatenate back to (M, 2*I)
        grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)
        return grad_gate_up


class _TritonFusedGLUFunction(torch.autograd.Function):
    """Autograd wrapper for Triton fused GLU with variable activation (SiLU/ReLU²/Tanh)."""
    
    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, act_type: int) -> torch.Tensor:
        """
        Forward: use Triton kernel for speed.
        Input: (M, 2*I) — concatenated gate and up
        Output: (M, I) — act(gate) * up
        """
        ctx.save_for_backward(gate_up)
        ctx.act_type = act_type
        from .moe_dispatch import triton_fused_glu_activation
        return triton_fused_glu_activation(gate_up, act_type)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward: recompute from saved gate_up using PyTorch ops.
        Supports SiLU (0), ReLU² (1), Tanh (2).
        """
        gate_up, = ctx.saved_tensors
        act_type = ctx.act_type
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        
        gate = gate_up[:, :I]
        up = gate_up[:, I:]
        
        if act_type == 0:  # SiLU
            sig_gate = torch.sigmoid(gate)
            act_gate = gate * sig_gate
            # d(silu)/d(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            dact = sig_gate * (1.0 + gate * (1.0 - sig_gate))
        elif act_type == 1:  # ReLU²
            relu_gate = F.relu(gate)
            act_gate = relu_gate * relu_gate
            # d(relu²)/d(x) = 2 * relu(x) * (x > 0) = 2 * relu(x)
            dact = 2.0 * relu_gate
        else:  # Tanh
            act_gate = torch.tanh(gate)
            # d(tanh)/d(x) = 1 - tanh²(x)
            dact = 1.0 - act_gate * act_gate
        
        # Gradient w.r.t. up: grad_output * act(gate)
        grad_up = grad_output * act_gate
        
        # Gradient w.r.t. gate: grad_output * up * d(act)/d(gate)
        grad_gate = grad_output * up * dact
        
        # Concatenate back to (M, 2*I)
        grad_gate_up = torch.cat([grad_gate, grad_up], dim=-1)
        return grad_gate_up, None  # None for act_type


# ═══════════════════════════════════════════════════════════════
# Fused Dense MLP Module
#
# Replaces BiBoMLP with a version that:
# 1. Uses a fused gate_up linear (1 GEMM instead of 2)
# 2. Uses Triton fused SwiGLU activation
# 3. Uses standard down_proj linear
# ═══════════════════════════════════════════════════════════════

class BiBoFusedDenseMLP(nn.Module):
    """
    Fused Dense MLP with combined gate_up projection + Triton SwiGLU.
    
    Architecture:
        x → fused_gate_up_proj(x) → triton_swiglu → down_proj → out
        
    vs original:
        x → gate_proj(x) → silu → * up_proj(x) → down_proj → out
        
    Benefits:
        - 1 GEMM (gate_up) instead of 2 separate GEMMs (gate + up)
        - Triton fused activation eliminates intermediate tensors
        - Same mathematical result, fewer kernel launches
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int,
                 gate_proj_weight: torch.Tensor,
                 up_proj_weight: torch.Tensor,
                 down_proj: nn.Linear):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Fused gate_up weight: (2*I, H) — gate on top, up on bottom
        # This allows a single F.linear call to produce (M, 2*I)
        fused_weight = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.gate_up_proj.weight = nn.Parameter(fused_weight)
        
        # Down projection stays as-is
        self.down_proj = down_proj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for 2D operation
        orig_shape = x.shape[:-1]
        x_2d = x.view(-1, self.hidden_size)
        
        # 1. Fused gate+up GEMM: (M, H) → (M, 2*I)
        gate_up = F.linear(x_2d, self.gate_up_proj.weight)
        
        # 2. Triton fused SwiGLU: (M, 2*I) → (M, I)
        intermediate = triton_fused_swiglu(gate_up)
        
        # 3. Down GEMM: (M, I) → (M, H)
        out = F.linear(intermediate, self.down_proj.weight)
        
        return out.view(*orig_shape, self.hidden_size)


# ═══════════════════════════════════════════════════════════════
# Monkey-Patch Interface
# ═══════════════════════════════════════════════════════════════

def _triton_dense_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for BiBoMLP.forward using fused gate_up + Triton SwiGLU.
    
    Concatenates gate_proj.weight and up_proj.weight ON EVERY FORWARD so that
    autograd traces through the actual parameters (enabling gradient flow).
    
    The torch.cat is cheap (~0.01ms) compared to the GEMM it feeds.
    """
    orig_shape = x.shape[:-1]
    x_2d = x.view(-1, self.hidden_size)
    
    # Fused gate+up: concatenate LIVE parameters (not a buffer!) for autograd
    # gate_proj.weight: (I, H), up_proj.weight: (I, H)
    # fused: (2*I, H) → F.linear(x, fused) → (M, 2*I)
    fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
    gate_up = F.linear(x_2d, fused_weight)
    
    # Triton fused SwiGLU: (M, 2*I) → (M, I)
    # Use autograd-compatible wrapper that preserves gradient flow
    intermediate = _TritonSwiGLUFunction.apply(gate_up)
    
    # Down projection: (M, I) → (M, H)
    out = self.down_proj(intermediate)
    
    return out.view(*orig_shape, self.hidden_size)


def patch_dense_mlp_with_triton(model):
    """
    Patch BiBoMLP (dense layers) to use fused gate_up GEMM + Triton SwiGLU.
    
    What's fused:
    - gate_proj + up_proj → single fused GEMM (1 kernel launch instead of 2)
    - SiLU activation + gate*up multiply → 1 Triton kernel (eliminates intermediates)
    
    What's NOT fused (cuBLAS is already optimal):
    - down_proj linear (GEMM)
    
    Only patches non-MoE layers (those using BiBoMLP with is_expert=False).
    
    IMPORTANT: The fused weight is recomputed from live parameters on every forward
    to preserve gradient flow. The torch.cat cost (~0.01ms) is negligible vs GEMM.
    
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
            # Clean up legacy buffer if present (from old buggy version)
            if hasattr(module, '_fused_gate_up_weight'):
                del module._fused_gate_up_weight
    
    model._triton_dense_mlp_patched = False
    return model


# ═══════════════════════════════════════════════════════════════
# Qwen3 / Qwen3MoE Dense MLP Patching
#
# Both Qwen3MLP and Qwen3MoeMLP use the exact same SwiGLU pattern:
#   down_proj(act_fn(gate_proj(x)) * up_proj(x))
# Same attributes: gate_proj, up_proj, down_proj, hidden_size, intermediate_size
# ═══════════════════════════════════════════════════════════════

def _triton_qwen_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for Qwen3MLP/Qwen3MoeMLP.forward using fused gate_up + Triton SwiGLU.
    
    Concatenates live parameters on every forward for correct gradient flow.
    """
    orig_shape = x.shape[:-1]
    x_2d = x.view(-1, self.hidden_size)
    
    # Fused gate+up GEMM from LIVE parameters: (M, H) → (M, 2*I)
    fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
    gate_up = F.linear(x_2d, fused_weight)
    
    # Triton fused SwiGLU with autograd: (M, 2*I) → (M, I)
    intermediate = _TritonSwiGLUFunction.apply(gate_up)
    
    # Down projection: (M, I) → (M, H)
    out = self.down_proj(intermediate)
    
    return out.view(*orig_shape, self.hidden_size)


def patch_qwen_dense_mlp_with_triton(model):
    """
    Patch Qwen3MLP and Qwen3MoeMLP to use fused gate_up GEMM + Triton SwiGLU.
    
    Works on both Qwen3ForCausalLM (all layers are dense MLP) and
    Qwen3MoeForCausalLM (dense layers use Qwen3MoeMLP).
    
    What's fused:
    - gate_proj + up_proj → single fused GEMM (1 kernel launch instead of 2)
    - SiLU activation + gate*up multiply → 1 Triton kernel (eliminates intermediates)
    
    IMPORTANT: Fused weight is recomputed from live parameters on every forward
    to preserve gradient flow for training.
    
    Args:
        model: Qwen3ForCausalLM or Qwen3MoeForCausalLM
    Returns:
        model (modified in-place)
    """
    # Collect all MLP classes we can patch
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
            # Clean up legacy buffer if present
            if hasattr(module, '_fused_gate_up_weight'):
                del module._fused_gate_up_weight
    
    model._triton_qwen_dense_mlp_patched = False
    return model
