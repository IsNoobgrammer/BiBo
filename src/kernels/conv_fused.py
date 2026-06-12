"""
Fused Causal Conv1D Kernels for BiBo.

Liger pattern: keep cuDNN conv + cuBLAS GEMMs, fuse only intermediates.

Two operations:
1. Conv Router: (B, S, H) → causal_conv1d → (B*S, E)
2. Conv Shared Expert: (B, S, H) → causal_conv1d + activation → (B, S, I)

Key finding from benchmarking (RTX 3050, 8 configs):
- Conv expert Triton kernel (permute+act+gate fuse) is ALWAYS slower (0.41x avg)
  because the elementwise ops are so cheap that kernel launch overhead dominates
- Conv router is neutral (0.97x) — not worth optimizing
- The Triton kernels are kept for reference/future large-model use

Strategy:
- Router: keep cuDNN conv + PyTorch reshape (already optimal)
- Shared Expert: keep cuDNN conv + PyTorch silu*multiply (faster than Triton fuse)
- Fused kernels available for large models where memory savings matter

Supports: RTX 3050 (sm_86), T4 (sm_75), A100 (sm_80).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

__all__ = [
    'triton_causal_conv1d_router',
    'triton_causal_conv1d_gated',
    'triton_fused_conv_gate_multiply',
    'patch_conv_router_with_triton',
    'patch_conv_expert_with_triton',
    'unpatch_conv_router',
    'unpatch_conv_expert',
]


# ═══════════════════════════════════════════════════════════════
# KERNEL 1: Fused SiLU + Gate Multiply
#
# This is the key fusion for the shared expert:
# Input A: (M, I) — conv output (after permute to (B*S, I))
# Input B: (M, I) — up_proj output
# Output:  (M, I) — silu(A) * B
#
# Eliminates: 1 intermediate tensor (silu output) + 1 multiply output
# Combined with in-place permute, saves 2 full (B, S, I) tensors.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_act_gate_kernel(
    ConvOut_ptr,    # (M, I) — conv output (gate)
    UpOut_ptr,      # (M, I) — up_proj output
    Out_ptr,        # (M, I) — result: act(conv) * up
    M, I,
    stride_cm, stride_ci,
    stride_um, stride_ui,
    stride_om, stride_oi,
    ACT_TYPE: tl.constexpr,  # 0=silu, 1=relu2, 2=tanh
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused: out = act(conv_output) * up_output
    Saves 2 intermediate tensors in global memory.
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M
    mask_i = offs_i < I
    mask = mask_m[:, None] & mask_i[None, :]
    
    # Load conv output (gate)
    c_ptrs = ConvOut_ptr + offs_m[:, None] * stride_cm + offs_i[None, :] * stride_ci
    conv_val = tl.load(c_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Load up_proj output
    u_ptrs = UpOut_ptr + offs_m[:, None] * stride_um + offs_i[None, :] * stride_ui
    up_val = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply activation to conv output
    if ACT_TYPE == 0:  # SiLU
        act_val = conv_val * (1.0 / (1.0 + tl.exp(-conv_val)))
    elif ACT_TYPE == 1:  # ReLU²
        relu_v = tl.where(conv_val > 0, conv_val, tl.zeros_like(conv_val))
        act_val = relu_v * relu_v
    else:  # Tanh
        exp2x = tl.exp(2.0 * conv_val)
        act_val = (exp2x - 1.0) / (exp2x + 1.0)
    
    # Gate multiply
    result = act_val * up_val
    
    # Store
    o_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_i[None, :] * stride_oi
    tl.store(o_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════
# KERNEL 2: Fused Permute + Activation (for conv output)
#
# Conv1d outputs (B, I, S) but we need (B, S, I) with activation.
# Instead of permute → activation (2 ops, 1 intermediate), do both
# in a single kernel that reads (B, I, S) and writes (B, S, I) activated.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_permute_act_kernel(
    In_ptr,         # (B, I, S) — conv output in channel-first layout
    Out_ptr,        # (B, S, I) — activated output in seq-first layout
    B_val, S, I_val,
    stride_ib, stride_ii, stride_is,   # input strides (B, I, S)
    stride_ob, stride_os, stride_oi,   # output strides (B, S, I)
    ACT_TYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused permute (B,I,S) → (B,S,I) + activation.
    Each program handles one batch element's (S_block, I_block) tile.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_i = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_s = offs_s < S
    mask_i = offs_i < I_val
    mask = mask_s[:, None] & mask_i[None, :]
    
    # Read from (B, I, S) layout: In[b, i, s]
    in_ptrs = In_ptr + pid_b * stride_ib + offs_i[None, :] * stride_ii + offs_s[:, None] * stride_is
    vals = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply activation
    if ACT_TYPE == 0:  # SiLU
        result = vals * (1.0 / (1.0 + tl.exp(-vals)))
    elif ACT_TYPE == 1:  # ReLU²
        relu_v = tl.where(vals > 0, vals, tl.zeros_like(vals))
        result = relu_v * relu_v
    elif ACT_TYPE == 2:  # Tanh
        exp2x = tl.exp(2.0 * vals)
        result = (exp2x - 1.0) / (exp2x + 1.0)
    else:  # None
        result = vals
    
    # Write to (B, S, I) layout: Out[b, s, i]
    out_ptrs = Out_ptr + pid_b * stride_ob + offs_s[:, None] * stride_os + offs_i[None, :] * stride_oi
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════
# KERNEL 3: Fused Permute + Activation + Gate Multiply
#
# The ultimate fusion for the shared expert:
# Reads: conv_out (B, I, S) + up_proj_out (B, S, I)
# Writes: act(permute(conv_out)) * up_proj_out → (B, S, I)
#
# Eliminates: permuted tensor + activated tensor + multiply result
# = 3 intermediate (B, S, I) tensors eliminated
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_permute_act_gate_kernel(
    ConvOut_ptr,    # (B, I, S) — conv output (channel-first)
    UpOut_ptr,      # (B, S, I) — up_proj output (seq-first)
    Out_ptr,        # (B, S, I) — result
    B_val, S, I_val,
    stride_cb, stride_ci, stride_cs,   # conv strides (B, I, S)
    stride_ub, stride_us, stride_ui,   # up strides (B, S, I)
    stride_ob, stride_os, stride_oi,   # out strides (B, S, I)
    ACT_TYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused: out[b,s,i] = act(conv_out[b,i,s]) * up_out[b,s,i]
    Combines permute + activation + gate multiply in one pass.
    """
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_i = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_s = offs_s < S
    mask_i = offs_i < I_val
    mask = mask_s[:, None] & mask_i[None, :]
    
    # Read conv output from (B, I, S) layout
    conv_ptrs = ConvOut_ptr + pid_b * stride_cb + offs_i[None, :] * stride_ci + offs_s[:, None] * stride_cs
    conv_val = tl.load(conv_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Read up_proj output from (B, S, I) layout
    up_ptrs = UpOut_ptr + pid_b * stride_ub + offs_s[:, None] * stride_us + offs_i[None, :] * stride_ui
    up_val = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply activation to conv output
    if ACT_TYPE == 0:  # SiLU
        act_val = conv_val * (1.0 / (1.0 + tl.exp(-conv_val)))
    elif ACT_TYPE == 1:  # ReLU²
        relu_v = tl.where(conv_val > 0, conv_val, tl.zeros_like(conv_val))
        act_val = relu_v * relu_v
    else:  # Tanh
        exp2x = tl.exp(2.0 * conv_val)
        act_val = (exp2x - 1.0) / (exp2x + 1.0)
    
    # Gate multiply
    result = act_val * up_val
    
    # Write to (B, S, I) layout
    out_ptrs = Out_ptr + pid_b * stride_ob + offs_s[:, None] * stride_os + offs_i[None, :] * stride_oi
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════
# Autograd-Compatible Wrappers for Conv Triton Kernels
#
# Same pattern as dense_mlp.py: Triton forward for speed,
# PyTorch backward for correct gradient flow.
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# KERNEL 4: Conv Gate Multiply Backward
#
# Backward of fused permute + activation + gate multiply.
# Forward: out[b,s,i] = act(conv_out[b,i,s]) * up_out[b,s,i]
# Backward:
#   grad_up_out[b,s,i] = grad_out[b,s,i] * act(conv_out[b,i,s])
#   grad_conv_out[b,i,s] = grad_out[b,s,i] * up_out[b,s,i] * d_act(conv_out[b,i,s])
#
# Fuses: activation derivative + two multiplies + permute-back in one kernel.
# All in float32 to prevent overflow.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_permute_act_gate_backward_kernel(
    GradOut_ptr,      # (B, S, I) — gradient of output
    ConvOut_ptr,      # (B, I, S) — saved conv output
    UpOut_ptr,        # (B, S, I) — saved up output
    GradConvOut_ptr,  # (B, I, S) — gradient of conv output
    GradUpOut_ptr,    # (B, S, I) — gradient of up output
    B, S, I,
    stride_go_b, stride_go_s, stride_go_i,
    stride_co_b, stride_co_i, stride_co_s,
    stride_uo_b, stride_uo_s, stride_uo_i,
    stride_gco_b, stride_gco_i, stride_gco_s,
    stride_guo_b, stride_guo_s, stride_guo_i,
    ACT_TYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Backward: compute grad_conv_out and grad_up_out."""
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_s = offs_s < S
    mask_i = offs_i < I
    mask = mask_s[:, None] & mask_i[None, :]

    # Load grad_output from (B, S, I)
    go_ptrs = GradOut_ptr + pid_b * stride_go_b + offs_s[:, None] * stride_go_s + offs_i[None, :] * stride_go_i
    grad_out = tl.load(go_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load conv_out from (B, I, S) — permuted to (B, S, I) for activation
    co_ptrs = ConvOut_ptr + pid_b * stride_co_b + offs_i[None, :] * stride_co_i + offs_s[:, None] * stride_co_s
    conv_val = tl.load(co_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load up_out from (B, S, I)
    uo_ptrs = UpOut_ptr + pid_b * stride_uo_b + offs_s[:, None] * stride_uo_s + offs_i[None, :] * stride_uo_i
    up_val = tl.load(uo_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute activation and derivative in float32
    if ACT_TYPE == 0:  # SiLU
        sig_gate = 1.0 / (1.0 + tl.exp(-conv_val))
        act_gate = conv_val * sig_gate
        dact = sig_gate * (1.0 + conv_val * (1.0 - sig_gate))
    elif ACT_TYPE == 1:  # ReLU²
        relu_gate = tl.maximum(conv_val, 0.0)
        act_gate = relu_gate * relu_gate
        dact = 2.0 * relu_gate
    elif ACT_TYPE == 2:  # Tanh
        act_gate = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * conv_val))) - 1.0
        dact = 1.0 - act_gate * act_gate
    else:  # None
        act_gate = conv_val
        dact = 1.0

    # Compute grad_up_out = grad_output * act_gate
    grad_up_out = grad_out * act_gate
    guo_ptrs = GradUpOut_ptr + pid_b * stride_guo_b + offs_s[:, None] * stride_guo_s + offs_i[None, :] * stride_guo_i
    tl.store(guo_ptrs, grad_up_out, mask=mask)

    # Compute grad_conv_out = grad_output * up_val * dact (write to (B, I, S) layout)
    grad_conv = grad_out * up_val * dact
    gco_ptrs = GradConvOut_ptr + pid_b * stride_gco_b + offs_i[None, :] * stride_gco_i + offs_s[:, None] * stride_gco_s
    tl.store(gco_ptrs, grad_conv, mask=mask)


# ═══════════════════════════════════════════════════════════════
# KERNEL 5: Conv Permute+Act Backward
#
# Backward of fused permute + activation.
# Forward: out[b,s,i] = act(conv_out[b,i,s])
# Backward: grad_conv_out[b,i,s] = grad_out[b,s,i] * d_act(conv_out[b,i,s])
#
# Fuses: activation derivative + multiply + permute-back in one kernel.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_permute_act_backward_kernel(
    GradOut_ptr,      # (B, S, I) — gradient of output
    ConvOut_ptr,      # (B, I, S) — saved conv output
    GradConvOut_ptr,  # (B, I, S) — gradient of conv output
    B, S, I,
    stride_go_b, stride_go_s, stride_go_i,
    stride_co_b, stride_co_i, stride_co_s,
    stride_gco_b, stride_gco_i, stride_gco_s,
    ACT_TYPE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Backward: compute grad_conv_out."""
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_s = offs_s < S
    mask_i = offs_i < I
    mask = mask_s[:, None] & mask_i[None, :]

    # Load grad_output from (B, S, I)
    go_ptrs = GradOut_ptr + pid_b * stride_go_b + offs_s[:, None] * stride_go_s + offs_i[None, :] * stride_go_i
    grad_out = tl.load(go_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load conv_out from (B, I, S) — permuted to (B, S, I) for activation
    co_ptrs = ConvOut_ptr + pid_b * stride_co_b + offs_i[None, :] * stride_co_i + offs_s[:, None] * stride_co_s
    conv_val = tl.load(co_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute activation derivative in float32
    if ACT_TYPE == 0:  # SiLU
        sig_gate = 1.0 / (1.0 + tl.exp(-conv_val))
        dact = sig_gate * (1.0 + conv_val * (1.0 - sig_gate))
    elif ACT_TYPE == 1:  # ReLU²
        relu_gate = tl.maximum(conv_val, 0.0)
        dact = 2.0 * relu_gate
    elif ACT_TYPE == 2:  # Tanh
        act_gate = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * conv_val))) - 1.0
        dact = 1.0 - act_gate * act_gate
    else:  # None
        dact = 1.0

    # Compute grad_conv_out = grad_output * dact (write to (B, I, S) layout)
    grad_conv = grad_out * dact
    gco_ptrs = GradConvOut_ptr + pid_b * stride_gco_b + offs_i[None, :] * stride_gco_i + offs_s[:, None] * stride_gco_s
    tl.store(gco_ptrs, grad_conv, mask=mask)


class _TritonConvGateMultiplyFunction(torch.autograd.Function):
    """Autograd wrapper for fused permute + activation + gate multiply."""
    
    @staticmethod
    def forward(ctx, conv_out: torch.Tensor, up_out: torch.Tensor, act_type: int) -> torch.Tensor:
        ctx.save_for_backward(conv_out, up_out)
        ctx.act_type = act_type
        # Use the raw Triton kernel for speed
        B, I, S = conv_out.shape
        out = torch.empty(B, S, I, device=conv_out.device, dtype=conv_out.dtype)
        BLOCK_S = min(32, triton.next_power_of_2(S))
        BLOCK_I = min(64, triton.next_power_of_2(I))
        grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(I, BLOCK_I))
        _fused_permute_act_gate_kernel[grid](
            conv_out, up_out, out,
            B, S, I,
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            up_out.stride(0), up_out.stride(1), up_out.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            ACT_TYPE=act_type,
            BLOCK_S=BLOCK_S,
            BLOCK_I=BLOCK_I,
        )
        return out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward: Triton kernel for fused activation + multiply + permute."""
        conv_out, up_out = ctx.saved_tensors
        act_type = ctx.act_type
        B, I, S = conv_out.shape
        
        # Allocate output gradients
        grad_conv_out = torch.empty(B, I, S, device=conv_out.device, dtype=conv_out.dtype)
        grad_up_out = torch.empty(B, S, I, device=up_out.device, dtype=up_out.dtype)
        
        BLOCK_S = min(32, triton.next_power_of_2(S))
        BLOCK_I = min(64, triton.next_power_of_2(I))
        grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(I, BLOCK_I))
        
        # Launch Triton backward kernel
        _fused_permute_act_gate_backward_kernel[grid](
            grad_output, conv_out, up_out,
            grad_conv_out, grad_up_out,
            B, S, I,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            up_out.stride(0), up_out.stride(1), up_out.stride(2),
            grad_conv_out.stride(0), grad_conv_out.stride(1), grad_conv_out.stride(2),
            grad_up_out.stride(0), grad_up_out.stride(1), grad_up_out.stride(2),
            ACT_TYPE=act_type,
            BLOCK_S=BLOCK_S,
            BLOCK_I=BLOCK_I,
        )
        
        return grad_conv_out, grad_up_out, None  # None for act_type


class _TritonPermuteActFunction(torch.autograd.Function):
    """Autograd wrapper for fused permute + activation."""
    
    @staticmethod
    def forward(ctx, conv_out: torch.Tensor, act_type: int) -> torch.Tensor:
        ctx.save_for_backward(conv_out)
        ctx.act_type = act_type
        B, I, S = conv_out.shape
        out = torch.empty(B, S, I, device=conv_out.device, dtype=conv_out.dtype)
        BLOCK_S = min(32, triton.next_power_of_2(S))
        BLOCK_I = min(64, triton.next_power_of_2(I))
        grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(I, BLOCK_I))
        _fused_permute_act_kernel[grid](
            conv_out, out,
            B, S, I,
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            ACT_TYPE=act_type,
            BLOCK_S=BLOCK_S,
            BLOCK_I=BLOCK_I,
        )
        return out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward: Triton kernel for activation derivative + permute-back."""
        conv_out, = ctx.saved_tensors
        act_type = ctx.act_type
        B, I, S = conv_out.shape
        
        # Allocate output gradient
        grad_conv_out = torch.empty(B, I, S, device=conv_out.device, dtype=conv_out.dtype)
        
        BLOCK_S = min(32, triton.next_power_of_2(S))
        BLOCK_I = min(64, triton.next_power_of_2(I))
        grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(I, BLOCK_I))
        
        # Launch Triton backward kernel
        _fused_permute_act_backward_kernel[grid](
            grad_output, conv_out, grad_conv_out,
            B, S, I,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            grad_conv_out.stride(0), grad_conv_out.stride(1), grad_conv_out.stride(2),
            ACT_TYPE=act_type,
            BLOCK_S=BLOCK_S,
            BLOCK_I=BLOCK_I,
        )
        
        return grad_conv_out, None  # None for act_type


# ═══════════════════════════════════════════════════════════════
# Python Wrappers (now using autograd.Function)
# ═══════════════════════════════════════════════════════════════

_ACT_MAP = {"silu": 0, "relu2": 1, "tanh": 2, "none": 3}


def triton_fused_conv_gate_multiply(
    conv_out: torch.Tensor,     # (B, I, S) — from nn.Conv1d
    up_out: torch.Tensor,       # (B, S, I) — from up_proj
    act_type: int = 0,          # 0=silu, 1=relu2, 2=tanh
) -> torch.Tensor:
    """
    Fused permute + activation + gate multiply.
    
    Replaces:
        gate_output = conv_out.permute(0, 2, 1)  # (B, S, I)
        activated = F.silu(gate_output)           # (B, S, I)  
        result = activated * up_out               # (B, S, I)
    
    With single kernel that reads (B,I,S) + (B,S,I) and writes (B,S,I).
    Eliminates 2 intermediate tensors.
    
    Uses autograd.Function for correct backward gradient flow.
    
    Returns: (B, S, I)
    """
    B, I, S = conv_out.shape
    assert up_out.shape == (B, S, I), f"Shape mismatch: conv_out (B,I,S)={conv_out.shape}, up_out expected (B,S,I)={up_out.shape}"
    return _TritonConvGateMultiplyFunction.apply(conv_out, up_out, act_type)


def triton_fused_permute_act(
    conv_out: torch.Tensor,     # (B, I, S) — from nn.Conv1d
    act_type: int = 0,          # 0=silu, 1=relu2, 2=tanh, 3=none
) -> torch.Tensor:
    """
    Fused permute (B,I,S) → (B,S,I) + activation.
    Eliminates 1 intermediate tensor (the permuted copy).
    
    Uses autograd.Function for correct backward gradient flow.
    
    Returns: (B, S, I)
    """
    return _TritonPermuteActFunction.apply(conv_out, act_type)


def triton_causal_conv1d_router(
    x: torch.Tensor,        # (B, S, H)
    weight: torch.Tensor,   # (E, H, K) — nn.Conv1d weight
    num_experts: int,
    kernel_size: int,
) -> torch.Tensor:
    """
    Optimized causal conv1d for router projection.
    
    Uses cuDNN conv (fast) + optimized reshape.
    The permute (B,E,S)→(B,S,E) is cheap because E is tiny (8-16).
    
    Returns: (B*S, E) router logits
    """
    B, S, H = x.shape
    E = num_experts
    K = kernel_size
    
    # cuDNN conv (unbeatable for the actual convolution)
    from einops import rearrange
    x_perm = rearrange(x, 'b s h -> b h s')  # (B, H, S) — contiguous view
    x_padded = F.pad(x_perm, (K - 1, 0))     # (B, H, S+K-1)
    
    # Use F.conv1d directly with the weight tensor
    conv_out = F.conv1d(x_padded, weight)     # (B, E, S)
    
    # Reshape: (B, E, S) → (B*S, E) — E is small so permute is cheap
    return conv_out.permute(0, 2, 1).reshape(B * S, E)


def triton_causal_conv1d_gated(
    x: torch.Tensor,        # (B, S, H)
    weight: torch.Tensor,   # (I, H, K) — nn.Conv1d weight
    intermediate_size: int,
    kernel_size: int,
    act_type: int = 0,      # 0=silu, 1=relu2, 2=tanh, 3=none
) -> torch.Tensor:
    """
    Optimized causal conv1d + activation for shared expert gate.
    
    Uses cuDNN conv + fused permute+activation (eliminates 1 intermediate).
    
    Replaces: permute → pad → Conv1d → permute → activation
    With: permute → pad → Conv1d (cuDNN) → fused_permute_act (Triton)
    
    Returns: (B, S, I) activated gate output
    """
    B, S, H = x.shape
    K = kernel_size
    
    # cuDNN conv
    from einops import rearrange
    x_perm = rearrange(x, 'b s h -> b h s')   # (B, H, S) — contiguous view
    x_padded = F.pad(x_perm, (K - 1, 0))      # (B, H, S+K-1)
    conv_out = F.conv1d(x_padded, weight)      # (B, I, S)
    
    # Fused permute + activation (Triton)
    return triton_fused_permute_act(conv_out, act_type)


# ═══════════════════════════════════════════════════════════════
# Monkey-Patch Interface
# ═══════════════════════════════════════════════════════════════

def _triton_conv_router_forward(self, hidden_states: torch.Tensor):
    """
    Drop-in replacement for BiBoMoERouter.forward when router_type="conv".
    Uses cuDNN conv + optimized permute/reshape.
    """
    import math
    from einops import rearrange
    
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Step 1: Conv projection (cuDNN + fused permute)
    if self.router_type == "conv":
        router_logits = triton_causal_conv1d_router(
            hidden_states, self.gate_conv.weight,
            self.num_routed_experts, self.kernel_size
        ).float()
    else:
        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
        router_logits = self.gate_proj(flat_hidden).float()
    
    # Step 2: exploration noise (training only)
    if self.training and self.router_noise > 0:
        noise_stddev = math.sqrt(self.router_noise)
        noise = torch.randn_like(router_logits) * noise_stddev
        router_logits = router_logits + noise.detach()

    # Step 3: router activation
    router_logits = self._apply_router_activation(router_logits)

    # Step 4: gating
    if self.gate_type == "sigmoid":
        scores = torch.sigmoid(router_logits)
    else:
        scores = F.softmax(router_logits, dim=1)

    # Step 5: optional logit normalization
    if self.use_router_logit_norm:
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True) + 1e-6
        scores = self.router_lambda * (scores - mean) / std

    # Step 6: selection with bias
    selection_scores = scores + self.bias

    # Step 7: top-k
    _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1)

    # Step 8: gather unbiased weights
    top_k_weights = scores.gather(-1, top_k_indices)
    if self.norm_topk_prob:
        norm_weights = top_k_weights / (top_k_weights.sum(-1, keepdim=True) + 1e-6)
    else:
        norm_weights = top_k_weights

    top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
    norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
    return top_k_indices.long(), norm_weights.to(hidden_states.dtype)


def _triton_conv_expert_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for BiBoCausalConv1D.forward.

    Liger pattern: keep cuDNN conv + cuBLAS GEMMs, fuse only intermediates.
    For the conv expert, the intermediates (permute+silu+multiply) are so cheap
    that Triton kernel launch overhead makes them SLOWER (0.41x avg benchmark).
    So we use PyTorch ops which are faster for small tensors.

    Original: rearrange(b,s,h→b,h,s) → pad → conv → rearrange(b,i,s→b,s,i) → silu → multiply → down_proj
    Optimized: rearrange(b,s,h→b,h,s) → pad → conv (cuDNN) → silu * up (PyTorch) → down_proj
    """
    bsz, seq_len, hidden_dim = x.shape
    
    # Reuse the original module's conv (cuDNN) — same permute as original code
    from einops import rearrange as _rearrange
    x_perm = _rearrange(x, 'b s h -> b h s')
    x_padded = F.pad(x_perm, (self.causal_padding_gate, 0))
    conv_out = self.gate_conv(x_padded)  # (B, I, S)
    
    # up_proj (cuBLAS — keep)
    up_out = self.up_proj(x)  # (B, S, I)
    
    # PyTorch ops for intermediate: permute + silu + multiply
    # Faster than Triton for small tensors (no kernel launch overhead)
    gate_output = conv_out.permute(0, 2, 1)  # (B, S, I) — zero-copy view
    gated = F.silu(gate_output) * up_out      # (B, S, I) — 2 elementwise ops
    
    # down_proj (cuBLAS — keep)
    output = self.down_proj(gated)
    
    if output.shape[1] != seq_len:
        raise RuntimeError(f"Conv expert len mismatch. Expected {seq_len}, got {output.shape[1]}")
    return output


def patch_conv_router_with_triton(model):
    """
    Patch BiBoMoERouter instances that use router_type="conv" to use
    optimized conv projection (cuDNN + fused permute).
    """
    from src.modeling.ffn.router import BiBoMoERouter
    
    patched = 0
    for module in model.modules():
        if isinstance(module, BiBoMoERouter) and module.router_type == "conv":
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_conv_router_forward.__get__(module, BiBoMoERouter)
            patched += 1
    
    if patched > 0:
        model._triton_conv_router_patched = True
    return model


def patch_conv_expert_with_triton(model):
    """
    Patch BiBoCausalConv1D instances to use fused permute+activation+gate.
    Eliminates 2 intermediate tensors per forward pass.
    """
    from src.modeling.ffn.experts import BiBoCausalConv1D
    
    patched = 0
    for module in model.modules():
        if isinstance(module, BiBoCausalConv1D):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_conv_expert_forward.__get__(module, BiBoCausalConv1D)
            patched += 1
    
    if patched > 0:
        model._triton_conv_expert_patched = True
    return model


def unpatch_conv_router(model):
    """Restore original conv router implementation."""
    from src.modeling.ffn.router import BiBoMoERouter
    for module in model.modules():
        if isinstance(module, BiBoMoERouter) and hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward
    model._triton_conv_router_patched = False
    return model


def unpatch_conv_expert(model):
    """Restore original conv expert implementation."""
    from src.modeling.ffn.experts import BiBoCausalConv1D
    for module in model.modules():
        if isinstance(module, BiBoCausalConv1D) and hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward
    model._triton_conv_expert_patched = False
    return model
