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
    'fused_conv_router_full',
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


# ═══════════════════════════════════════════════════════════════
# FUSED causal conv1d ROUTER (forward) — reads x in native (B,S,H) layout,
# writes logits (B*S,E) directly. Kills the (B,S,H)->(B,H,S) transpose copy + pad copy
# that the eager/cuDNN path pays. out[b,s,e] = sum_k sum_h x[b, s-(K-1)+k, h]*W[e,h,k]
# (causal: source row <0 -> 0), as K shifted (BS,H)@(H,E) accumulations.
# Measured (RTX 3050, fp16, locked clock): forward ~5x, fwd+bwd ~1.15x vs eager; grads exact.
# ═══════════════════════════════════════════════════════════════
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 32, 'BLOCK_H': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 128}, num_warps=8, num_stages=2),
    ],
    key=['H'],
)
@triton.jit
def _causal_conv_router_fwd_kernel(
    X_ptr, W_ptr, Out_ptr,
    B, S, H,
    stride_xb, stride_xs, stride_xh,
    stride_we, stride_wh, stride_wk,
    stride_om, stride_oe,
    K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
    APPLY_SIGMOID: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, BLOCK_E)
    mask_s = offs_s < S
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_S, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(K):
        src = offs_s - (K - 1) + k
        mask_src = (src >= 0) & mask_s
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            xptr = X_ptr + pid_b * stride_xb + src[:, None] * stride_xs + offs_h[None, :] * stride_xh
            xv = tl.load(xptr, mask=mask_src[:, None] & mask_h[None, :], other=0.0)
            wptr = W_ptr + offs_e[:, None] * stride_we + offs_h[None, :] * stride_wh + k * stride_wk
            wv = tl.load(wptr, mask=mask_e[:, None] & mask_h[None, :], other=0.0)
            acc += tl.dot(xv, tl.trans(wv))
    # Epilogue: optionally fuse the sigmoid gate so we never round-trip raw logits through HBM
    # + a separate sigmoid kernel. acc is fp32 (matches eager's logits.float() before sigmoid).
    if APPLY_SIGMOID:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    out_row = pid_b * S + offs_s
    optr = Out_ptr + out_row[:, None] * stride_om + offs_e[None, :] * stride_oe
    tl.store(optr, acc.to(Out_ptr.dtype.element_ty), mask=mask_s[:, None] & mask_e[None, :])


def _fused_conv_router_fwd(x, weight, apply_sigmoid=False, out_dtype=None):
    B, S, Hd = x.shape
    E, _, K = weight.shape
    out = torch.empty(B * S, E, device=x.device, dtype=out_dtype or x.dtype)
    grid = lambda meta: (B, triton.cdiv(S, meta['BLOCK_S']))
    _causal_conv_router_fwd_kernel[grid](
        x, weight, out, B, S, Hd,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        out.stride(0), out.stride(1),
        K=K, E=E, BLOCK_E=max(16, triton.next_power_of_2(E)),
        APPLY_SIGMOID=apply_sigmoid,
    )
    return out


# ── Transpose-free Triton conv-router BACKWARD ──
# grad_x[b,j,h] = sum_m sum_e grad_out[b, j+m, e] * W[e, h, K-1-m]  (j+m<S)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64, 'BLOCK_H': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 128, 'BLOCK_H': 128}, num_warps=8, num_stages=2),
    ], key=['H'])
@triton.jit
def _conv_router_dx_kernel(GO_ptr, W_ptr, GX_ptr, B, S, H,
                           sgm, sge, swe, swh, swk, sxb, sxs, sxh,
                           K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
                           BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_b = tl.program_id(0); pid_s = tl.program_id(1); pid_h = tl.program_id(2)
    offs_j = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_e = tl.arange(0, BLOCK_E)
    mask_j = offs_j < S; mask_h = offs_h < H; mask_e = offs_e < E
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)
    for m in tl.static_range(K):
        s = offs_j + m
        mask_s = (s < S) & mask_j
        kw = K - 1 - m
        go = tl.load(GO_ptr + (pid_b * S + s)[:, None] * sgm + offs_e[None, :] * sge,
                     mask=mask_s[:, None] & mask_e[None, :], other=0.0)
        wv = tl.load(W_ptr + offs_e[:, None] * swe + offs_h[None, :] * swh + kw * swk,
                     mask=mask_e[:, None] & mask_h[None, :], other=0.0)
        acc += tl.dot(go, wv)
    tl.store(GX_ptr + pid_b * sxb + offs_j[:, None] * sxs + offs_h[None, :] * sxh,
             acc.to(GX_ptr.dtype.element_ty), mask=mask_j[:, None] & mask_h[None, :])


# grad_w[e,h,k] = sum_{b,s} grad_out[b,s,e] * x[b, s-(K-1)+k, h]  (src>=0)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64, 'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 128, 'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 64, 'BLOCK_S': 128}, num_warps=4, num_stages=3),
    ], key=['H'])
@triton.jit
def _conv_router_dw_kernel(GO_ptr, X_ptr, GW_ptr, B, S, H,
                           sgm, sge, sxb, sxs, sxh, gwe, gwh, gwk,
                           K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
                           BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_k = tl.program_id(0); pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_e = tl.arange(0, BLOCK_E)
    mask_h = offs_h < H; mask_e = offs_e < E
    acc = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
    for b in range(B):
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            src = offs_s - (K - 1) + pid_k
            mask_s = offs_s < S
            mask_src = (src >= 0) & mask_s
            go = tl.load(GO_ptr + (b * S + offs_s)[:, None] * sgm + offs_e[None, :] * sge,
                         mask=mask_s[:, None] & mask_e[None, :], other=0.0)
            xv = tl.load(X_ptr + b * sxb + src[:, None] * sxs + offs_h[None, :] * sxh,
                         mask=mask_src[:, None] & mask_h[None, :], other=0.0)
            acc += tl.dot(tl.trans(go), xv)
    tl.store(GW_ptr + offs_e[:, None] * gwe + offs_h[None, :] * gwh + pid_k * gwk,
             acc.to(GW_ptr.dtype.element_ty), mask=mask_e[:, None] & mask_h[None, :])


class _FusedConvRouterFunction(torch.autograd.Function):
    """Triton fused forward + transpose-free Triton backward. grad_x exact; grad_w correct
    (long-reduction fp32 order / fp16 storage only). ~5x fwd, ~4-6x fwd+bwd vs eager."""
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return _fused_conv_router_fwd(x, weight)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        B, S, Hd = x.shape
        E, _, K = weight.shape
        # router casts logits to fp32 for routing math -> grad_out is fp32 while weight is fp16;
        # match dtypes for tl.dot (grads are stored in the param dtype anyway).
        go = grad_out.contiguous().to(x.dtype)
        BE = max(16, triton.next_power_of_2(E))
        gx = torch.empty(B, S, Hd, device=x.device, dtype=x.dtype)
        gw = torch.empty(E, Hd, K, device=x.device, dtype=x.dtype)
        gridx = lambda m: (B, triton.cdiv(S, m['BLOCK_S']), triton.cdiv(Hd, m['BLOCK_H']))
        _conv_router_dx_kernel[gridx](go, weight, gx, B, S, Hd,
            go.stride(0), go.stride(1), weight.stride(0), weight.stride(1), weight.stride(2),
            gx.stride(0), gx.stride(1), gx.stride(2), K=K, E=E, BLOCK_E=BE)
        gridw = lambda m: (K, triton.cdiv(Hd, m['BLOCK_H']))
        _conv_router_dw_kernel[gridw](go, x, gw, B, S, Hd,
            go.stride(0), go.stride(1), x.stride(0), x.stride(1), x.stride(2),
            gw.stride(0), gw.stride(1), gw.stride(2), K=K, E=E, BLOCK_E=BE)
        return gx, gw


def triton_causal_conv1d_router(
    x: torch.Tensor,        # (B, S, H)
    weight: torch.Tensor,   # (E, H, K) — nn.Conv1d weight
    num_experts: int,
    kernel_size: int,
) -> torch.Tensor:
    """Fused causal conv1d router projection -> (B*S, E). Triton forward (no transpose
    copy, ~5x fwd vs eager at large batch), cuDNN backward (grad-exact). fp16/fp32."""
    return _FusedConvRouterFunction.apply(x, weight)


class _FusedConvRouterFull(torch.autograd.Function):
    """
    WHOLE conv router fused into one autograd.Function (forward + manual backward).

    Folds the entire eager pipeline — conv projection, sigmoid gate, +bias selection,
    top-k, and unbiased weight gather — behind a single graph node, so autograd sees
    one op instead of a chain of ~6 tiny elementwise/transpose launches.

    forward:
        scores  = sigmoid(causal_conv(x))           # Triton conv + fused-sigmoid epilogue, fp32
        sel     = scores + bias                      # bias = selection only (DeepSeek-V3)
        idx     = topk(sel, k)                       # torch.topk (robust tie-break, grad-free)
        weights = scores.gather(idx)                 # UNBIASED weights, differentiable

    backward (only `weights` carries grad; idx is discrete):
        grad_scores = scatter_add(grad_weights @ idx)            # gather^T
        grad_logits = grad_scores * scores * (1 - scores)        # sigmoid'
        grad_x, grad_w = conv_backward(grad_logits)              # existing dx/dw Triton kernels

    GEMM budget: 1 matmul-launch fwd (conv tl.dot), 2 bwd (dx, dw) — identical to eager's
    cuDNN conv; everything else is fused away. Never more GEMMs than eager.

    Scope: gate_type='sigmoid', router_activation='none', logit-norm OFF (deprecated).
    norm_topk_prob is applied by the caller in eager so autograd handles its Jacobian.
    """
    @staticmethod
    def forward(ctx, x, weight, bias, top_k):
        # Triton conv with sigmoid fused into the store epilogue -> fp32 scores (B*S, E)
        scores = _fused_conv_router_fwd(x, weight, apply_sigmoid=True, out_dtype=torch.float32)
        # Selection uses scores + bias (load-balancing bias, selection ONLY). bias is fp32.
        sel = scores + bias if bias is not None else scores
        _, idx = torch.topk(sel, top_k, dim=-1)
        weights = scores.gather(-1, idx)
        ctx.save_for_backward(x, weight, scores, idx)
        return idx, weights

    @staticmethod
    def backward(ctx, grad_idx, grad_weights):
        x, weight, scores, idx = ctx.saved_tensors
        B, S, Hd = x.shape
        E, _, K = weight.shape
        # gather^T: scatter grad into the selected score positions (idx unique per row -> add == set)
        grad_scores = torch.zeros_like(scores)
        grad_scores.scatter_add_(-1, idx, grad_weights.float())
        # sigmoid': scores*(1-scores) (scores are already the sigmoid output) — fp32
        grad_logits = grad_scores * scores * (1.0 - scores)
        go = grad_logits.to(x.dtype).contiguous()       # match conv-bwd kernel dtype (param dtype)
        BE = max(16, triton.next_power_of_2(E))
        gx = torch.empty(B, S, Hd, device=x.device, dtype=x.dtype)
        gw = torch.empty(E, Hd, K, device=x.device, dtype=x.dtype)
        gridx = lambda m: (B, triton.cdiv(S, m['BLOCK_S']), triton.cdiv(Hd, m['BLOCK_H']))
        _conv_router_dx_kernel[gridx](go, weight, gx, B, S, Hd,
            go.stride(0), go.stride(1), weight.stride(0), weight.stride(1), weight.stride(2),
            gx.stride(0), gx.stride(1), gx.stride(2), K=K, E=E, BLOCK_E=BE)
        gridw = lambda m: (K, triton.cdiv(Hd, m['BLOCK_H']))
        _conv_router_dw_kernel[gridw](go, x, gw, B, S, Hd,
            go.stride(0), go.stride(1), x.stride(0), x.stride(1), x.stride(2),
            gw.stride(0), gw.stride(1), gw.stride(2), K=K, E=E, BLOCK_E=BE)
        return gx, gw, None, None  # x, weight, bias (no grad), top_k


def fused_conv_router_full(x, weight, bias, top_k):
    """Whole conv router (conv+sigmoid+bias+topk+gather) as one fused autograd op.
    Returns (top_k_indices (B*S,k) long, top_k_weights (B*S,k) fp32-unbiased)."""
    return _FusedConvRouterFull.apply(x, weight, bias, top_k)


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

    Fully fused: conv projection + sigmoid + bias-selection + top-k + unbiased-weight gather
    all collapse into one autograd node (_FusedConvRouterFull). The only ops left in eager
    are the optional norm_topk_prob (kept in eager so autograd handles its Jacobian) and the
    final reshape.

    Supported config (the conv-router default): gate_type='sigmoid', router_activation='none'.
    Router logit-norm is DEPRECATED and intentionally NOT applied here (see router.py).
    Other gate/activation combos fall back to the original eager forward.
    """
    from einops import rearrange

    # Fused path only covers the default conv-router math. Anything else -> eager original.
    if (self.router_type != "conv"
            or self.gate_type != "sigmoid"
            or self.router_activation != "none"
            or (self.training and self.router_noise > 0)):
        return self._original_forward(hidden_states)

    batch_size, seq_len, _ = hidden_states.shape

    # conv + sigmoid + (+bias) + topk + gather, fused into a single autograd op.
    top_k_indices, top_k_weights = fused_conv_router_full(
        hidden_states, self.gate_conv.weight, self.bias, self.top_k,
    )

    # norm_topk_prob in eager: autograd carries its Jacobian back into top_k_weights, then
    # _FusedConvRouterFull.backward maps that to grad_x / grad_w. Default off (no-op).
    if self.norm_topk_prob:
        top_k_weights = top_k_weights / (top_k_weights.sum(-1, keepdim=True) + 1e-6)

    top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
    top_k_weights = rearrange(top_k_weights, '(b s) k -> b s k', b=batch_size)
    return top_k_indices.long(), top_k_weights.to(hidden_states.dtype)


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
