"""
Fused MoE Dispatch — Optimized expert computation for BiBo.

Strategy: cuBLAS for GEMMs (already optimal) + Triton for fusion.

Key optimizations over baseline:
1. Fused activation + GLU multiply + down_proj weight application (1 kernel)
2. Fused router: sigmoid + logit_norm + bias (1 kernel vs 3 ops)
3. Batched expert processing: pre-sort once, process all experts
4. Eliminated intermediate tensor materialization where possible

The sequential expert loop remains (optimal for ≤16 experts on single GPU),
but each expert's compute is fused tighter.

Supports: RTX 3050 (sm_86), T4 (sm_75), A100 (sm_80).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from typing import List, Tuple, Optional

__all__ = ['patch_moe_with_triton', 'unpatch_moe']


# ═══════════════════════════════════════════════════════════════
# KERNEL 1: Fused GLU Activation + Weight Application
# 
# Replaces: gate, up = chunk(gate_up, 2) → act(gate) → gate*up → *weight
# This is the hot inner loop — fusing it eliminates 3 intermediate tensors.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_glu_act_kernel(
    GateUp_ptr,     # (M, 2*I) — fused gate+up output from linear
    Out_ptr,        # (M, I) — activated GLU output
    M, I,           # M=num_tokens, I=intermediate_size
    stride_gu_m, stride_gu_i,
    stride_o_m, stride_o_i,
    ACT_TYPE: tl.constexpr,  # 0=silu, 1=relu2, 2=tanh
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused: split gate_up → activate gate → multiply with up.
    Input: (M, 2I) concatenated gate+up from F.linear
    Output: (M, I) = act(gate) * up
    
    Saves 2 intermediate tensors (gate, up) and 1 activation output.
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
    
    # Apply activation to gate
    if ACT_TYPE == 0:  # SiLU: x * sigmoid(x)
        act_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    elif ACT_TYPE == 1:  # ReLU²: relu(x)²
        relu_g = tl.where(gate > 0, gate, tl.zeros_like(gate))
        act_gate = relu_g * relu_g
    else:  # Tanh: (exp(2x) - 1) / (exp(2x) + 1)
        exp2x = tl.exp(2.0 * gate)
        act_gate = (exp2x - 1.0) / (exp2x + 1.0)
    
    # GLU multiply
    result = act_gate * up
    
    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_o_m + offs_i[None, :] * stride_o_i
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════
# KERNEL 2: Fused Down-Proj + Weight + Scatter-Add
#
# Replaces: F.linear(intermediate, down) → *weight → index_add_
# Fuses the weight multiplication into the output write.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_down_weight_kernel(
    X_ptr,          # (M, I) — intermediate (after GLU activation)
    Down_ptr,       # (H, I) — down projection weight
    W_ptr,          # (M,) — routing weights
    Out_ptr,        # (M, H) — output buffer
    M, H, I,
    stride_xm, stride_xi,
    stride_dh, stride_di,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused: out[m, h] = weight[m] * sum_i(X[m, i] * Down[h, i])
    
    This is a GEMM with per-row scaling. For small M (typical in MoE
    where tokens are split across experts), this can be faster than
    separate matmul + broadcast multiply.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < M
    mask_h = offs_h < H
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    
    # Reduce over intermediate dim
    for i_start in range(0, I, BLOCK_I):
        offs_i = i_start + tl.arange(0, BLOCK_I)
        mask_i = offs_i < I
        
        # Load X tile: (BLOCK_M, BLOCK_I)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_i[None, :] * stride_xi
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_i[None, :], other=0.0)
        
        # Load Down tile: (BLOCK_H, BLOCK_I)
        d_ptrs = Down_ptr + offs_h[:, None] * stride_dh + offs_i[None, :] * stride_di
        d_tile = tl.load(d_ptrs, mask=mask_h[:, None] & mask_i[None, :], other=0.0)
        
        # acc[m, h] += x[m, i] * d[h, i] → matmul
        acc += tl.dot(x_tile, tl.trans(d_tile))
    
    # Apply routing weight
    w = tl.load(W_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    acc = acc * w[:, None]
    
    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_h[None, :])


# ═══════════════════════════════════════════════════════════════
# KERNEL 6: Fused Linear + GLU Activation
#
# Fuses: gate_up GEMM (X @ W.T) + GLU activation in one kernel.
# Eliminates the (M, 2*I) intermediate HBM write+read.
# Input: X (M, K), W_gate_up (2*I, K)
# Output: (M, I) = act(gate) * up
#
# No autotuning — M varies per expert, causes recompilation.
# Uses fixed block sizes with triton.next_power_of_2.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_linear_glu_kernel(
    X_ptr,          # (M, K) — input
    W_ptr,          # (2*I, K) — fused gate+up weight
    Out_ptr,        # (M, I) — output
    GateUp_ptr,     # (M, 2*I) — saved for backward (can be None if not training)
    M, K, I,
    stride_xm, stride_xk,
    stride_wi, stride_wk,
    stride_om, stride_oi,
    save_gate_up: tl.constexpr,  # whether to save gate_up for backward
    ACT_TYPE: tl.constexpr,  # 0=silu, 1=relu2, 2=tanh
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused: gate_up = X @ W.T; intermediate = act(gate) * up
    Eliminates the (M, 2*I) intermediate HBM write+read.
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M
    mask_i = offs_i < I

    # Accumulators for gate and up
    acc_gate = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    # Reduce over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load X tile: (BLOCK_M, BLOCK_K)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load gate weight tile: (BLOCK_I, BLOCK_K) — first I rows of W
        w_gate_ptrs = W_ptr + offs_i[:, None] * stride_wi + offs_k[None, :] * stride_wk
        w_gate_tile = tl.load(w_gate_ptrs, mask=mask_i[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load up weight tile: (BLOCK_I, BLOCK_K) — second I rows of W
        w_up_ptrs = W_ptr + (I + offs_i)[:, None] * stride_wi + offs_k[None, :] * stride_wk
        w_up_tile = tl.load(w_up_ptrs, mask=mask_i[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc[m, i] += x[m, k] * w[i, k]
        acc_gate += tl.dot(x_tile, tl.trans(w_gate_tile))
        acc_up += tl.dot(x_tile, tl.trans(w_up_tile))

    # Apply GLU activation in registers
    if ACT_TYPE == 0:  # SiLU: gate * sigmoid(gate)
        act_gate = acc_gate * (1.0 / (1.0 + tl.exp(-acc_gate)))
    elif ACT_TYPE == 1:  # ReLU²: max(gate, 0)²
        act_gate = tl.maximum(acc_gate, 0.0) ** 2
    else:  # Tanh: 2*sigmoid(2*gate) - 1 (fast approximation)
        act_gate = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * acc_gate))) - 1.0

    result = act_gate * acc_up

    # Store output
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_i[None, :] * stride_oi
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_i[None, :])

    # Save gate_up for backward if needed
    if save_gate_up:
        # Store gate (first I columns)
        gate_ptrs = GateUp_ptr + offs_m[:, None] * (2 * I) + offs_i[None, :]
        tl.store(gate_ptrs, acc_gate.to(GateUp_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_i[None, :])
        # Store up (second I columns)
        up_ptrs = GateUp_ptr + offs_m[:, None] * (2 * I) + (I + offs_i)[None, :]
        tl.store(up_ptrs, acc_up.to(GateUp_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_i[None, :])


class _FusedLinearGLUFunction(torch.autograd.Function):
    """Fused Linear + GLU with autograd support.
    
    Forward: computes gate_up = X @ W.T AND applies GLU activation.
    Backward: recomputes gate_up from x and weight, then applies GLU backward + GEMM backward.
    """
    
    @staticmethod
    def forward(ctx, x, weight, act_type):
        M, K = x.shape
        I = weight.shape[0] // 2
        
        ctx.save_for_backward(x, weight)
        ctx.act_type = act_type
        ctx.M, ctx.K, ctx.I = M, K, I
        
        out = torch.empty(M, I, device=x.device, dtype=x.dtype)
        
        BLOCK_M = min(64, triton.next_power_of_2(M))
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_I = min(128, triton.next_power_of_2(I))
        BLOCK_I = max(16, BLOCK_I)
        BLOCK_K = min(64, triton.next_power_of_2(K))
        BLOCK_K = max(16, BLOCK_K)
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
        
        # Use save_gate_up=False — backward will recompute
        _fused_linear_glu_kernel[grid](
            x, weight, out, out,  # dummy gate_up pointer (not used when save_gate_up=False)
            M, K, I,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1),
            save_gate_up=False,
            ACT_TYPE=act_type,
            BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I, BLOCK_K=BLOCK_K,
        )
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        act_type = ctx.act_type
        M, K, I = ctx.M, ctx.K, ctx.I
        
        # Recompute gate_up from x and weight (avoids saving large tensor)
        gate_up = F.linear(x.float(), weight.float())  # (M, 2*I) in float32
        
        # Allocate output gradient
        grad_gate_up = torch.empty(M, 2 * I, device=x.device, dtype=x.dtype)
        
        BLOCK_M = min(64, triton.next_power_of_2(M))
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_I = min(128, triton.next_power_of_2(I))
        BLOCK_I = max(16, BLOCK_I)
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
        
        # Launch Triton backward kernel for GLU derivatives
        _fused_glu_act_backward_kernel[grid](
            grad_output, gate_up.to(x.dtype), grad_gate_up,
            M, I,
            grad_output.stride(0), grad_output.stride(1),
            gate_up.stride(0), gate_up.stride(1),
            grad_gate_up.stride(0), grad_gate_up.stride(1),
            ACT_TYPE=act_type,
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
        )
        
        # GEMM backward in float32 (cuBLAS is optimal for large shapes)
        grad_x = torch.mm(grad_gate_up.float(), weight.float())
        grad_weight = torch.mm(grad_gate_up.float().t(), x.float())
        
        return grad_x, grad_weight, None


def triton_fused_linear_glu(
    x: torch.Tensor,      # (M, K) — input
    weight: torch.Tensor,  # (2*I, K) — fused gate+up weight
    act_type: int,         # 0=silu, 1=relu2, 2=tanh
) -> torch.Tensor:
    """
    Fused Linear + GLU activation.
    Computes: gate_up = x @ weight.T; out = act(gate) * up
    Returns (M, I).
    """
    return _FusedLinearGLUFunction.apply(x, weight, act_type)

# (KERNEL 3 "Fused Router" REMOVED Jun 28 2026 — it carried the Skywork z-score logit-norm, which
# is not part of MiMo/DeepSeek-V3 routing. The eager router is the single source of truth; sigmoid +
# bias is a trivial 2-op elementwise that the compiler fuses for free, so no kernel is warranted.)


# ═══════════════════════════════════════════════════════════════
# KERNEL 4: Batched GLU Activation (all experts, 1 launch)
#
# Instead of launching _fused_glu_act_kernel once per expert (8 launches),
# concatenate all gate_ups and run ONE kernel with per-row act_type dispatch.
# Reduces 8 Triton launches → 1 launch.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _batched_glu_act_kernel(
    GateUp_ptr,      # (N_total, 2*I) — concatenated gate_ups from all experts
    ActType_ptr,     # (N_total,) — int32 act type per row (0=silu, 1=relu2, 2=tanh)
    Out_ptr,         # (N_total, I) — activated output
    N_total, I,
    stride_gu_m, stride_gu_i,
    stride_o_m, stride_o_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused: per-row activation dispatch + GLU multiply.
    Each row can have a different activation type (SiLU/ReLU²/Tanh).
    Reads act_type from a per-row array instead of constexpr.
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < N_total
    mask_i = offs_i < I
    mask = mask_m[:, None] & mask_i[None, :]

    # Load gate (first I columns)
    gate_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + offs_i[None, :] * stride_gu_i
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load up (second I columns)
    up_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + (I + offs_i)[None, :] * stride_gu_i
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load per-row activation type
    act_type = tl.load(ActType_ptr + offs_m, mask=mask_m, other=0)

    # Apply activation (branch on runtime act_type)
    # SiLU: x * sigmoid(x)
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    # ReLU²: relu(x)²
    relu_g = tl.where(gate > 0, gate, tl.zeros_like(gate))
    relu2_gate = relu_g * relu_g
    # Tanh: tanh(x)
    exp2x = tl.exp(2.0 * gate)
    tanh_gate = (exp2x - 1.0) / (exp2x + 1.0)

    # Select activation based on per-row act_type
    act_gate = tl.where(act_type[:, None] == 0, silu_gate,
               tl.where(act_type[:, None] == 1, relu2_gate, tanh_gate))

    # GLU multiply
    result = act_gate * up

    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_o_m + offs_i[None, :] * stride_o_i
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=mask)


def triton_batched_glu_activation(
    all_gate_ups: torch.Tensor,   # (N_total, 2*I) — concatenated
    all_act_types: torch.Tensor,  # (N_total,) — int32, per-row act type
    I: int,
) -> torch.Tensor:
    """
    Batched GLU activation for all experts in a single kernel launch.
    Replaces N expert calls to _fused_glu_act_kernel with 1 call.
    """
    N_total = all_gate_ups.shape[0]
    out = torch.empty(N_total, I, device=all_gate_ups.device, dtype=all_gate_ups.dtype)

    BLOCK_M = min(64, triton.next_power_of_2(N_total))
    BLOCK_M = max(16, BLOCK_M)
    BLOCK_I = min(128, triton.next_power_of_2(I))
    BLOCK_I = max(16, BLOCK_I)

    grid = (triton.cdiv(N_total, BLOCK_M), triton.cdiv(I, BLOCK_I))

    _batched_glu_act_kernel[grid](
        all_gate_ups, all_act_types, out,
        N_total, I,
        all_gate_ups.stride(0), all_gate_ups.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_I=BLOCK_I,
    )
    return out


# ═══════════════════════════════════════════════════════════════
# KERNEL 5: Fused Weight-Multiply + Scatter-Add
#
# Replaces: expert_out * weights.unsqueeze(-1) → index_add_
# Fuses the weight scaling into the scatter-add write.
# Eliminates 1 intermediate (n, H) tensor per expert per forward.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_weight_scatter_kernel(
    ExpertOut_ptr,      # (N, H) — expert output
    Weights_ptr,        # (N,) — per-token routing weights
    SortedIdx_ptr,      # (N,) — sorted token indices for scatter
    Out_ptr,            # (M, H) — output buffer (accumulated)
    N, H, M,
    stride_eo_n, stride_eo_h,
    stride_o_m, stride_o_h,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fused: out[sorted_idx[n], h] += expert_out[n, h] * weights[n]
    Eliminates the intermediate tensor from weight scaling.
    Uses atomic_add for scatter (safe for MoE where expert chunks don't overlap).
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_n = offs_n < N
    mask_h = offs_h < H
    mask = mask_n[:, None] & mask_h[None, :]

    # Load expert output
    eo_ptrs = ExpertOut_ptr + offs_n[:, None] * stride_eo_n + offs_h[None, :] * stride_eo_h
    expert_out = tl.load(eo_ptrs, mask=mask, other=0.0)

    # Load weights and scale
    w = tl.load(Weights_ptr + offs_n, mask=mask_n, other=0.0)
    scaled = expert_out * w[:, None]

    # Load scatter indices
    idx = tl.load(SortedIdx_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)

    # Scatter-add: out[idx[n], h] += scaled[n, h]
    out_ptrs = Out_ptr + idx[:, None] * stride_o_m + offs_h[None, :] * stride_o_h
    tl.atomic_add(out_ptrs, scaled, mask=mask)


# ═══════════════════════════════════════════════════════════════
# Autograd wrapper for fused weight scatter
#
# Forward: atomic_add (fast, fused)
# Backward: standard index ops (differentiable)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# KERNEL 8: Weight Scatter Backward
#
# Backward of fused weight multiply + scatter-add.
# Forward: out[idx[n]] += expert_out[n] * weights[n]
# Backward: grad_expert_out[n] = grad_out[idx[n]] * weights[n]
#           grad_weights[n] = (grad_out[idx[n]] * expert_out[n]).sum()
#
# Uses tl.load with gathered indices (differentiable pattern).
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_weight_scatter_backward_kernel(
    GradOut_ptr,      # (M, H) — gradient of output
    ExpertOut_ptr,    # (N, H) — saved expert output
    Weights_ptr,      # (N,) — saved routing weights
    SortedIdx_ptr,    # (N,) — sorted token indices
    GradExpertOut_ptr,# (N, H) — gradient of expert output
    GradWeights_ptr,  # (N,) — gradient of weights
    N, H, M,
    stride_go_m, stride_go_h,
    stride_eo_n, stride_eo_h,
    stride_ge_n, stride_ge_h,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backward: gather grad_output at sorted positions, scale by weights."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_n = offs_n < N
    mask_h = offs_h < H
    mask = mask_n[:, None] & mask_h[None, :]

    # Load sorted indices
    idx = tl.load(SortedIdx_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)

    # Load grad_output at sorted positions: grad_output[idx[n], h]
    go_ptrs = GradOut_ptr + idx[:, None] * stride_go_m + offs_h[None, :] * stride_go_h
    grad_out = tl.load(go_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load weights
    w = tl.load(Weights_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # Load expert_out
    eo_ptrs = ExpertOut_ptr + offs_n[:, None] * stride_eo_n + offs_h[None, :] * stride_eo_h
    expert_out = tl.load(eo_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute grad_expert_out = grad_output[idx] * weights
    grad_expert_out = grad_out * w[:, None]

    # Store grad_expert_out
    geo_ptrs = GradExpertOut_ptr + offs_n[:, None] * stride_ge_n + offs_h[None, :] * stride_ge_h
    tl.store(geo_ptrs, grad_expert_out, mask=mask)

    # Compute grad_weights = (grad_output[idx] * expert_out).sum(dim=-1)
    # This is a reduction across H — each program handles a chunk of N
    grad_w_partial = tl.sum(grad_out * expert_out, axis=1)  # (BLOCK_N,)
    
    # Atomic add to accumulate across H blocks
    tl.atomic_add(GradWeights_ptr + offs_n, grad_w_partial, mask=mask_n)


class _FusedWeightScatterFunction(torch.autograd.Function):
    """Fused weight multiply + scatter-add with autograd support.
    
    Forward: fuses expert_out * weights → scatter-add via atomic_add
    Backward: standard index_select for gradient flow
    """
    
    @staticmethod
    def forward(ctx, expert_out, weights, sorted_indices, M):
        """Forward: fused weight multiply + scatter-add."""
        ctx.save_for_backward(expert_out, weights, sorted_indices)
        ctx.M = M
        
        N, H = expert_out.shape
        output = torch.zeros(M, H, device=expert_out.device, dtype=expert_out.dtype)
        
        BLOCK_N = min(64, triton.next_power_of_2(N))
        BLOCK_N = max(16, BLOCK_N)
        BLOCK_H = min(128, triton.next_power_of_2(H))
        BLOCK_H = max(16, BLOCK_H)
        
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))
        
        _fused_weight_scatter_kernel[grid](
            expert_out, weights, sorted_indices, output,
            N, H, M,
            expert_out.stride(0), expert_out.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
        )
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward: Triton gather for gradient flow."""
        expert_out, weights, sorted_indices = ctx.saved_tensors
        M = ctx.M
        N, H = expert_out.shape
        
        # Allocate output gradients
        grad_expert_out = torch.empty(N, H, device=expert_out.device, dtype=expert_out.dtype)
        grad_weights = torch.zeros(N, device=weights.device, dtype=weights.dtype)
        
        BLOCK_N = min(64, triton.next_power_of_2(N))
        BLOCK_N = max(16, BLOCK_N)
        BLOCK_H = min(128, triton.next_power_of_2(H))
        BLOCK_H = max(16, BLOCK_H)
        
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))
        
        # Launch Triton backward kernel
        _fused_weight_scatter_backward_kernel[grid](
            grad_output, expert_out, weights, sorted_indices,
            grad_expert_out, grad_weights,
            N, H, M,
            grad_output.stride(0), grad_output.stride(1),
            expert_out.stride(0), expert_out.stride(1),
            grad_expert_out.stride(0), grad_expert_out.stride(1),
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
        )
        
        return grad_expert_out, grad_weights, None, None


def triton_fused_weight_scatter_autograd(
    expert_out: torch.Tensor,      # (N, H) — expert output
    weights: torch.Tensor,         # (N,) — routing weights
    sorted_indices: torch.Tensor,  # (N,) — scatter indices
    M: int,                        # total number of tokens
) -> torch.Tensor:
    """
    Differentiable fused weight multiply + scatter-add.
    Uses autograd.Function for correct gradient flow.
    Returns (M, H) accumulated output.
    """
    return _FusedWeightScatterFunction.apply(expert_out, weights, sorted_indices, M)


# ═══════════════════════════════════════════════════════════════
# Python Wrappers
# ═══════════════════════════════════════════════════════════════

_ACT_MAP = {"silu": 0, "relu2": 1, "tanh": 2}


# ═══════════════════════════════════════════════════════════════
# KERNEL 7: GLU Activation Backward
#
# Computes grad_gate_up from grad_output and saved gate_up.
# grad_gate = grad_output * up * d_act(gate)
# grad_up = grad_output * act(gate)
# grad_gate_up = [grad_gate, grad_up]
#
# All computed in float32 to prevent overflow in derivatives.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_glu_act_backward_kernel(
    GradOut_ptr,      # (M, I) — gradient of output
    GateUp_ptr,       # (M, 2*I) — saved gate_up
    GradGateUp_ptr,   # (M, 2*I) — output gradient
    M, I,
    stride_go_m, stride_go_i,
    stride_gu_m, stride_gu_i,
    stride_ggu_m, stride_ggu_i,
    ACT_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Backward: compute grad_gate_up from grad_output and gate_up."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M
    mask_i = offs_i < I
    mask = mask_m[:, None] & mask_i[None, :]

    # Load grad_output
    go_ptrs = GradOut_ptr + offs_m[:, None] * stride_go_m + offs_i[None, :] * stride_go_i
    grad_out = tl.load(go_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load gate (first I columns of gate_up)
    gate_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + offs_i[None, :] * stride_gu_i
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load up (second I columns of gate_up)
    up_ptrs = GateUp_ptr + offs_m[:, None] * stride_gu_m + (I + offs_i)[None, :] * stride_gu_i
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute activation and its derivative in float32
    if ACT_TYPE == 0:  # SiLU
        sig_gate = 1.0 / (1.0 + tl.exp(-gate))
        act_gate = gate * sig_gate
        dact = sig_gate * (1.0 + gate * (1.0 - sig_gate))
    elif ACT_TYPE == 1:  # ReLU²
        relu_gate = tl.maximum(gate, 0.0)
        act_gate = relu_gate * relu_gate
        dact = 2.0 * relu_gate
    else:  # Tanh
        act_gate = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * gate))) - 1.0
        dact = 1.0 - act_gate * act_gate

    # Compute gradients
    grad_up = grad_out * act_gate
    grad_gate = grad_out * up * dact

    # Store grad_gate (first I columns)
    gg_ptrs = GradGateUp_ptr + offs_m[:, None] * stride_ggu_m + offs_i[None, :] * stride_ggu_i
    tl.store(gg_ptrs, grad_gate, mask=mask)

    # Store grad_up (second I columns)
    gu_ptrs = GradGateUp_ptr + offs_m[:, None] * stride_ggu_m + (I + offs_i)[None, :] * stride_ggu_i
    tl.store(gu_ptrs, grad_up, mask=mask)


class _TritonMoEGLUFunction(torch.autograd.Function):
    """Autograd wrapper for MoE GLU activation.
    
    Forward: Triton fused GLU (split → activate gate → multiply with up)
    Backward: PyTorch recompute (correct gradient flow to gate_up_proj)
    """
    
    @staticmethod
    def forward(ctx, gate_up, act_type):
        ctx.save_for_backward(gate_up)
        ctx.act_type = act_type
        
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
        
        BLOCK_M = min(64, triton.next_power_of_2(M))
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_I = min(128, triton.next_power_of_2(I))
        BLOCK_I = max(16, BLOCK_I)
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
        
        _fused_glu_act_kernel[grid](
            gate_up, out,
            M, I,
            gate_up.stride(0), gate_up.stride(1),
            out.stride(0), out.stride(1),
            ACT_TYPE=act_type,
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
        )
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        gate_up, = ctx.saved_tensors
        act_type = ctx.act_type
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        
        # Allocate output gradient
        grad_gate_up = torch.empty(M, 2 * I, device=gate_up.device, dtype=gate_up.dtype)
        
        BLOCK_M = min(64, triton.next_power_of_2(M))
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_I = min(128, triton.next_power_of_2(I))
        BLOCK_I = max(16, BLOCK_I)
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
        
        # Launch Triton backward kernel
        _fused_glu_act_backward_kernel[grid](
            grad_output, gate_up, grad_gate_up,
            M, I,
            grad_output.stride(0), grad_output.stride(1),
            gate_up.stride(0), gate_up.stride(1),
            grad_gate_up.stride(0), grad_gate_up.stride(1),
            ACT_TYPE=act_type,
            BLOCK_M=BLOCK_M,
            BLOCK_I=BLOCK_I,
        )
        return grad_gate_up, None


def triton_fused_glu_activation(
    gate_up: torch.Tensor,  # (M, 2*I)
    act_type: int,          # 0=silu, 1=relu2, 2=tanh
) -> torch.Tensor:
    """
    Differentiable fused GLU activation: split → activate gate → multiply with up.
    Uses autograd.Function for correct gradient flow to gate_up_proj.
    Returns (M, I).
    """
    return _TritonMoEGLUFunction.apply(gate_up, act_type)
    return out


def triton_fused_down_weight(
    intermediate: torch.Tensor,  # (M, I)
    down_w: torch.Tensor,        # (H, I)
    weights: torch.Tensor,       # (M,)
) -> torch.Tensor:
    """
    Fused down projection + weight: out = weight * (intermediate @ down^T)
    Returns (M, H).
    """
    M, I = intermediate.shape
    H = down_w.shape[0]
    
    out = torch.empty(M, H, device=intermediate.device, dtype=intermediate.dtype)
    
    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_M = max(16, BLOCK_M)
    BLOCK_H = min(64, triton.next_power_of_2(H))
    BLOCK_H = max(16, BLOCK_H)
    BLOCK_I = min(64, triton.next_power_of_2(I))
    BLOCK_I = max(16, BLOCK_I)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_H))
    
    _fused_down_weight_kernel[grid](
        intermediate, down_w, weights, out,
        M, H, I,
        intermediate.stride(0), intermediate.stride(1),
        down_w.stride(0), down_w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        BLOCK_I=BLOCK_I,
    )
    return out


# ═══════════════════════════════════════════════════════════════
# Optimized MoE Expert Forward
# ═══════════════════════════════════════════════════════════════

def triton_moe_experts_forward(
    hidden_states: torch.Tensor,    # (num_tokens, hidden_size)
    top_k_indices: torch.Tensor,    # (num_tokens, top_k)
    top_k_weights: torch.Tensor,    # (num_tokens, top_k)
    gate_up_proj: torch.Tensor,     # (num_polyglu, 2*inter, hidden)
    down_proj: torch.Tensor,        # (num_polyglu, hidden, inter)
    expert_activations: List[str],
    num_polyglu_experts: int,
    identity_start: int,
    zero_start: int,
    num_routed_experts: int,
) -> torch.Tensor:
    """
    MoE dispatch with per-expert Triton GLU activation.

    Per-expert approach (faster for small models):
    - cuBLAS GEMM → Triton GLU → cuBLAS GEMM → scatter per expert
    - Skips empty experts (no wasted compute)
    - No concat/split overhead

    Batched GLU kernel available via triton_batched_glu_activation()
    for cases where launch overhead dominates (large models).
    """
    num_tokens, hidden_size = hidden_states.shape

    # Sort tokens by expert
    flat_expert_indices = top_k_indices.flatten()
    flat_token_indices = torch.arange(
        num_tokens, device=hidden_states.device
    ).unsqueeze(1).expand_as(top_k_indices).flatten()
    flat_weights = top_k_weights.flatten()

    sorted_expert_indices, sort_order = flat_expert_indices.sort()
    sorted_token_indices = flat_token_indices[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Expert boundaries — pulled to CPU ints with ONE sync, then the loop below
    # uses Python ints. Comparing/slicing with CUDA *scalar* tensors (the old
    # `start = boundaries[i]; if end - start == 0`) forces an implicit GPU sync
    # EVERY iteration, serializing the whole dispatch. That alone made this path
    # ~1.6x slower and lose to PyTorch eager in fp16 (measured, RTX 3050).
    expert_counts = torch.bincount(sorted_expert_indices, minlength=num_routed_experts).tolist()
    boundaries = [0]
    for _c in expert_counts:
        boundaries.append(boundaries[-1] + _c)

    # Output buffer
    output = torch.zeros(num_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

    ACT_MAP = {"silu": 0, "relu2": 1, "tanh": 2}

    for expert_idx in range(num_routed_experts):
        start = boundaries[expert_idx]
        end = boundaries[expert_idx + 1]

        if end == start:
            continue

        token_idx = sorted_token_indices[start:end]
        weights = sorted_weights[start:end]
        current_state = hidden_states[token_idx]

        if expert_idx < num_polyglu_experts:
            act_name = expert_activations[expert_idx]
            act_type = ACT_MAP[act_name]

            # cuBLAS GEMM for gate_up
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])

            # Triton GLU activation
            intermediate = triton_fused_glu_activation(gate_up, act_type)

            # cuBLAS GEMM for down
            expert_out = F.linear(intermediate, down_proj[expert_idx])

            # Weight multiply + scatter
            output.index_add_(0, token_idx, expert_out * weights.unsqueeze(-1))

        elif expert_idx < zero_start:
            # Identity expert: pass through with weight
            output.index_add_(0, token_idx, current_state * weights.unsqueeze(-1))
        else:
            # Zero expert: multiply by 0, skip entirely (no-op)
            pass

    return output


# ═══════════════════════════════════════════════════════════════
# Monkey-Patch Interface
# ═══════════════════════════════════════════════════════════════

@torch._dynamo.disable
def _triton_fused_experts_forward(self, hidden_states, top_k_indices, top_k_weights):
    """Drop-in replacement for BiBoFusedExperts.forward.
    
    @torch._dynamo.disable: MoE dispatch has fundamentally dynamic shapes
    (variable tokens per expert per step) + variable act_type per expert.
    torch.compile recompiles endlessly on this — disable it here.
    The Triton kernels inside already provide the fusion benefit.
    """
    return triton_moe_experts_forward(
        hidden_states=hidden_states,
        top_k_indices=top_k_indices,
        top_k_weights=top_k_weights,
        gate_up_proj=self.gate_up_proj,
        down_proj=self.down_proj,
        expert_activations=self._expert_activations,
        num_polyglu_experts=self.num_polyglu_experts,
        identity_start=self.identity_start,
        zero_start=self.zero_start,
        num_routed_experts=self.num_routed_experts,
    )


def patch_moe_with_triton(model):
    """
    Patch BiBoFusedExperts to use Triton-fused GLU activation kernels.
    
    What's fused:
    - GLU activation (gate/up split + activation + multiply) → 1 Triton kernel
    - Eliminates 3 intermediate tensors per expert per forward pass
    
    What's NOT fused (cuBLAS is already optimal):
    - gate_up linear projection (GEMM)
    - down linear projection (GEMM)
    
    Args:
        model: BiBoForCausalLM or any model containing BiBoFusedExperts
    Returns:
        model (modified in-place)
    """
    from src.modeling.ffn.moe import BiBoFusedExperts
    
    patched = 0
    for module in model.modules():
        if isinstance(module, BiBoFusedExperts):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_fused_experts_forward.__get__(module, BiBoFusedExperts)
            patched += 1
    
    model._triton_moe_patched = True
    return model


def unpatch_moe(model):
    """Restore original PyTorch MoE implementation."""
    from src.modeling.ffn.moe import BiBoFusedExperts
    
    for module in model.modules():
        if isinstance(module, BiBoFusedExperts):
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                del module._original_forward
    
    model._triton_moe_patched = False
    return model
