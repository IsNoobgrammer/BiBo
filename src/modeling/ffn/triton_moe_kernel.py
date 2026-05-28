"""
Triton-accelerated MoE expert dispatch for BiBo.

Strategy: Don't reimplement GEMM in Triton (cuBLAS is already optimal).
Instead, fuse the operations AROUND the GEMMs that cause kernel launch overhead:
  1. Fused gather + weight application (scatter-to-output after down proj)
  2. Fused activation kernels (SiLU-GLU, ReLU²-GLU, Tanh-GLU)
  3. Fused weighted scatter-add

For BiBo's scale (hidden=320, intermediate=768, ~1024 tokens/expert):
- The GEMMs themselves are fast (cuBLAS handles them)
- The overhead is in gather/scatter/activation being separate kernel launches
- Fusing activation with the gating (gate * up) saves one kernel launch per expert

On Kaggle (Linux + T4/A100), Triton kernels compile and run.
On Windows, falls back to PyTorch (still fast due to grouped dispatch).
"""

import torch
import torch.nn.functional as F

# Try importing Triton — available on Linux (Kaggle), not performant on Windows
try:
    import triton
    import triton.language as tl
    import platform
    # Triton kernels are only beneficial on Linux (native Triton).
    # triton-windows exists but atomic_add and JIT are slower than PyTorch.
    HAS_TRITON = (platform.system() == "Linux")
    _TRITON_IMPORTABLE = True
except ImportError:
    HAS_TRITON = False
    _TRITON_IMPORTABLE = False


# ═══════════════════════════════════════════════════════════════
# Triton Kernels (Linux/Kaggle only)
# ═══════════════════════════════════════════════════════════════

if _TRITON_IMPORTABLE:

    @triton.jit
    def _fused_silu_gate_kernel(
        gate_ptr,       # (N, intermediate_size) — gate projection output
        up_ptr,         # (N, intermediate_size) — up projection output  
        out_ptr,        # (N, intermediate_size) — SiLU(gate) * up
        N,              # number of tokens
        D: tl.constexpr,  # intermediate_size
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused SiLU(gate) * up — eliminates 2 separate elementwise kernels."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        n_mask = n_offsets < N
        d_mask = d_offsets < D
        mask = n_mask[:, None] & d_mask[None, :]
        
        # Load gate and up values
        offsets = n_offsets[:, None] * D + d_offsets[None, :]
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # SiLU(gate) * up = gate * sigmoid(gate) * up
        result = gate * tl.sigmoid(gate) * up
        
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)

    @triton.jit
    def _fused_relu2_gate_kernel(
        gate_ptr, up_ptr, out_ptr,
        N, D: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Fused ReLU²(gate) * up."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        n_mask = n_offsets < N
        d_mask = d_offsets < D
        mask = n_mask[:, None] & d_mask[None, :]
        
        offsets = n_offsets[:, None] * D + d_offsets[None, :]
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # ReLU²(gate) * up = max(0, gate)² * up
        relu_gate = tl.maximum(gate, 0.0)
        result = relu_gate * relu_gate * up
        
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)

    @triton.jit
    def _fused_tanh_gate_kernel(
        gate_ptr, up_ptr, out_ptr,
        N, D: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Fused tanh(gate) * up."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        n_mask = n_offsets < N
        d_mask = d_offsets < D
        mask = n_mask[:, None] & d_mask[None, :]
        
        offsets = n_offsets[:, None] * D + d_offsets[None, :]
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # tanh via libdevice
        result = tl.extra.cuda.libdevice.tanh(gate) * up
        
        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)

    @triton.jit
    def _fused_weighted_scatter_add_kernel(
        expert_output_ptr,   # (N, hidden_size) — expert output to scatter
        output_ptr,          # (total_tokens, hidden_size) — global output accumulator
        token_ids_ptr,       # (N,) — which global token each row maps to
        weights_ptr,         # (N,) — routing weights
        N,                   # number of tokens for this expert
        D: tl.constexpr,     # hidden_size
        stride_out: tl.constexpr,  # output stride (hidden_size)
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused: output[token_ids] += expert_output * weights."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        
        n_mask = n_offsets < N
        d_mask = d_offsets < D
        
        # Load token IDs and weights
        token_ids = tl.load(token_ids_ptr + n_offsets, mask=n_mask, other=0)
        weights = tl.load(weights_ptr + n_offsets, mask=n_mask, other=0.0)
        
        # Load expert output
        in_offsets = n_offsets[:, None] * D + d_offsets[None, :]
        mask_2d = n_mask[:, None] & d_mask[None, :]
        expert_out = tl.load(expert_output_ptr + in_offsets, mask=mask_2d, other=0.0)
        
        # Apply weights
        weighted = expert_out * weights[:, None]
        
        # Scatter-add to output
        out_offsets = token_ids[:, None] * stride_out + d_offsets[None, :]
        tl.atomic_add(output_ptr + out_offsets, weighted, mask=mask_2d)


def _triton_fused_glu_activation(gate: torch.Tensor, up: torch.Tensor, activation: str) -> torch.Tensor:
    """
    Launch the appropriate fused GLU activation Triton kernel.
    Fuses: activation(gate) * up into a single kernel (saves 1-2 kernel launches).
    """
    N, D = gate.shape
    out = torch.empty_like(gate)
    
    # Grid: tile over (N, D)
    BLOCK_N = 32
    BLOCK_D = min(128, triton.next_power_of_2(D))
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
    
    if activation == "silu":
        _fused_silu_gate_kernel[grid](gate, up, out, N, D, BLOCK_N, BLOCK_D)
    elif activation == "relu2":
        _fused_relu2_gate_kernel[grid](gate, up, out, N, D, BLOCK_N, BLOCK_D)
    else:  # tanh
        _fused_tanh_gate_kernel[grid](gate, up, out, N, D, BLOCK_N, BLOCK_D)
    
    return out


def _triton_weighted_scatter_add(
    expert_output: torch.Tensor,
    output: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
):
    """
    Fused weighted scatter-add via Triton.
    output[token_ids] += expert_output * weights[:, None]
    """
    N, D = expert_output.shape
    BLOCK_N = 32
    BLOCK_D = min(128, triton.next_power_of_2(D))
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(D, BLOCK_D))
    
    _fused_weighted_scatter_add_kernel[grid](
        expert_output, output, token_ids, weights,
        N, D, output.stride(0),
        BLOCK_N, BLOCK_D,
    )


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def fused_moe_activation_group(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    activation: str = "silu",
):
    """
    Dispatch one activation group of MoE experts.
    
    On Linux (Kaggle): Uses Triton kernels for fused activation + scatter.
    On Windows: Falls back to pure PyTorch.
    
    The GEMMs (gate_up and down projections) always use cuBLAS via F.linear
    because cuBLAS is already optimal for these shapes.
    
    What Triton fuses:
      - activation(gate) * up → single kernel instead of 2-3 separate ops
      - output[token_ids] += result * weights → fused scatter-add
    
    Args:
        hidden_states: (num_tokens, hidden_size)
        output: (num_tokens, hidden_size) — accumulated output
        gate_up_weights: (num_experts_in_group, 2*intermediate, hidden)
        down_weights: (num_experts_in_group, hidden, intermediate)
        sorted_token_ids: token indices sorted by expert
        sorted_weights: routing weights sorted by expert
        expert_offsets: (num_experts_in_group + 1,) cumulative token counts
        activation: "silu", "relu2", or "tanh"
    """
    if HAS_TRITON and hidden_states.is_cuda:
        _triton_activation_group_dispatch(
            hidden_states, output, gate_up_weights, down_weights,
            sorted_token_ids, sorted_weights, expert_offsets, activation
        )
    else:
        _pytorch_activation_group_dispatch(
            hidden_states, output, gate_up_weights, down_weights,
            sorted_token_ids, sorted_weights, expert_offsets, activation
        )


def _triton_activation_group_dispatch(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    activation: str,
):
    """
    Triton-accelerated dispatch for one activation group.
    
    Per expert:
      1. Gather tokens (indexing — no kernel needed)
      2. gate_up = F.linear(tokens, weight)  [cuBLAS GEMM]
      3. activated = triton_fused_glu(gate, up)  [FUSED — saves 2 kernels]
      4. down_out = F.linear(activated, down_weight)  [cuBLAS GEMM]
      5. triton_weighted_scatter_add(down_out, output, ids, weights)  [FUSED]
    
    Total kernel launches per expert: 2 GEMMs + 1 fused activation + 1 fused scatter = 4
    vs old code: 2 GEMMs + 3 elementwise + 1 index_add = 6
    Savings: 2 kernel launches per expert × num_experts_in_group
    """
    num_experts = gate_up_weights.shape[0]
    
    for i in range(num_experts):
        start = expert_offsets[i].item()
        end = expert_offsets[i + 1].item()
        if start == end:
            continue
        
        token_idx = sorted_token_ids[start:end]
        weights = sorted_weights[start:end]
        current_state = hidden_states[token_idx]  # gather (just indexing)
        
        # Step 2: gate_up GEMM (cuBLAS — optimal)
        gate_up = F.linear(current_state, gate_up_weights[i])
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Step 3: Fused activation — Triton kernel
        activated = _triton_fused_glu_activation(gate.contiguous(), up.contiguous(), activation)
        
        # Step 4: down GEMM (cuBLAS — optimal)
        expert_output = F.linear(activated, down_weights[i])
        
        # Step 5: Fused weighted scatter-add — Triton kernel
        _triton_weighted_scatter_add(expert_output, output, token_idx, weights)


def _pytorch_activation_group_dispatch(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    activation: str,
):
    """
    PyTorch fallback for activation-grouped dispatch.
    Used on Windows or when Triton is unavailable.
    """
    num_experts = gate_up_weights.shape[0]
    
    if activation == "silu":
        act_fn = F.silu
    elif activation == "relu2":
        act_fn = lambda x: F.relu(x).square()
    else:  # tanh
        act_fn = torch.tanh
    
    for i in range(num_experts):
        start = expert_offsets[i].item()
        end = expert_offsets[i + 1].item()
        if start == end:
            continue
        
        token_idx = sorted_token_ids[start:end]
        weights = sorted_weights[start:end].unsqueeze(-1)
        current_state = hidden_states[token_idx]
        
        gate_up = F.linear(current_state, gate_up_weights[i])
        gate, up = gate_up.chunk(2, dim=-1)
        activated = act_fn(gate)
        expert_output = F.linear(activated * up, down_weights[i])
        output.index_add_(0, token_idx, expert_output * weights)
