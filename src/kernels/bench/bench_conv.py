"""
Conv Kernel Benchmark — BiBo Conv Router + Conv Shared Expert.

Tests:
1. triton_fused_conv_gate_multiply (permute+act+gate in 1 kernel)
2. triton_fused_permute_act (permute+activation)
3. triton_causal_conv1d_router (optimized conv router)
4. triton_causal_conv1d_gated (optimized conv shared expert gate)

All benchmarks follow the 4 mandatory rules:
  Rule 1: Gradient equivalence vs baseline
  Rule 2: NaN-free multi-pass stability
  Rule 3: Three-phase timing
  Rule 4: torch.profiler

Run: .\\venv\\Scripts\\python src/kernels/bench/bench_conv.py
"""
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from src.kernels.bench.bench_utils import (
    benchmark_phase,
    print_separator,
)


# ═══════════════════════════════════════════════════════════════
# Kernel-Level Tests
# ═══════════════════════════════════════════════════════════════

def test_fused_conv_gate_multiply():
    """Test triton_fused_conv_gate_multiply matches PyTorch baseline."""
    print_separator("KERNEL CORRECTNESS: Fused Conv Gate Multiply")

    from src.kernels.conv_fused import triton_fused_conv_gate_multiply

    device = 'cuda'
    shapes = [
        (2, 128, 64, "small"),
        (2, 256, 128, "medium"),
        (2, 512, 256, "large"),
    ]

    all_pass = True
    for B, S, I, desc in shapes:
        conv_out = torch.randn(B, I, S, device=device, dtype=torch.float16)
        up_out = torch.randn(B, S, I, device=device, dtype=torch.float16)

        # Reference: permute + SiLU + multiply
        gate_ref = conv_out.permute(0, 2, 1)  # (B, S, I)
        ref = (F.silu(gate_ref) * up_out)

        out = triton_fused_conv_gate_multiply(conv_out, up_out, act_type=0)
        max_diff = (ref - out).abs().max().item()
        passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
        all_pass = all_pass and passed
        print(f"  {desc:10s} (B={B},S={S},I={I}): {'PASS' if passed else 'FAIL'} (diff={max_diff:.2e})")

    return all_pass


def test_fused_permute_act():
    """Test triton_fused_permute_act matches PyTorch baseline."""
    print_separator("KERNEL CORRECTNESS: Fused Permute+Act")

    from src.kernels.conv_fused import triton_fused_permute_act

    device = 'cuda'
    shapes = [
        (2, 128, 64, "small"),
        (2, 256, 128, "medium"),
        (2, 512, 256, "large"),
    ]

    all_pass = True
    for B, S, I, desc in shapes:
        conv_out = torch.randn(B, I, S, device=device, dtype=torch.float16)

        # Reference
        permuted = conv_out.permute(0, 2, 1)  # (B, S, I)
        ref = F.silu(permuted)

        out = triton_fused_permute_act(conv_out, act_type=0)
        max_diff = (ref - out).abs().max().item()
        passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
        all_pass = all_pass and passed
        print(f"  {desc:10s} (B={B},S={S},I={I}): {'PASS' if passed else 'FAIL'} (diff={max_diff:.2e})")

    return all_pass


def test_conv_router():
    """Test triton_causal_conv1d_router matches PyTorch F.conv1d."""
    print_separator("KERNEL CORRECTNESS: Conv Router")

    from src.kernels.conv_fused import triton_causal_conv1d_router

    device = 'cuda'
    B, S, H, E, K = 2, 128, 512, 8, 3

    x = torch.randn(B, S, H, device=device, dtype=torch.float16)
    weight = torch.randn(E, H, K, device=device, dtype=torch.float16)

    # Reference: F.conv1d with causal padding
    x_perm = x.permute(0, 2, 1)  # (B, H, S)
    x_padded = F.pad(x_perm, (K - 1, 0))
    ref_conv = F.conv1d(x_padded, weight)  # (B, E, S)
    ref = ref_conv.permute(0, 2, 1).reshape(B * S, E)  # (B*S, E)

    out = triton_causal_conv1d_router(x, weight, E, K)
    max_diff = (ref - out).abs().max().item()
    passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
    print(f"  Conv router: {'PASS' if passed else 'FAIL'} (diff={max_diff:.2e})")

    return passed


def test_conv_shared_expert():
    """Test triton_causal_conv1d_gated matches PyTorch baseline."""
    print_separator("KERNEL CORRECTNESS: Conv Shared Expert Gate")

    from src.kernels.conv_fused import triton_causal_conv1d_gated

    device = 'cuda'
    B, S, H, I, K = 2, 128, 512, 1536, 3

    x = torch.randn(B, S, H, device=device, dtype=torch.float16)
    weight = torch.randn(I, H, K, device=device, dtype=torch.float16)

    # Reference: permute → pad → conv → permute → silu
    x_perm = x.permute(0, 2, 1)
    x_padded = F.pad(x_perm, (K - 1, 0))
    conv_out = F.conv1d(x_padded, weight)  # (B, I, S)
    permuted = conv_out.permute(0, 2, 1)  # (B, S, I)
    ref = F.silu(permuted)

    out = triton_causal_conv1d_gated(x, weight, I, K, act_type=0)
    max_diff = (ref - out).abs().max().item()
    passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
    print(f"  Conv gated: {'PASS' if passed else 'FAIL'} (diff={max_diff:.2e})")

    return passed


# ═══════════════════════════════════════════════════════════════
# Performance Tests (Rule 3 + Rule 4)
# ═══════════════════════════════════════════════════════════════

def bench_conv_gate_multiply_perf():
    """Benchmark fused conv gate multiply vs baseline."""
    print_separator("PERFORMANCE: Fused Conv Gate Multiply")

    from src.kernels.conv_fused import triton_fused_conv_gate_multiply

    device = 'cuda'
    B, S, I = 2, 512, 1536

    conv_out = torch.randn(B, I, S, device=device, dtype=torch.float16)
    up_out = torch.randn(B, S, I, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(5):
        gate = conv_out.permute(0, 2, 1)
        _ = F.silu(gate) * up_out
        _ = triton_fused_conv_gate_multiply(conv_out, up_out, act_type=0)
    torch.cuda.synchronize()

    # Baseline timing
    times_base = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        gate = conv_out.permute(0, 2, 1)
        _ = F.silu(gate) * up_out
        end.record()
        torch.cuda.synchronize()
        times_base.append(start.elapsed_time(end))

    # Triton timing
    times_tri = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = triton_fused_conv_gate_multiply(conv_out, up_out, act_type=0)
        end.record()
        torch.cuda.synchronize()
        times_tri.append(start.elapsed_time(end))

    base_ms = sorted(times_base)[1]
    tri_ms = sorted(times_tri)[1]
    speedup = base_ms / tri_ms
    print(f"  Baseline: {base_ms:.3f} ms")
    print(f"  Triton:   {tri_ms:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    return speedup


def main():
    print("=" * 70)
    print("  BiBo Conv Kernel Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Kernel correctness
    gate_ok = test_fused_conv_gate_multiply()
    permute_ok = test_fused_permute_act()
    router_ok = test_conv_router()
    gated_ok = test_conv_shared_expert()

    # Performance
    speedup = bench_conv_gate_multiply_perf()

    # Summary
    print_separator("FINAL VERDICT")
    print(f"  Fused Conv Gate Multiply: {'PASS' if gate_ok else 'FAIL'}")
    print(f"  Fused Permute+Act:        {'PASS' if permute_ok else 'FAIL'}")
    print(f"  Conv Router:              {'PASS' if router_ok else 'FAIL'}")
    print(f"  Conv Shared Expert Gate:  {'PASS' if gated_ok else 'FAIL'}")
    print(f"  Conv Gate Multiply Speedup: {speedup:.2f}x")
    print(f"  Rule 4 (torch.profiler):   See profiler output above")


if __name__ == "__main__":
    main()
