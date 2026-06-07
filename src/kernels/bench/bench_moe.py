"""
MoE Layer Benchmark — Triton vs PyTorch baseline.

Tests the Triton MoE GLU activation kernel (_fused_glu_act_kernel) and
the fused router kernel (_fused_router_kernel) against PyTorch baselines.

All benchmarks follow the 4 mandatory rules:
  Rule 1: Gradient equivalence vs baseline (original PyTorch)
  Rule 2: NaN-free multi-pass stability (>=2 fwd+bwd passes)
  Rule 3: Three-phase timing: forward-only, backward-only, forward+backward
  Rule 4: torch.profiler for all benchmarking

Run: .\\venv\\Scripts\\python src/kernels/bench/bench_moe.py
"""
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from src.kernels.bench.bench_utils import (
    benchmark_phase,
    benchmark_three_phase,
    check_gradient_equivalence,
    check_nan_stability,
    print_separator,
    print_three_phase_results,
)


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

def make_config():
    from src.configuration_bibo import BiBoConfig
    return BiBoConfig(
        vocab_size=5000, hidden_size=512, intermediate_size=1536,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=2048, use_ssmax=True,
        polyglu_expert_multiplier=2, special_expert_pairs=1,
        num_experts_per_tok=4, moe_intermediate_size=256,
        use_shared_expert=True, shared_expert_type="mlp",
        router_type="mlp", router_lambda=1.0, router_noise=0.0,
        moe_shared_scaling=2.0, gate_type="sigmoid",
        load_balance_strategy="none", tie_word_embeddings=True,
    )


# ═══════════════════════════════════════════════════════════════
# Kernel-Level Tests
# ═══════════════════════════════════════════════════════════════

def test_glu_kernel_correctness():
    """Test triton_fused_glu_activation matches PyTorch for all 3 activation types."""
    print_separator("KERNEL CORRECTNESS: Fused GLU Activation")

    from src.kernels.moe_dispatch import triton_fused_glu_activation

    device = 'cuda'
    act_names = {0: "silu", 1: "relu2", 2: "tanh"}
    shapes = [
        (64, 256, "small"),
        (256, 1536, "medium"),
        (1024, 1536, "large"),
    ]

    all_pass = True
    for act_type, act_name in act_names.items():
        print(f"\n  Activation: {act_name}")
        for M, I, desc in shapes:
            gate_up = torch.randn(M, 2 * I, device=device, dtype=torch.float16)
            gate = gate_up[:, :I].float()
            up = gate_up[:, I:].float()

            # Reference
            if act_type == 0:
                ref = (gate * torch.sigmoid(gate) * up).half()
            elif act_type == 1:
                ref = (F.relu(gate).square() * up).half()
            else:
                ref = (torch.tanh(gate) * up).half()

            out = triton_fused_glu_activation(gate_up, act_type)
            max_diff = (ref - out).abs().max().item()
            passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
            all_pass = all_pass and passed
            print(f"    {desc:10s} ({M}x{I}): {'PASS' if passed else 'FAIL'} (diff={max_diff:.2e})")

    return all_pass


def test_router_kernel_correctness():
    """Test triton_fused_router matches PyTorch baseline."""
    print_separator("KERNEL CORRECTness: Fused Router")

    from src.kernels.moe_dispatch import triton_fused_router

    device = 'cuda'
    N, E = 256, 8
    logits = torch.randn(N, E, device=device, dtype=torch.float16)
    bias = torch.randn(E, device=device, dtype=torch.float16)

    # Reference: PyTorch
    scores_ref = torch.sigmoid(logits.float())
    mean = scores_ref.mean(dim=-1, keepdim=True)
    std = scores_ref.std(dim=-1, keepdim=True) + 1e-6
    scores_norm_ref = (scores_ref - mean) / std
    selection_ref = scores_norm_ref + bias.float()

    # Triton
    scores_tri, selection_tri = triton_fused_router(logits, bias, router_lambda=1.0, use_logit_norm=True)

    score_diff = (scores_norm_ref.half() - scores_tri).abs().max().item()
    sel_diff = (selection_ref.half() - selection_tri).abs().max().item()
    passed = score_diff < 1e-2 and sel_diff < 1e-2  # Router has higher tolerance
    print(f"  Scores diff: {score_diff:.4e}")
    print(f"  Selection diff: {sel_diff:.4e}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# ═══════════════════════════════════════════════════════════════
# Module-Level Tests
# ═══════════════════════════════════════════════════════════════

def test_gradient_equivalence(config):
    """Rule 1: Gradient equivalence — patched vs unpatched BiBo MoE."""
    print_separator("RULE 1: Gradient Equivalence vs Baseline")

    from src.modeling.models import BiBoForCausalLM
    from src.kernels.moe_dispatch import patch_moe_with_triton

    device = 'cuda'
    input_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)
    label_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)

    torch.manual_seed(42)
    baseline = BiBoForCausalLM(config).to(device).train().half()
    torch.manual_seed(42)
    triton = BiBoForCausalLM(config).to(device).train().half()
    patch_moe_with_triton(triton)

    result = check_gradient_equivalence(baseline, triton, input_fn, label_fn, fp16=True)
    status = "PASS" if result['passed'] else "FAIL"
    print(f"  MoE patch gradient equivalence: {status} (max_diff={result['max_grad_diff']:.2e})")

    del baseline, triton
    torch.cuda.empty_cache()
    return result['passed']


def test_nan_stability(config):
    """Rule 2: NaN-free multi-pass stability."""
    print_separator("RULE 2: NaN-Free Multi-Pass Stability (2 passes)")

    from src.modeling.models import BiBoForCausalLM
    from src.kernels.moe_dispatch import patch_moe_with_triton

    device = 'cuda'
    model = BiBoForCausalLM(config).to(device).train().half()
    patch_moe_with_triton(model)

    input_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)
    label_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)

    result = check_nan_stability(model, input_fn, label_fn, n_passes=2)
    status = "PASS" if result['passed'] else "FAIL"
    print(f"  MoE NaN stability: {status}")
    for pr in result['pass_results']:
        print(f"    Pass {pr['pass']}: loss={pr['loss_value']:.6f}, nan_loss={pr['has_nan_loss']}, nan_grad={pr['has_nan_grad']}")

    del model
    torch.cuda.empty_cache()
    return result['passed']


def test_three_phase_performance(config):
    """Rule 3: Three-phase benchmark."""
    print_separator("RULE 3: Three-Phase Performance")

    from src.modeling.models import BiBoForCausalLM
    from src.kernels.moe_dispatch import patch_moe_with_triton

    device = 'cuda'
    batch, seq = 4, 256

    input_fn = lambda: torch.randint(0, 5000, (batch, seq), device=device)
    label_fn = lambda: torch.randint(0, 5000, (batch, seq), device=device)

    variants = [
        ("baseline", lambda: BiBoForCausalLM(config).to(device).train().half()),
        ("triton_moe", lambda: BiBoForCausalLM(config).to(device).train().half()),
    ]

    all_results = {}
    baseline_results = None

    for name, setup_fn in variants:
        print(f"\n  Benchmarking {name}...")
        model = setup_fn()
        if name == "triton_moe":
            patch_moe_with_triton(model)

        result = benchmark_three_phase(
            model=model,
            input_fn=lambda: input_fn(),
            label_fn=lambda: label_fn(),
            n_warmup=5, n_steps=3, do_profile=True,
        )

        all_results[name] = result
        if name == "baseline":
            baseline_results = result

        print_three_phase_results(result, name, baseline_results)
        del model
        torch.cuda.empty_cache()

    # Summary
    print_separator("MoE PERFORMANCE SUMMARY")
    base_fb = baseline_results['fwd_bwd']['median_ms']
    print(f"  {'Variant':20s} | {'Fwd (ms)':>10s} | {'Bwd (ms)':>10s} | {'Fwd+Bwd (ms)':>12s} | {'Speedup':>10s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    for name in all_results:
        r = all_results[name]
        fb = r['fwd_bwd']['median_ms']
        speedup = base_fb / fb if fb > 0 else 0
        print(f"  {name:20s} | {r['forward']['median_ms']:10.3f} | {r['backward']['median_ms']:10.3f} | {fb:12.3f} | {speedup:9.2f}x")

    return all_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BiBo MoE Layer Benchmark — Triton vs Baseline")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    config = make_config()

    # Kernel correctness
    glu_ok = test_glu_kernel_correctness()
    router_ok = test_router_kernel_correctness()

    # Rule 1: Gradient equivalence
    grad_ok = test_gradient_equivalence(config)

    # Rule 2: NaN stability
    nan_ok = test_nan_stability(config)

    # Rule 3: Three-phase performance
    perf_results = test_three_phase_performance(config)

    # Summary
    print_separator("FINAL VERDICT")
    print(f"  GLU Kernel Correctness:    {'PASS' if glu_ok else 'FAIL'}")
    print(f"  Router Kernel Correctness: {'PASS' if router_ok else 'FAIL'}")
    print(f"  Rule 1 (Grad Equiv):       {'PASS' if grad_ok else 'FAIL'}")
    print(f"  Rule 2 (NaN Stable):       {'PASS' if nan_ok else 'FAIL'}")
    print(f"  Rule 3 (Perf):             See table above")
    print(f"  Rule 4 (torch.profiler):   See profiler output above")


if __name__ == "__main__":
    main()
