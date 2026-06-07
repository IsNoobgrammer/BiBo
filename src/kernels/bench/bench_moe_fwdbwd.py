"""
MoE Full Training Step Benchmark — Forward + Backward through full model.

Measures the COMPLETE training iteration cost with all MoE layers:
  forward pass + loss.backward() + gradient accumulation

All benchmarks follow the 4 mandatory rules:
  Rule 1: Gradient equivalence vs baseline
  Rule 2: NaN-free multi-pass stability
  Rule 3: Three-phase timing (fwd, bwd, fwd+bwd)
  Rule 4: torch.profiler

Run: .\\venv\\Scripts\\python src/kernels/bench/bench_moe_fwdbwd.py
"""
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

import torch

from src.kernels.bench.bench_utils import (
    benchmark_phase,
    benchmark_three_phase,
    check_gradient_equivalence,
    check_nan_stability,
    print_separator,
    print_three_phase_results,
)


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


def setup_models(config, device):
    """Create baseline and triton-patched models."""
    from src.modeling.models import BiBoForCausalLM
    from src.kernels.patch import patch_bibo_with_triton
    from src.kernels.moe_dispatch import patch_moe_with_triton

    torch.manual_seed(42)
    baseline = BiBoForCausalLM(config).to(device).train().half()

    torch.manual_seed(42)
    triton = BiBoForCausalLM(config).to(device).train().half()
    patch_bibo_with_triton(triton)
    patch_moe_with_triton(triton)

    return baseline, triton


def test_gradient_equivalence(config):
    """Rule 1: Gradient equivalence — full model with all MoE patches."""
    print_separator("RULE 1: Gradient Equivalence (full model + MoE patches)")

    device = 'cuda'
    input_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)
    label_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)

    baseline, triton = setup_models(config, device)
    result = check_gradient_equivalence(baseline, triton, input_fn, label_fn, fp16=True)

    status = "PASS" if result['passed'] else "FAIL"
    print(f"  Full model gradient equivalence: {status} (max_diff={result['max_grad_diff']:.2e})")

    del baseline, triton
    torch.cuda.empty_cache()
    return result['passed']


def test_nan_stability(config):
    """Rule 2: NaN-free multi-pass stability on full model."""
    print_separator("RULE 2: NaN-Free Stability (full model, 2 passes)")

    device = 'cuda'
    from src.modeling.models import BiBoForCausalLM
    from src.kernels.patch import patch_bibo_with_triton
    from src.kernels.moe_dispatch import patch_moe_with_triton

    torch.manual_seed(42)
    model = BiBoForCausalLM(config).to(device).train().half()
    patch_bibo_with_triton(model)
    patch_moe_with_triton(model)

    input_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)
    label_fn = lambda: torch.randint(0, 5000, (2, 64), device=device)

    result = check_nan_stability(model, input_fn, label_fn, n_passes=2)
    status = "PASS" if result['passed'] else "FAIL"
    print(f"  Full model NaN stability: {status}")
    for pr in result['pass_results']:
        print(f"    Pass {pr['pass']}: loss={pr['loss_value']:.6f}, nan={pr['has_nan_loss']}, nan_grad={pr['has_nan_grad']}")

    del model
    torch.cuda.empty_cache()
    return result['passed']


def test_three_phase_performance(config):
    """Rule 3: Three-phase benchmark — baseline vs triton full model."""
    print_separator("RULE 3: Three-Phase Performance (full model)")

    device = 'cuda'
    batch, seq = 4, 256

    input_fn = lambda: torch.randint(0, 5000, (batch, seq), device=device)
    label_fn = lambda: torch.randint(0, 5000, (batch, seq), device=device)

    all_results = {}
    baseline_results = None

    for name in ["baseline", "triton_all"]:
        print(f"\n  Benchmarking {name}...")
        baseline, triton = setup_models(config, device)
        model = baseline if name == "baseline" else triton

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
        del baseline, triton
        torch.cuda.empty_cache()

    # Summary
    print_separator("FULL MODEL PERFORMANCE SUMMARY")
    base_fb = baseline_results['fwd_bwd']['median_ms']
    print(f"  {'Variant':20s} | {'Fwd (ms)':>10s} | {'Bwd (ms)':>10s} | {'Fwd+Bwd (ms)':>12s} | {'Speedup':>10s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
    for name in all_results:
        r = all_results[name]
        fb = r['fwd_bwd']['median_ms']
        speedup = base_fb / fb if fb > 0 else 0
        print(f"  {name:20s} | {r['forward']['median_ms']:10.3f} | {r['backward']['median_ms']:10.3f} | {fb:12.3f} | {speedup:9.2f}x")

    return all_results


def main():
    print("=" * 70)
    print("  BiBo Full Model MoE Benchmark — Forward + Backward")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    config = make_config()

    grad_ok = test_gradient_equivalence(config)
    nan_ok = test_nan_stability(config)
    perf_results = test_three_phase_performance(config)

    print_separator("FINAL VERDICT")
    print(f"  Rule 1 (Grad Equiv):  {'PASS' if grad_ok else 'FAIL'}")
    print(f"  Rule 2 (NaN Stable):  {'PASS' if nan_ok else 'FAIL'}")
    print(f"  Rule 3 (Perf):        See table above")
    print(f"  Rule 4 (Profiler):    See profiler output above")


if __name__ == "__main__":
    main()
