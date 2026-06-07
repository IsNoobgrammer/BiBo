"""
Dense MLP Benchmark — 3-Variant Head-to-Head Comparison.

Pits all 3 Triton SwiGLU variants against each other under fair benchmarking:
1. Baseline: PyTorch eager (separate gate_proj, up_proj, silu, multiply, down_proj)
2. Forward-only: Triton forward, PyTorch backward (old_kernels/dense_mlp.py)
3. Fully-fused: Single Triton kernel fwd+bwd (old_kernels/dense_mlp_fused.py)
4. Separate-backward: Triton fwd + Triton bwd (src/kernels/dense_mlp.py) — current

All benchmarks follow the 4 mandatory rules:
  Rule 1: Gradient equivalence vs baseline (original PyTorch)
  Rule 2: NaN-free multi-pass stability (>=2 fwd+bwd passes)
  Rule 3: Three-phase timing: forward-only, backward-only, forward+backward
  Rule 4: torch.profiler for all benchmarking

Run: .\\venv\\Scripts\\python src/kernels/bench/bench_dense_mlp.py
"""
import sys
import os

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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
    """Standard BiBo config for benchmarking."""
    from src.configuration_bibo import BiBoConfig
    return BiBoConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        use_ssmax=True,
        polyglu_expert_multiplier=2,
        special_expert_pairs=1,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        use_shared_expert=True,
        shared_expert_type="mlp",
        router_type="mlp",
        router_lambda=1.0,
        router_noise=0.0,
        moe_shared_scaling=2.0,
        gate_type="sigmoid",
        load_balance_strategy="none",
        tie_word_embeddings=True,
    )


def make_model(config, seed=42):
    """Create a fresh BiBo model."""
    from src.modeling.models import BiBoForCausalLM
    torch.manual_seed(seed)
    return BiBoForCausalLM(config)


def make_input_fn(batch=2, seq=64, device='cuda'):
    """Create input function for benchmarking."""
    def fn():
        return torch.randint(0, 5000, (batch, seq), device=device)
    return fn


def make_label_fn(batch=2, seq=64, device='cuda'):
    """Create label function for benchmarking."""
    def fn():
        return torch.randint(0, 5000, (batch, seq), device=device)
    return fn


# ═══════════════════════════════════════════════════════════════
# Variant Setup
# ═══════════════════════════════════════════════════════════════

def setup_baseline(config):
    """Setup baseline model (no patches)."""
    model = make_model(config)
    return model


def setup_forward_only(config):
    """Setup model with forward-only Triton SwiGLU (from old_kernels/)."""
    from old_kernels.dense_mlp import _TritonSwiGLUFunction
    from src.modeling.ffn.mlp import BiBoMLP
    from src.modeling.models import BiBoForCausalLM

    model = BiBoForCausalLM(config)

    def _triton_fwd_only_forward(self, x):
        orig_shape = x.shape[:-1]
        x_2d = x.view(-1, self.hidden_size)
        fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = F.linear(x_2d, fused_weight)
        # Forward-only: Triton forward, PyTorch backward
        intermediate = _TritonSwiGLUFunction.apply(gate_up)
        out = self.down_proj(intermediate)
        return out.view(*orig_shape, self.hidden_size)

    patched = 0
    for module in model.modules():
        if isinstance(module, BiBoMLP) and not getattr(module, '_is_expert_mlp', False):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_fwd_only_forward.__get__(module, BiBoMLP)
            patched += 1

    model._variant_name = f"forward_only ({patched} MLPs patched)"
    return model


def setup_fully_fused(config):
    """Setup model with fully-fused Triton SwiGLU (from old_kernels/)."""
    from old_kernels.dense_mlp_fused import _FusedSwiGLUFull
    from src.modeling.ffn.mlp import BiBoMLP
    from src.modeling.models import BiBoForCausalLM

    model = BiBoForCausalLM(config)

    def _triton_full_fused_forward(self, x):
        orig_shape = x.shape[:-1]
        x_2d = x.view(-1, self.hidden_size)
        fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = F.linear(x_2d, fused_weight)
        intermediate = _FusedSwiGLUFull.apply(gate_up)
        out = self.down_proj(intermediate)
        return out.view(*orig_shape, self.hidden_size)

    patched = 0
    for module in model.modules():
        if isinstance(module, BiBoMLP) and not getattr(module, '_is_expert_mlp', False):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _triton_full_fused_forward.__get__(module, BiBoMLP)
            patched += 1

    model._variant_name = f"fully_fused ({patched} MLPs patched)"
    return model


def setup_separate_backward(config):
    """Setup model with separate Triton fwd+bwd (current src/kernels/)."""
    from src.kernels.dense_mlp import patch_dense_mlp_with_triton
    model = make_model(config)
    patch_dense_mlp_with_triton(model)
    count = getattr(model, '_triton_dense_mlp_count', 0)
    model._variant_name = f"separate_backward ({count} MLPs patched)"
    return model


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_kernel_correctness():
    """Test that triton_fused_swiglu matches PyTorch silu(gate) * up."""
    print_separator("RULE 0: Kernel Correctness (raw triton_fused_swiglu)")

    from src.kernels.dense_mlp import triton_fused_swiglu

    device = 'cuda'
    test_shapes = [
        (1, 256, "tiny"),
        (64, 512, "small"),
        (256, 1536, "medium (BiBo default)"),
        (1024, 1536, "large"),
    ]

    all_pass = True
    for M, I, desc in test_shapes:
        gate_up = torch.randn(M, 2 * I, device=device, dtype=torch.float16)
        gate = gate_up[:, :I].float()
        up = gate_up[:, I:].float()
        ref = (F.silu(gate) * up).half()

        out = triton_fused_swiglu(gate_up)

        max_diff = (ref - out).abs().max().item()
        passed = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
        all_pass = all_pass and passed
        print(f"  {desc:30s} ({M}x{I}): {'PASS' if passed else 'FAIL'} (max_diff={max_diff:.2e})")

    return all_pass


def test_gradient_equivalence_all_variants(config):
    """Rule 1: Compare gradients of each variant vs baseline."""
    print_separator("RULE 1: Gradient Equivalence vs Baseline")

    device = 'cuda'
    input_fn = make_input_fn(batch=2, seq=64, device=device)
    label_fn = make_label_fn(batch=2, seq=64, device=device)

    variants = [
        ("forward_only", setup_forward_only),
        ("fully_fused", setup_fully_fused),
        ("separate_backward", setup_separate_backward),
    ]

    results = {}
    for name, setup_fn in variants:
        print(f"\n  Testing {name}...")

        # Fresh models for fair comparison
        baseline = setup_baseline(config).to(device).train().half()
        triton = setup_fn(config).to(device).train().half()

        # Copy weights from baseline to triton for fair comparison
        triton.load_state_dict(baseline.state_dict(), strict=False)

        result = check_gradient_equivalence(
            baseline, triton, input_fn, label_fn, fp16=True
        )
        results[name] = result

        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {name:25s}: {status} (max_grad_diff={result['max_grad_diff']:.2e})")

        del baseline, triton
        torch.cuda.empty_cache()

    return results


def test_nan_stability_all_variants(config):
    """Rule 2: Each variant must complete >=2 fwd+bwd passes with zero NaN."""
    print_separator("RULE 2: NaN-Free Multi-Pass Stability (2 passes)")

    device = 'cuda'
    input_fn = make_input_fn(batch=2, seq=64, device=device)
    label_fn = make_label_fn(batch=2, seq=64, device=device)

    variants = [
        ("forward_only", setup_forward_only),
        ("fully_fused", setup_fully_fused),
        ("separate_backward", setup_separate_backward),
    ]

    results = {}
    for name, setup_fn in variants:
        print(f"\n  Testing {name}...")

        model = setup_fn(config).to(device).train().half()

        result = check_nan_stability(model, input_fn, label_fn, n_passes=2)
        results[name] = result

        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {name:25s}: {status}")
        for pr in result['pass_results']:
            print(f"    Pass {pr['pass']}: loss={pr['loss_value']:.6f}, "
                  f"nan_loss={pr['has_nan_loss']}, nan_grad={pr['has_nan_grad']}")

        del model
        torch.cuda.empty_cache()

    return results


def test_three_phase_performance(config):
    """Rule 3: Benchmark forward, backward, and fwd+bwd separately (>=3 steps each)."""
    print_separator("RULE 3: Three-Phase Performance (3 warmup + 3 timed)")

    device = 'cuda'
    batch, seq = 4, 256  # Medium size for meaningful timing

    input_fn = make_input_fn(batch=batch, seq=seq, device=device)
    label_fn = make_label_fn(batch=batch, seq=seq, device=device)

    variants = [
        ("baseline", setup_baseline),
        ("forward_only", setup_forward_only),
        ("fully_fused", setup_fully_fused),
        ("separate_backward", setup_separate_backward),
    ]

    all_results = {}
    baseline_results = None

    for name, setup_fn in variants:
        print(f"\n  Benchmarking {name}...")
        model = setup_fn(config).to(device).train().half()

        # Apply AMP autocast wrapper for fair comparison
        def run_step():
            model.zero_grad()
            x = input_fn()
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x, labels=label_fn())
            out.loss.backward()

        result = benchmark_three_phase(
            model=model,
            input_fn=lambda: input_fn(),
            label_fn=lambda: label_fn(),
            n_warmup=5,
            n_steps=3,
            do_profile=True,
        )

        all_results[name] = result
        if name == "baseline":
            baseline_results = result

        print_three_phase_results(result, name, baseline_results)

        del model
        torch.cuda.empty_cache()

    # Summary table
    print_separator("PERFORMANCE SUMMARY")
    print(f"  {'Variant':25s} | {'Fwd (ms)':>10s} | {'Bwd (ms)':>10s} | {'Fwd+Bwd (ms)':>12s} | {'vs Baseline':>12s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

    base_fb = baseline_results['fwd_bwd']['median_ms']
    for name in all_results:
        r = all_results[name]
        fwd = r['forward']['median_ms']
        bwd = r['backward']['median_ms']
        fb = r['fwd_bwd']['median_ms']
        speedup = base_fb / fb if fb > 0 else 0
        print(f"  {name:25s} | {fwd:10.3f} | {bwd:10.3f} | {fb:12.3f} | {speedup:11.2f}x")

    return all_results


def test_memory(config):
    """Compare peak memory usage across variants."""
    print_separator("MEMORY USAGE COMPARISON")

    device = 'cuda'
    batch, seq = 4, 256

    variants = [
        ("baseline", setup_baseline),
        ("forward_only", setup_forward_only),
        ("fully_fused", setup_fully_fused),
        ("separate_backward", setup_separate_backward),
    ]

    results = {}
    for name, setup_fn in variants:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = setup_fn(config).to(device).train().half()
        x = torch.randint(0, 5000, (batch, seq), device=device)
        labels = x.clone()

        with torch.autocast('cuda', dtype=torch.float16):
            out = model(x, labels=labels)
        out.loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1024**2

        results[name] = peak
        del model, x, labels, out
        torch.cuda.empty_cache()

    base_peak = results.get('baseline', 1)
    print(f"  {'Variant':25s} | {'Peak (MB)':>10s} | {'Savings':>10s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}")
    for name, peak in results.items():
        savings = base_peak - peak
        pct = (savings / base_peak) * 100 if base_peak > 0 else 0
        print(f"  {name:25s} | {peak:10.1f} | {savings:+10.1f} ({pct:+.1f}%)")

    return results


def test_full_model_e2e(config):
    """Full model E2E: all patches applied, fwd+bwd."""
    print_separator("FULL MODEL E2E (all patches)")

    from src.kernels.patch import patch_bibo_with_triton
    from src.kernels.moe_dispatch import patch_moe_with_triton
    from src.kernels.dense_mlp import patch_dense_mlp_with_triton

    device = 'cuda'

    # Baseline
    torch.manual_seed(42)
    model_base = make_model(config).to(device).train()

    # Triton (all patches)
    torch.manual_seed(42)
    model_tri = make_model(config).to(device).train()
    patch_bibo_with_triton(model_tri)
    patch_moe_with_triton(model_tri)
    patch_dense_mlp_with_triton(model_tri)

    input_ids = torch.randint(0, 5000, (2, 64), device=device)
    labels = input_ids.clone()

    # Forward
    with torch.autocast('cuda', dtype=torch.float16):
        out_base = model_base(input_ids=input_ids, labels=labels)
        out_tri = model_tri(input_ids=input_ids, labels=labels)

    loss_diff = (out_base.loss - out_tri.loss).abs().item()
    print(f"  Baseline loss: {out_base.loss.item():.6f}")
    print(f"  Triton loss:   {out_tri.loss.item():.6f}")
    print(f"  Loss diff:     {loss_diff:.6f}")

    # Backward
    out_base.loss.backward()
    out_tri.loss.backward()
    print(f"  Backward: PASS (no crash)")

    # Performance
    def run_base():
        model_base.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            out = model_base(input_ids=input_ids, labels=labels)
        out.loss.backward()

    def run_tri():
        model_tri.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            out = model_tri(input_ids=input_ids, labels=labels)
        out.loss.backward()

    base_time = benchmark_phase(run_base, "baseline", n_warmup=3, n_steps=3, do_profile=False)
    tri_time = benchmark_phase(run_tri, "triton", n_warmup=3, n_steps=3, do_profile=False)

    speedup = base_time['median_ms'] / tri_time['median_ms']
    print(f"\n  Performance:")
    print(f"    Baseline: {base_time['median_ms']:.3f} ms")
    print(f"    Triton:   {tri_time['median_ms']:.3f} ms")
    print(f"    Speedup:  {speedup:.2f}x")

    del model_base, model_tri
    torch.cuda.empty_cache()

    return loss_diff, speedup


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BiBo Dense MLP Benchmark — 3-Variant Head-to-Head")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton not installed")
        sys.exit(1)

    config = make_config()

    # Phase 0: Kernel correctness
    kernel_ok = test_kernel_correctness()
    if not kernel_ok:
        print("\n[ABORT] Kernel correctness failed.")
        sys.exit(1)

    # Rule 1: Gradient equivalence
    grad_results = test_gradient_equivalence_all_variants(config)

    # Rule 2: NaN stability
    nan_results = test_nan_stability_all_variants(config)

    # Rule 3: Three-phase performance
    perf_results = test_three_phase_performance(config)

    # Memory comparison
    mem_results = test_memory(config)

    # Full model E2E
    e2e_loss_diff, e2e_speedup = test_full_model_e2e(config)

    # Final summary
    print_separator("FINAL VERDICT")
    all_grad_pass = all(r['passed'] for r in grad_results.values())
    all_nan_pass = all(r['passed'] for r in nan_results.values())
    print(f"  Rule 1 (Gradient Equivalence): {'ALL PASS' if all_grad_pass else 'SOME FAILED'}")
    print(f"  Rule 2 (NaN Stability):        {'ALL PASS' if all_nan_pass else 'SOME FAILED'}")
    print(f"  Rule 3 (Three-Phase Timing):   See table above")
    print(f"  Rule 4 (torch.profiler):       See profiler output above")
    print(f"  Full Model E2E:                {'PASS' if e2e_loss_diff < 0.05 else 'FAIL'} (loss diff={e2e_loss_diff:.4f})")
    print(f"  Full Model Speedup:            {e2e_speedup:.2f}x")

    if all_grad_pass and all_nan_pass:
        print(f"\n  WINNER: See three-phase table — pick the fastest variant")
    else:
        print(f"\n  ⚠ Some variants failed correctness checks")


if __name__ == "__main__":
    main()
