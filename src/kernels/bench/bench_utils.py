"""
Shared Benchmarking Utilities for BiBo Kernel Benchmarks.

All benchmarks follow the 4 mandatory rules (see AGENTS.md Triton Kernels section):
1. Gradient equivalence vs baseline (original PyTorch, not Triton-patched)
2. NaN-free multi-pass stability (>=2 fwd+bwd passes)
3. Three-phase timing: forward-only, backward-only, forward+backward (>=3 steps each)
4. torch.profiler for all benchmarking (never time.time())

Usage:
    from src.kernels.bench.bench_utils import (
        benchmark_phase,
        benchmark_three_phase,
        check_gradient_equivalence,
        check_nan_stability,
        print_separator,
    )
"""
import torch
import torch.profiler
from typing import Callable, Dict, List, Optional, Tuple


def print_separator(title: str, char: str = "=", width: int = 70):
    """Print a formatted section separator."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def benchmark_phase(
    fn: Callable,
    phase_name: str,
    n_warmup: int = 5,
    n_steps: int = 3,
    do_profile: bool = True,
) -> Dict:
    """
    Benchmark a single phase (forward, backward, or fwd+bwd).

    Rule 3: Each phase runs n_warmup warmup + n_steps timed iterations,
    then averages. Rule 4: torch.profiler for kernel breakdown on last step.

    Args:
        fn: Callable to benchmark (must handle its own zero_grad if needed)
        phase_name: "forward" | "backward" | "fwd_bwd"
        n_warmup: Warmup iterations (to fill Triton autotune cache)
        n_steps: Timed iterations (results averaged)
        do_profile: Whether to run torch.profiler for kernel breakdown

    Returns:
        {
            'phase': str,
            'median_ms': float,
            'mean_ms': float,
            'all_times': list[float],
            'profiler_table': str | None,
        }
    """
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Timed iterations
    times = []
    last_profiler_table = None

    for i in range(n_steps):
        is_last = (i == n_steps - 1)
        if do_profile and is_last:
            # Profile last iteration for kernel breakdown
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            last_profiler_table = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=20
            )
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    sorted_times = sorted(times)
    return {
        'phase': phase_name,
        'median_ms': sorted_times[len(sorted_times) // 2],
        'mean_ms': sum(times) / len(times),
        'all_times': times,
        'profiler_table': last_profiler_table,
    }


def benchmark_three_phase(
    model: torch.nn.Module,
    input_fn: Callable,
    label_fn: Optional[Callable] = None,
    n_warmup: int = 5,
    n_steps: int = 3,
    do_profile: bool = True,
) -> Dict:
    """
    Benchmark forward-only, backward-only, and fwd+bwd separately.

    Rule 3: Three separate phases, each with >=3 warmup + >=3 timed steps.

    Args:
        model: The model to benchmark (in train mode)
        input_fn: Callable that returns input tensor(s) — called fresh each step
        label_fn: Optional callable that returns labels (if model uses labels= kwarg)
        n_warmup: Warmup iterations per phase
        n_steps: Timed iterations per phase
        do_profile: Whether to use torch.profiler

    Returns:
        {
            'forward': benchmark_phase result dict,
            'backward': benchmark_phase result dict,
            'fwd_bwd': benchmark_phase result dict,
        }
    """
    use_labels = label_fn is not None

    def forward_only():
        with torch.no_grad():
            x = input_fn()
            if use_labels:
                model(x, labels=label_fn())
            else:
                model(x)

    def backward_only():
        x = input_fn()
        if use_labels:
            out = model(x, labels=label_fn())
        else:
            out = model(x)
            out = out.sum() if not hasattr(out, 'loss') else out
        out.loss.backward() if hasattr(out, 'loss') else out.backward()
        model.zero_grad()

    def fwd_bwd():
        model.zero_grad()
        x = input_fn()
        if use_labels:
            out = model(x, labels=label_fn())
        else:
            out = model(x)
            out = out.sum() if not hasattr(out, 'loss') else out
        out.loss.backward() if hasattr(out, 'loss') else out.backward()

    # Phase 1: Forward-only
    fwd = benchmark_phase(forward_only, "forward", n_warmup, n_steps, do_profile)

    # Phase 2: Backward-only
    bwd = benchmark_phase(backward_only, "backward", n_warmup, n_steps, do_profile)

    # Phase 3: Forward+Backward
    fb = benchmark_phase(fwd_bwd, "fwd_bwd", n_warmup, n_steps, do_profile)

    return {
        'forward': fwd,
        'backward': bwd,
        'fwd_bwd': fb,
    }


def check_gradient_equivalence(
    baseline_model: torch.nn.Module,
    triton_model: torch.nn.Module,
    input_fn: Callable,
    label_fn: Optional[Callable] = None,
    fp16: bool = True,
    grad_atol_factor: float = 10.0,
) -> Dict:
    """
    Rule 1: Compare gradients of triton model vs baseline model.

    Baseline = original PyTorch (no patches).
    Triton = patched version.

    Returns:
        {
            'passed': bool,
            'max_grad_diff': float,
            'details': list[str],
        }
    """
    atol = 1e-3 if fp16 else 1e-5
    threshold = atol * grad_atol_factor

    use_labels = label_fn is not None

    # Baseline forward + backward
    baseline_model.zero_grad()
    x_base = input_fn()
    if use_labels:
        out_base = baseline_model(x_base, labels=label_fn())
    else:
        out_base = baseline_model(x_base)
    loss_base = out_base.loss if hasattr(out_base, 'loss') else out_base.sum()
    loss_base.backward()

    # Triton forward + backward
    triton_model.zero_grad()
    x_tri = input_fn()
    if use_labels:
        out_tri = triton_model(x_tri, labels=label_fn())
    else:
        out_tri = triton_model(x_tri)
    loss_tri = out_tri.loss if hasattr(out_tri, 'loss') else out_tri.sum()
    loss_tri.backward()

    details = []
    max_diff = 0.0
    all_pass = True

    for (name_base, p_base), (name_tri, p_tri) in zip(
        baseline_model.named_parameters(), triton_model.named_parameters()
    ):
        if p_base.grad is None and p_tri.grad is None:
            continue
        if p_base.grad is None or p_tri.grad is None:
            details.append(f"FAIL {name_base}: grad=None in one but not other")
            all_pass = False
            continue

        diff = (p_base.grad - p_tri.grad).abs().max().item()
        max_diff = max(max_diff, diff)

        if diff > threshold:
            details.append(f"FAIL {name_base}: grad diff={diff:.2e} > {threshold:.2e}")
            all_pass = False
        else:
            details.append(f"PASS {name_base}: grad diff={diff:.2e}")

    return {
        'passed': all_pass,
        'max_grad_diff': max_diff,
        'details': details,
    }


def check_nan_stability(
    model: torch.nn.Module,
    input_fn: Callable,
    label_fn: Optional[Callable] = None,
    n_passes: int = 2,
) -> Dict:
    """
    Rule 2: Run n_passes forward+backward, check for NaN in loss and gradients.

    Returns:
        {
            'passed': bool,
            'pass_results': list[dict],
        }
    """
    results = []
    use_labels = label_fn is not None

    for i in range(n_passes):
        model.zero_grad()
        x = input_fn()
        if use_labels:
            out = model(x, labels=label_fn())
        else:
            out = model(x)
            out = out.sum() if not hasattr(out, 'loss') else out

        loss = out.loss if hasattr(out, 'loss') else out
        has_nan_loss = loss.isnan().any().item()

        loss.backward()

        has_nan_grad = False
        nan_params = []
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.isnan().any():
                has_nan_grad = True
                nan_params.append(name)

        results.append({
            'pass': i + 1,
            'loss_value': loss.item(),
            'has_nan_loss': has_nan_loss,
            'has_nan_grad': has_nan_grad,
            'nan_params': nan_params,
        })

    all_pass = all(
        not r['has_nan_loss'] and not r['has_nan_grad']
        for r in results
    )

    return {
        'passed': all_pass,
        'pass_results': results,
    }


def print_three_phase_results(
    results: Dict,
    variant_name: str,
    baseline_results: Optional[Dict] = None,
):
    """Pretty-print three-phase benchmark results."""
    print(f"\n  {variant_name}:")
    for phase in ['forward', 'backward', 'fwd_bwd']:
        r = results[phase]
        speedup_str = ""
        if baseline_results and phase in baseline_results:
            base_ms = baseline_results[phase]['median_ms']
            if base_ms > 0:
                speedup = base_ms / r['median_ms']
                speedup_str = f"  ({speedup:.2f}x vs baseline)"
        print(f"    {phase:12s}: {r['median_ms']:.3f} ms (median){speedup_str}")
        if r.get('profiler_table'):
            # Print top 5 kernel lines from profiler
            lines = r['profiler_table'].split('\n')
            for line in lines[:8]:
                print(f"      {line}")
