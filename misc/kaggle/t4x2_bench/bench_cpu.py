"""
CPU Benchmark — BiMo vs Qwen3MoE with torch.compile

Tests: compile time, warmup, avg fwd pass, avg bwd pass
Params: ~10M each
Device: CPU

Usage:
    python misc/kaggle/t4x2_bench/bench_cpu.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # fall back to eager if inductor fails (e.g. no cl.exe on Windows)

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_bibo_10m():
    """~10M param BiMo model."""
    config = BiBoConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        polyglu_expert_multiplier=2,
        special_expert_pairs=1,
        num_experts_per_tok=2,
        use_shared_expert=True,
        shared_expert_type="mlp",
        gate_type="sigmoid",
        router_type="mlp",
        load_balance_strategy="bias",
        bias_update_factor=0.001,
        use_ssmax=False,
        hidden_act="silu",
    )
    return BiBoForCausalLM(config)


def make_qwen_10m():
    """~10M param Qwen3MoE model."""
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        num_shared_experts=1,
        shared_expert_intermediate_size=512,
        decoder_sparse_step=1,
        mlp_only_layers=[0, 3],
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
    )
    return Qwen3MoeForCausalLM(config)


def benchmark_model(name, model_fn, batch_size=8, seq_len=64, num_warmup=5, num_iters=50):
    """Full benchmark: compile, warmup, fwd, bwd."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # --- Create model ---
    t0 = time.perf_counter()
    model = model_fn()
    n_params = count_params(model)
    t_create = time.perf_counter() - t0
    print(f"  Params: {n_params/1e6:.2f}M")
    print(f"  Create time: {t_create*1000:.1f}ms")

    # --- Dummy input (needed for compile dry run) ---
    x = torch.randint(0, 256, (batch_size, seq_len))
    y = torch.randint(0, 256, (batch_size, seq_len))

    # --- Compile ---
    print(f"  Compiling with torch.compile...")
    compile_ok = False
    t0 = time.perf_counter()
    try:
        compiled_model = torch.compile(model, dynamic=False)
        with torch.no_grad():
            _ = compiled_model(x[:1], labels=y[:1])
        compile_ok = True
    except Exception:
        pass
    t_compile = time.perf_counter() - t0
    
    if compile_ok:
        print(f"  Compile time: {t_compile*1000:.1f}ms")
    else:
        print(f"  Compile failed (no C++ compiler on Windows), using eager")
        compiled_model = model
        t_compile = 0

    # --- Warmup (trigger actual compilation) ---
    print(f"  Warmup: {num_warmup} iterations...")
    model.train()
    t0 = time.perf_counter()
    for i in range(num_warmup):
        out = compiled_model(x, labels=y)
        out.loss.backward()
        compiled_model.zero_grad(set_to_none=True)
    t_warmup = time.perf_counter() - t0
    print(f"  Warmup time: {t_warmup:.2f}s ({t_warmup/num_warmup*1000:.1f}ms/iter)")

    # --- Forward pass benchmark ---
    print(f"  Benchmarking fwd: {num_iters} iterations...")
    fwd_times = []
    with torch.no_grad():
        for i in range(num_iters):
            t0 = time.perf_counter()
            out = compiled_model(x, labels=y)
            t1 = time.perf_counter()
            fwd_times.append(t1 - t0)

    fwd_avg = np.mean(fwd_times) * 1000
    fwd_std = np.std(fwd_times) * 1000
    fwd_p50 = np.percentile(fwd_times, 50) * 1000
    fwd_p95 = np.percentile(fwd_times, 95) * 1000
    fwd_p99 = np.percentile(fwd_times, 99) * 1000
    print(f"  Fwd: avg={fwd_avg:.2f}ms  std={fwd_std:.2f}ms  p50={fwd_p50:.2f}ms  p95={fwd_p95:.2f}ms  p99={fwd_p99:.2f}ms")

    # --- Forward + Backward pass benchmark ---
    print(f"  Benchmarking fwd+bwd: {num_iters} iterations...")
    bwd_times = []
    for i in range(num_iters):
        t0 = time.perf_counter()
        out = compiled_model(x, labels=y)
        out.loss.backward()
        compiled_model.zero_grad(set_to_none=True)
        t1 = time.perf_counter()
        bwd_times.append(t1 - t0)

    bwd_avg = np.mean(bwd_times) * 1000
    bwd_std = np.std(bwd_times) * 1000
    bwd_p50 = np.percentile(bwd_times, 50) * 1000
    bwd_p95 = np.percentile(bwd_times, 95) * 1000
    bwd_p99 = np.percentile(bwd_times, 99) * 1000
    print(f"  Fwd+Bwd: avg={bwd_avg:.2f}ms  std={bwd_std:.2f}ms  p50={bwd_p50:.2f}ms  p95={bwd_p95:.2f}ms  p99={bwd_p99:.2f}ms")

    # --- Loss check ---
    with torch.no_grad():
        out = compiled_model(x, labels=y)
    print(f"  Loss: {out.loss.item():.4f}")

    return {
        'name': name,
        'params': n_params,
        'compile_ms': t_compile * 1000,
        'warmup_ms': t_warmup / num_warmup * 1000,
        'fwd_avg_ms': fwd_avg,
        'fwd_std_ms': fwd_std,
        'fwd_p50_ms': fwd_p50,
        'fwd_p95_ms': fwd_p95,
        'bwd_avg_ms': bwd_avg,
        'bwd_std_ms': bwd_std,
        'bwd_p50_ms': bwd_p50,
        'bwd_p95_ms': bwd_p95,
        'loss': out.loss.item(),
    }


def main():
    print("=" * 60)
    print("  CPU Benchmark — BiMo vs Qwen3MoE")
    print("  torch.compile | ~10M params | CPU only")
    print("=" * 60)

    batch_size = 8
    seq_len = 64
    num_warmup = 5
    num_iters = 50

    print(f"  batch_size={batch_size}  seq_len={seq_len}")
    print(f"  warmup={num_warmup}  iters={num_iters}")

    r_bibo = benchmark_model("BiMo (BiBo MoE)", make_bibo_10m, batch_size, seq_len, num_warmup, num_iters)
    r_qwen = benchmark_model("Qwen3MoE", make_qwen_10m, batch_size, seq_len, num_warmup, num_iters)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Params':>8} {'Compile':>10} {'Fwd':>10} {'Fwd+Bwd':>10} {'Loss':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in [r_bibo, r_qwen]:
        print(f"  {r['name']:<20} {r['params']/1e6:>7.2f}M {r['compile_ms']:>9.1f}ms {r['fwd_avg_ms']:>9.2f}ms {r['bwd_avg_ms']:>9.2f}ms {r['loss']:>7.4f}")

    # Speedup
    if r_bibo['fwd_avg_ms'] > 0 and r_qwen['fwd_avg_ms'] > 0:
        fwd_ratio = r_qwen['fwd_avg_ms'] / r_bibo['fwd_avg_ms']
        bwd_ratio = r_qwen['bwd_avg_ms'] / r_bibo['bwd_avg_ms']
        print(f"\n  Fwd speedup (Qwen/BiMo): {fwd_ratio:.2f}x")
        print(f"  Fwd+Bwd speedup (Qwen/BiMo): {bwd_ratio:.2f}x")


if __name__ == '__main__':
    main()
