"""
Full-Model Benchmark — 4 variants × 4 sizes × batch×seq sweep.

Variants:
  1. Baseline: Pure PyTorch (no patches)
  2. Liger: Liger RMSNorm + RoPE only
  3. MoE: Triton GLU activation (patched MoE forward)
  4. All: Liger + MoE (everything that helps)

Model sizes: ~3.4M, ~10.8M, ~17.5M, ~32.2M
Sweep: batch_size=[2,4,8] × seq_length=[128,256,512,1024]

Determinism: seed + CPU model copy + deep copy to GPU per config.

Run:
    .\\venv\\Scripts\\python src\\kernels\\bench\\profile_benchmark.py
    .\\venv\\Scripts\\python src\\kernels\\bench\\profile_benchmark.py --quick
"""
import copy
import os
import sys
import time
import random
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.kernels.patch import patch_bibo_with_triton
from src.kernels.dense_mlp import patch_dense_mlp_with_triton
from src.kernels.moe_dispatch import patch_moe_with_triton

# ═══════════════════════════════════════════════════════════════
# Model Configs
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    'small': dict(
        vocab_size=5000, hidden_size=256, intermediate_size=512,
        moe_intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, polyglu_expert_multiplier=1, special_expert_pairs=1,
        num_experts_per_tok=2, mlp_only_layers=[0, 3], use_ssmax=True,
        max_position_embeddings=2048,
    ),
    'medium': dict(
        vocab_size=5000, hidden_size=320, intermediate_size=1024,
        moe_intermediate_size=256, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, polyglu_expert_multiplier=2, special_expert_pairs=1,
        num_experts_per_tok=4, mlp_only_layers=[0, 1, 5], use_ssmax=True,
        max_position_embeddings=2048,
    ),
    'large': dict(
        vocab_size=5000, hidden_size=384, intermediate_size=1024,
        moe_intermediate_size=256, num_hidden_layers=8, num_attention_heads=6,
        num_key_value_heads=2, polyglu_expert_multiplier=2, special_expert_pairs=1,
        num_experts_per_tok=4, mlp_only_layers=[0, 1, 7], use_ssmax=True,
        max_position_embeddings=2048,
    ),
    'xlarge': dict(
        vocab_size=5000, hidden_size=384, intermediate_size=1536,
        moe_intermediate_size=256, num_hidden_layers=14, num_attention_heads=6,
        num_key_value_heads=2, polyglu_expert_multiplier=2, special_expert_pairs=1,
        num_experts_per_tok=4, mlp_only_layers=[0, 1, 13], use_ssmax=True,
        max_position_embeddings=2048,
    ),
}

# Sweep configs: (batch_size, seq_length)
FULL_SWEEP = [(bs, sl) for bs in [2, 4, 8] for sl in [128, 256, 512, 1024]]
QUICK_SWEEP = [(2, 128), (2, 512), (4, 256), (8, 128)]

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════
# Variants
# ═══════════════════════════════════════════════════════════════

VARIANTS = ['baseline', 'liger', 'dense_mlp', 'moe', 'all']
VARIANT_LABELS = {
    'baseline': 'Baseline',
    'liger': 'Liger (RMSNorm+RoPE)',
    'dense_mlp': 'Dense MLP (Liger SwiGLU)',
    'moe': 'MoE (Triton GLU)',
    'all': 'All (Liger+MLP+MoE)',
}


def apply_variant(model: torch.nn.Module, variant: str):
    if variant == 'liger':
        patch_bibo_with_triton(model)
    elif variant == 'dense_mlp':
        patch_dense_mlp_with_triton(model)
    elif variant == 'moe':
        patch_moe_with_triton(model)
    elif variant == 'all':
        patch_bibo_with_triton(model)
        patch_dense_mlp_with_triton(model)
        patch_moe_with_triton(model)


# ═══════════════════════════════════════════════════════════════
# Benchmark core
# ═══════════════════════════════════════════════════════════════

def bench_fb(model, input_ids, n_warmup=3, n_steps=5):
    """Benchmark forward+backward. Returns dict with timing/memory."""
    model.train()
    device = next(model.parameters()).device

    for _ in range(n_warmup):
        out = model(input_ids.to(device), labels=input_ids.to(device))
        out.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    fwd_times, bwd_times, fb_times, losses = [], [], [], []

    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(input_ids.to(device), labels=input_ids.to(device))
        loss = out.loss
        losses.append(loss.item())

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()
        model.zero_grad()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        fb_times.append((t2 - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        'fwd_ms': np.median(fwd_times),
        'bwd_ms': np.median(bwd_times),
        'fb_ms': np.median(fb_times),
        'loss': np.mean(losses),
        'peak_mem_mb': peak_mem,
        'loss_nan': any(np.isnan(l) for l in losses),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_benchmark(quick=False):
    seed = 42
    set_seed(seed)

    sweep = QUICK_SWEEP if quick else FULL_SWEEP

    print("=" * 90)
    print("  BiBo Full-Model Benchmark (No Dense MLP)")
    print("=" * 90)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Seed: {seed}")
    print(f"Variants: {', '.join(VARIANTS)}")
    print(f"Sweep: {len(sweep)} configs (batch × seq)")
    print()

    # ── Verify configs ──
    print("Model Configs:")
    print("-" * 90)
    model_configs = {}
    for name, cfg_dict in MODEL_CONFIGS.items():
        cfg = BiBoConfig(**cfg_dict)
        model = BiBoForCausalLM(cfg)
        params = count_params(model)
        del model
        model_configs[name] = cfg
        print(f"  {name:8s}: {params:>12,} params ({params/1e6:.2f}M)  "
              f"H={cfg_dict['hidden_size']} L={cfg_dict['num_hidden_layers']} "
              f"I={cfg_dict['intermediate_size']} MI={cfg_dict['moe_intermediate_size']}")
    print()

    # ── Create CPU models ──
    print("Creating CPU model copies...")
    cpu_models = {}
    for name, cfg in model_configs.items():
        set_seed(seed)
        cpu_models[name] = BiBoForCausalLM(cfg).cpu().float()
        print(f"  {name}: {count_params(cpu_models[name]):,} params")
    print()

    # ── Sweep ──
    all_results = {}  # {model: {config: {variant: result}}}

    for model_name in MODEL_CONFIGS:
        all_results[model_name] = {}
        cfg = model_configs[model_name]

        for bs, sl in sweep:
            config_key = f"bs{bs}_sl{sl}"
            all_results[model_name][config_key] = {}

            print(f"{'=' * 90}")
            print(f"  {model_name} | batch={bs} seq={sl}")
            print(f"{'=' * 90}")

            set_seed(seed + bs * 1000 + sl)
            input_ids_cpu = torch.randint(0, cfg.vocab_size, (bs, sl))

            for variant in VARIANTS:
                model_gpu = copy.deepcopy(cpu_models[model_name]).cuda()
                if variant != 'baseline':
                    apply_variant(model_gpu, variant)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                try:
                    result = bench_fb(model_gpu, input_ids_cpu, n_warmup=3, n_steps=5)
                    result['variant'] = variant
                    all_results[model_name][config_key][variant] = result

                    label = VARIANT_LABELS[variant]
                    if result['loss_nan']:
                        print(f"  {label:25s}: NaN!")
                    else:
                        print(f"  {label:25s}: fwd={result['fwd_ms']:7.2f}  bwd={result['bwd_ms']:7.2f}  "
                              f"total={result['fb_ms']:7.2f}ms  mem={result['peak_mem_mb']:7.1f}MB")
                except Exception as e:
                    print(f"  {variant:25s}: ERROR - {e}")
                    all_results[model_name][config_key][variant] = {'error': str(e)}

                del model_gpu
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Per-config comparison table
            print()
            baseline_fb = all_results[model_name][config_key].get('baseline', {}).get('fb_ms', 0)
            baseline_mem = all_results[model_name][config_key].get('baseline', {}).get('peak_mem_mb', 0)
            print(f"  {'Variant':25s} | {'Fwd':>7s} | {'Bwd':>7s} | {'Total':>7s} | {'Mem':>7s} | {'Speed':>6s} | {'MemSave':>7s}")
            print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}")
            for v in VARIANTS:
                r = all_results[model_name][config_key].get(v, {})
                if 'error' in r:
                    print(f"  {VARIANT_LABELS[v]:25s} | {'ERR':>7s}")
                else:
                    sp = baseline_fb / r['fb_ms'] if r['fb_ms'] > 0 else 0
                    ms = baseline_mem - r['peak_mem_mb']
                    print(f"  {VARIANT_LABELS[v]:25s} | {r['fwd_ms']:7.2f} | {r['bwd_ms']:7.2f} | {r['fb_ms']:7.2f} | {r['peak_mem_mb']:7.1f} | {sp:5.2f}x | {ms:+6.1f}MB")
            print()

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 90)
    print("  SPEEDUP vs BASELINE (forward+backward)")
    print("=" * 90)
    print()

    # Header
    header = f"  {'Model':8s} | {'Config':10s}"
    for v in VARIANTS[1:]:
        header += f" | {VARIANT_LABELS[v][:15]:>15s}"
    print(header)
    print(f"  {'-'*8}-+-{'-'*10}" + "".join([f"-+-{'-'*15}" for _ in VARIANTS[1:]]))

    for model_name in MODEL_CONFIGS:
        for config_key in sorted(all_results[model_name].keys()):
            row = f"  {model_name:8s} | {config_key:10s}"
            baseline_fb = all_results[model_name][config_key].get('baseline', {}).get('fb_ms', 0)
            for v in VARIANTS[1:]:
                r = all_results[model_name][config_key].get(v, {})
                if 'error' in r or baseline_fb == 0:
                    row += f" | {'N/A':>15s}"
                else:
                    speedup = baseline_fb / r['fb_ms']
                    row += f" | {speedup:14.2f}x"
            print(row)
    print()

    # Memory savings
    print("=" * 90)
    print("  MEMORY SAVINGS vs BASELINE")
    print("=" * 90)
    print()
    header = f"  {'Model':8s} | {'Config':10s}"
    for v in VARIANTS[1:]:
        header += f" | {VARIANT_LABELS[v][:15]:>15s}"
    print(header)
    print(f"  {'-'*8}-+-{'-'*10}" + "".join([f"-+-{'-'*15}" for _ in VARIANTS[1:]]))

    for model_name in MODEL_CONFIGS:
        for config_key in sorted(all_results[model_name].keys()):
            row = f"  {model_name:8s} | {config_key:10s}"
            baseline_mem = all_results[model_name][config_key].get('baseline', {}).get('peak_mem_mb', 0)
            for v in VARIANTS[1:]:
                r = all_results[model_name][config_key].get(v, {})
                if 'error' in r:
                    row += f" | {'N/A':>15s}"
                else:
                    savings = baseline_mem - r['peak_mem_mb']
                    row += f" | {savings:13.1f}MB"
            print(row)
    print()

    # Averages per model
    print("=" * 90)
    print("  AVERAGE SPEEDUP PER MODEL")
    print("=" * 90)
    print()
    for model_name in MODEL_CONFIGS:
        print(f"  {model_name}:", end="")
        for v in VARIANTS[1:]:
            speedups = []
            for config_key in all_results[model_name]:
                baseline_fb = all_results[model_name][config_key].get('baseline', {}).get('fb_ms', 0)
                r = all_results[model_name][config_key].get(v, {})
                if 'error' not in r and baseline_fb > 0:
                    speedups.append(baseline_fb / r['fb_ms'])
            if speedups:
                avg = np.mean(speedups)
                print(f"  {VARIANT_LABELS[v][:20]}={avg:.2f}x", end="")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.makedirs('./logs', exist_ok=True)
    run_benchmark(quick=args.quick)
