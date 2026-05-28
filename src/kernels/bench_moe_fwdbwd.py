"""
MoE Layer — Full Forward + Backward Benchmark.

Measures the COMPLETE training iteration cost:
  forward pass + loss.backward() + gradient accumulation

This is what matters for training speed.

Run: .\\venv\\Scripts\\python src/kernels/bench_moe_fwdbwd.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import gc


def make_config():
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


def benchmark_fn(fn, warmup=10, rep=50):
    """Precise CUDA timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    times.sort()
    n = len(times)
    return {
        'median_ms': times[n // 2],
        'mean_ms': sum(times) / n,
        'min_ms': times[0],
        'p95_ms': times[int(0.95 * n)],
        'p5_ms': times[int(0.05 * n)],
    }


def bench_moe_layer_fwdbwd():
    """Benchmark MoE layer forward + backward in isolation."""
    print("\n" + "=" * 60)
    print("MoE LAYER: Forward + Backward (isolated)")
    print("=" * 60)
    
    from src.configuration_bibo import BiBoConfig
    from src.modeling.ffn.moe import BiBoMoELayer
    from src.kernels.moe_dispatch import patch_moe_with_triton, unpatch_moe
    
    device = 'cuda'
    config = make_config()
    
    shapes = [
        (2, 128, "2×128 (256 tokens)"),
        (4, 256, "4×256 (1024 tokens)"),
        (8, 512, "8×512 (4096 tokens)"),
    ]
    
    results = []
    
    for batch, seq, desc in shapes:
        print(f"\n  Shape: {desc}")
        
        # --- Baseline ---
        torch.manual_seed(42)
        moe = BiBoMoELayer(config).to(device).train()
        x = torch.randn(batch, seq, config.hidden_size, device=device, requires_grad=True)
        
        def run_baseline():
            # Simulate training: zero grad, forward, backward
            if x.grad is not None:
                x.grad = None
            for p in moe.parameters():
                if p.grad is not None:
                    p.grad = None
            out = moe(x)
            loss = out.sum()
            loss.backward()
        
        base_times = benchmark_fn(run_baseline, warmup=10, rep=30)
        
        # --- Triton ---
        patch_moe_with_triton(moe)
        
        def run_triton():
            if x.grad is not None:
                x.grad = None
            for p in moe.parameters():
                if p.grad is not None:
                    p.grad = None
            out = moe(x)
            loss = out.sum()
            loss.backward()
        
        tri_times = benchmark_fn(run_triton, warmup=10, rep=30)
        unpatch_moe(moe)
        
        speedup = base_times['median_ms'] / tri_times['median_ms']
        results.append((desc, base_times, tri_times, speedup))
        
        print(f"    Baseline fwd+bwd: {base_times['median_ms']:.3f} ms (p5={base_times['p5_ms']:.3f}, p95={base_times['p95_ms']:.3f})")
        print(f"    Triton fwd+bwd:   {tri_times['median_ms']:.3f} ms (p5={tri_times['p5_ms']:.3f}, p95={tri_times['p95_ms']:.3f})")
        print(f"    Speedup:          {speedup:.2f}x")
        
        del moe, x
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def bench_full_model_fwdbwd():
    """Benchmark full model forward + backward (training step)."""
    print("\n" + "=" * 60)
    print("FULL MODEL: Forward + Backward (training step)")
    print("=" * 60)
    
    from src.configuration_bibo import BiBoConfig
    from src.modeling.models import BiBoForCausalLM
    from src.kernels.patch import patch_bibo_with_triton, unpatch_bibo
    from src.kernels.moe_dispatch import patch_moe_with_triton, unpatch_moe
    
    device = 'cuda'
    config = make_config()
    
    shapes = [
        (2, 64, "2×64 (128 tokens)"),
        (2, 128, "2×128 (256 tokens)"),
        (4, 128, "4×128 (512 tokens)"),
        (4, 256, "4×256 (1024 tokens)"),
    ]
    
    results = []
    
    for batch, seq, desc in shapes:
        print(f"\n  Shape: {desc}")
        
        # --- Baseline (pure PyTorch) ---
        torch.manual_seed(42)
        model = BiBoForCausalLM(config).to(device).train()
        input_ids = torch.randint(0, 5000, (batch, seq), device=device)
        labels = input_ids.clone()
        
        def run_baseline():
            model.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
        
        base_times = benchmark_fn(run_baseline, warmup=5, rep=20)
        
        # --- Triton (RMSNorm + RoPE + MoE) ---
        patch_bibo_with_triton(model)
        patch_moe_with_triton(model)
        
        def run_triton():
            model.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
        
        tri_times = benchmark_fn(run_triton, warmup=5, rep=20)
        
        # --- Triton (only RMSNorm + RoPE, no MoE) ---
        unpatch_moe(model)
        
        def run_liger_only():
            model.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
        
        liger_times = benchmark_fn(run_liger_only, warmup=5, rep=20)
        
        unpatch_bibo(model)
        
        speedup_full = base_times['median_ms'] / tri_times['median_ms']
        speedup_liger = base_times['median_ms'] / liger_times['median_ms']
        moe_contribution = liger_times['median_ms'] / tri_times['median_ms']
        
        results.append({
            'desc': desc,
            'baseline': base_times['median_ms'],
            'liger_only': liger_times['median_ms'],
            'full_triton': tri_times['median_ms'],
            'speedup_full': speedup_full,
            'speedup_liger': speedup_liger,
            'moe_contribution': moe_contribution,
        })
        
        print(f"    Baseline (PyTorch):     {base_times['median_ms']:.3f} ms")
        print(f"    Liger only (Norm+RoPE): {liger_times['median_ms']:.3f} ms ({speedup_liger:.2f}x)")
        print(f"    Full Triton (+MoE):     {tri_times['median_ms']:.3f} ms ({speedup_full:.2f}x)")
        print(f"    MoE kernel contrib:     {moe_contribution:.2f}x additional over Liger")
        
        del model, input_ids, labels
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def bench_memory():
    """Measure peak GPU memory for baseline vs Triton."""
    print("\n" + "=" * 60)
    print("MEMORY: Peak GPU allocation")
    print("=" * 60)
    
    from src.configuration_bibo import BiBoConfig
    from src.modeling.models import BiBoForCausalLM
    from src.kernels.patch import patch_bibo_with_triton, unpatch_bibo
    from src.kernels.moe_dispatch import patch_moe_with_triton, unpatch_moe
    
    device = 'cuda'
    config = make_config()
    batch, seq = 4, 256
    
    # Baseline
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    torch.manual_seed(42)
    model = BiBoForCausalLM(config).to(device).train()
    input_ids = torch.randint(0, 5000, (batch, seq), device=device)
    labels = input_ids.clone()
    
    model.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    
    base_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    del model, out
    torch.cuda.empty_cache()
    gc.collect()
    
    # Triton
    torch.cuda.reset_peak_memory_stats()
    
    torch.manual_seed(42)
    model = BiBoForCausalLM(config).to(device).train()
    patch_bibo_with_triton(model)
    patch_moe_with_triton(model)
    
    model.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    
    tri_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    del model, out
    torch.cuda.empty_cache()
    gc.collect()
    
    savings = base_peak - tri_peak
    print(f"  Baseline peak: {base_peak:.1f} MB")
    print(f"  Triton peak:   {tri_peak:.1f} MB")
    print(f"  Savings:       {savings:.1f} MB ({savings/base_peak*100:.1f}%)")
    
    return base_peak, tri_peak


if __name__ == "__main__":
    print("=" * 60)
    print("BiBo MoE — Full Forward+Backward Benchmark")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    import triton
    print(f"Triton: {triton.__version__}")
    
    # 1. MoE layer isolated
    moe_results = bench_moe_layer_fwdbwd()
    
    # 2. Full model
    model_results = bench_full_model_fwdbwd()
    
    # 3. Memory
    base_mem, tri_mem = bench_memory()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\n  MoE Layer (fwd+bwd):")
    print(f"  {'Shape':<25} {'Baseline':>10} {'Triton':>10} {'Speedup':>8}")
    for desc, base, tri, spd in moe_results:
        print(f"  {desc:<25} {base['median_ms']:>8.3f}ms {tri['median_ms']:>8.3f}ms {spd:>6.2f}x")
    
    print("\n  Full Model (fwd+bwd):")
    print(f"  {'Shape':<25} {'Baseline':>10} {'Liger':>10} {'Full':>10} {'Speedup':>8}")
    for r in model_results:
        print(f"  {r['desc']:<25} {r['baseline']:>8.3f}ms {r['liger_only']:>8.3f}ms {r['full_triton']:>8.3f}ms {r['speedup_full']:>6.2f}x")
    
    print(f"\n  Memory: {base_mem:.0f}MB → {tri_mem:.0f}MB ({base_mem-tri_mem:.0f}MB saved)")
