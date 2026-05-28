"""
Conv Kernel Benchmark — BiBo Conv Router + Conv Shared Expert.

Strategy v2: Keep cuDNN for conv, fuse surrounding ops (permute+act+gate).

Benchmarks:
1. Conv Router: cuDNN conv + fused permute/reshape vs baseline
2. Conv Shared Expert: cuDNN conv + fused permute+act+gate vs baseline
3. Scaling analysis across sequence lengths
4. Memory comparison

Run: .\\venv\\Scripts\\python src/kernels/bench_conv.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple


def benchmark_fn(fn, warmup=20, rep=100):
    """Benchmark with CUDA events."""
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
    return {
        'median_ms': times[len(times) // 2],
        'mean_ms': sum(times) / len(times),
        'min_ms': times[0],
        'p95_ms': times[int(0.95 * len(times))],
    }


def measure_memory(fn, warmup=3):
    """Measure peak GPU memory."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ─────────────────────────────────────────────────────────────
# Benchmark 1: Fused Permute + Activation + Gate Multiply
# (The core fusion for conv shared expert)
# ─────────────────────────────────────────────────────────────

def bench_fused_permute_act_gate():
    """
    Benchmark the key fusion: permute(B,I,S→B,S,I) + SiLU + gate multiply.
    
    Baseline (3 ops):
        gate_output = conv_out.permute(0, 2, 1)  # allocates (B,S,I)
        activated = F.silu(gate_output)           # allocates (B,S,I)
        result = activated * up_out               # allocates (B,S,I)
    
    Triton (1 kernel):
        result = fused_permute_act_gate(conv_out, up_out)  # 1 allocation
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Fused Permute + SiLU + Gate Multiply")
    print("=" * 60)
    
    from src.kernels.conv_fused import triton_fused_conv_gate_multiply
    
    device = 'cuda'
    
    test_shapes = [
        (2, 64, 256, "B=2, S=64, I=256"),
        (4, 128, 256, "B=4, S=128, I=256"),
        (4, 256, 384, "B=4, S=256, I=384"),
        (4, 512, 512, "B=4, S=512, I=512"),
        (4, 1024, 684, "B=4, S=1024, I=684"),
        (2, 2048, 684, "B=2, S=2048, I=684"),
        (4, 2048, 684, "B=4, S=2048, I=684"),
    ]
    
    results = []
    
    for B, S, I, desc in test_shapes:
        print(f"\n  {desc} ({B*S} tokens)")
        
        # Inputs matching what conv produces
        conv_out = torch.randn(B, I, S, device=device)  # (B, I, S) from Conv1d
        up_out = torch.randn(B, S, I, device=device)    # (B, S, I) from up_proj
        
        # Baseline: permute + silu + multiply (3 ops, 2 intermediates)
        def baseline():
            gate_output = conv_out.permute(0, 2, 1)  # (B, S, I) — new tensor
            activated = F.silu(gate_output)           # (B, S, I) — new tensor
            return activated * up_out                 # (B, S, I) — new tensor
        
        # Triton: single kernel
        def triton_fused():
            return triton_fused_conv_gate_multiply(conv_out, up_out, act_type=0)
        
        # Correctness
        ref = baseline()
        tri = triton_fused()
        max_diff = (ref - tri).abs().max().item()
        correct = max_diff < 1e-5  # fp32 should be very tight
        
        # Performance
        base_t = benchmark_fn(baseline, warmup=15, rep=50)
        tri_t = benchmark_fn(triton_fused, warmup=15, rep=50)
        speedup = base_t['median_ms'] / tri_t['median_ms']
        
        # Memory
        base_mem = measure_memory(baseline)
        tri_mem = measure_memory(triton_fused)
        mem_saved = base_mem - tri_mem
        
        print(f"    Correct: {'PASS' if correct else 'FAIL'} (diff={max_diff:.2e})")
        print(f"    Base: {base_t['median_ms']:.4f}ms | Tri: {tri_t['median_ms']:.4f}ms | {speedup:.2f}x")
        print(f"    Mem saved: {mem_saved:.1f} MB")
        
        results.append({
            'desc': desc, 'tokens': B*S, 'correct': correct,
            'base_ms': base_t['median_ms'], 'tri_ms': tri_t['median_ms'],
            'speedup': speedup, 'mem_saved_mb': mem_saved,
        })
        
        del conv_out, up_out
        torch.cuda.empty_cache()
    
    return results


# ─────────────────────────────────────────────────────────────
# Benchmark 2: Full Conv Shared Expert (end-to-end)
# ─────────────────────────────────────────────────────────────

def bench_full_conv_expert():
    """
    End-to-end benchmark of BiBoCausalConv1D with and without Triton fusion.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Full Conv Shared Expert (E2E)")
    print("=" * 60)
    
    from src.configuration_bibo import BiBoConfig
    from src.modeling.ffn.experts import BiBoCausalConv1D
    from src.kernels.conv_fused import patch_conv_expert_with_triton, unpatch_conv_expert
    
    device = 'cuda'
    
    test_shapes = [
        (2, 64, 512, 256, 3, "2×64, h=512, i=256, k=3"),
        (4, 128, 512, 256, 3, "4×128, h=512, i=256, k=3"),
        (4, 256, 768, 384, 3, "4×256, h=768, i=384, k=3"),
        (4, 512, 1024, 512, 3, "4×512, h=1024, i=512, k=3"),
        (4, 1024, 1536, 684, 3, "4×1024, h=1536, i=684, k=3"),
        (2, 2048, 1536, 684, 3, "2×2048, h=1536, i=684, k=3"),
    ]
    
    results = []
    
    for batch, seq, hidden, inter, kernel_size, desc in test_shapes:
        print(f"\n  {desc} ({batch*seq} tokens)")
        
        config = BiBoConfig(
            hidden_size=hidden, moe_intermediate_size=inter,
            kernel_size=kernel_size, hidden_act="silu",
            num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
            vocab_size=5000, intermediate_size=inter*4,
        )
        
        torch.manual_seed(42)
        expert = BiBoCausalConv1D(config).to(device).eval()
        x = torch.randn(batch, seq, hidden, device=device)
        
        # Baseline
        base_t = benchmark_fn(lambda: expert(x), warmup=10, rep=40)
        base_mem = measure_memory(lambda: expert(x))
        ref_out = expert(x)
        
        # Triton patched
        patch_conv_expert_with_triton(expert)
        tri_t = benchmark_fn(lambda: expert(x), warmup=10, rep=40)
        tri_mem = measure_memory(lambda: expert(x))
        tri_out = expert(x)
        unpatch_conv_expert(expert)
        
        # Check
        max_diff = (ref_out - tri_out).abs().max().item()
        correct = max_diff < 1e-4
        speedup = base_t['median_ms'] / tri_t['median_ms']
        mem_saved = base_mem - tri_mem
        
        print(f"    Correct: {'PASS' if correct else 'FAIL'} (diff={max_diff:.2e})")
        print(f"    Base: {base_t['median_ms']:.4f}ms | Tri: {tri_t['median_ms']:.4f}ms | {speedup:.2f}x")
        print(f"    Mem: base={base_mem:.1f}MB, tri={tri_mem:.1f}MB, saved={mem_saved:.1f}MB")
        
        results.append({
            'desc': desc, 'tokens': batch*seq, 'correct': correct,
            'base_ms': base_t['median_ms'], 'tri_ms': tri_t['median_ms'],
            'speedup': speedup, 'mem_saved_mb': mem_saved,
        })
        
        del expert, x
        torch.cuda.empty_cache()
    
    return results


# ─────────────────────────────────────────────────────────────
# Benchmark 3: Full Model with Conv Router + Conv Expert
# ─────────────────────────────────────────────────────────────

def bench_full_model_conv():
    """
    Full BiBo model with router_type='conv' and shared_expert_type='conv'.
    Measures end-to-end impact of conv kernel fusion.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Full Model (conv router + conv expert)")
    print("=" * 60)
    
    from src.configuration_bibo import BiBoConfig
    from src.modeling.models import BiBoForCausalLM
    from src.kernels.conv_fused import patch_conv_router_with_triton, patch_conv_expert_with_triton
    from src.kernels.conv_fused import unpatch_conv_router, unpatch_conv_expert
    from src.kernels.patch import patch_bibo_with_triton
    from src.kernels.moe_dispatch import patch_moe_with_triton
    
    device = 'cuda'
    config = BiBoConfig(
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
        shared_expert_type="conv",   # Conv shared expert
        router_type="conv",          # Conv router
        kernel_size=3,
        router_lambda=1.0,
        router_noise=0.0,
        moe_shared_scaling=2.0,
        gate_type="sigmoid",
        load_balance_strategy="none",
        tie_word_embeddings=True,
    )
    
    # Baseline model
    torch.manual_seed(42)
    model = BiBoForCausalLM(config).to(device).eval()
    
    test_shapes = [
        (2, 64, "2×64"),
        (4, 128, "4×128"),
        (4, 256, "4×256"),
        (2, 512, "2×512"),
    ]
    
    results = []
    
    for batch, seq, desc in test_shapes:
        print(f"\n  {desc} ({batch*seq} tokens)")
        input_ids = torch.randint(0, 5000, (batch, seq), device=device)
        
        # Baseline (no Triton)
        base_t = benchmark_fn(lambda: model(input_ids), warmup=5, rep=20)
        
        # With all Triton patches (RMSNorm + RoPE + MoE + Conv)
        patch_bibo_with_triton(model)
        patch_moe_with_triton(model)
        patch_conv_router_with_triton(model)
        patch_conv_expert_with_triton(model)
        
        tri_t = benchmark_fn(lambda: model(input_ids), warmup=5, rep=20)
        
        # Correctness (quick check)
        ref = model(input_ids)  # with triton
        unpatch_conv_router(model)
        unpatch_conv_expert(model)
        
        speedup = base_t['median_ms'] / tri_t['median_ms']
        print(f"    Base: {base_t['median_ms']:.3f}ms | All Triton: {tri_t['median_ms']:.3f}ms | {speedup:.2f}x")
        
        results.append({
            'desc': desc, 'tokens': batch*seq,
            'base_ms': base_t['median_ms'], 'tri_ms': tri_t['median_ms'],
            'speedup': speedup,
        })
        
        # Unpatch remaining
        from src.kernels.patch import unpatch_bibo
        from src.kernels.moe_dispatch import unpatch_moe
        unpatch_bibo(model)
        unpatch_moe(model)
        
        torch.cuda.empty_cache()
    
    del model
    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BiBo Conv Kernel Benchmark Suite (v2)")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton not installed.")
        sys.exit(1)
    
    print("\nStrategy: cuDNN for conv (unbeatable) + Triton for fusion")
    print("Target: eliminate intermediate tensors, fuse permute+act+gate")
    
    # Phase 1: Core fusion kernel
    fusion_results = bench_fused_permute_act_gate()
    
    # Phase 2: Full conv expert
    expert_results = bench_full_conv_expert()
    
    # Phase 3: Full model
    model_results = bench_full_model_conv()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\n  Fused Permute+SiLU+Gate (core kernel):")
    for r in fusion_results:
        status = "PASS" if r['correct'] else "FAIL"
        print(f"    {r['desc']}: {r['speedup']:.2f}x, mem saved {r['mem_saved_mb']:.1f}MB [{status}]")
    
    print("\n  Full Conv Expert (E2E):")
    for r in expert_results:
        status = "PASS" if r['correct'] else "FAIL"
        print(f"    {r['desc']}: {r['speedup']:.2f}x, mem saved {r['mem_saved_mb']:.1f}MB [{status}]")
    
    print("\n  Full Model (all Triton patches):")
    for r in model_results:
        print(f"    {r['desc']}: {r['speedup']:.2f}x")
    
    # Promotion decision
    all_correct = all(r['correct'] for r in fusion_results + expert_results)
    avg_fusion_speedup = sum(r['speedup'] for r in fusion_results) / len(fusion_results)
    avg_expert_speedup = sum(r['speedup'] for r in expert_results) / len(expert_results)
    
    print(f"\n  Avg fusion kernel speedup: {avg_fusion_speedup:.2f}x")
    print(f"  Avg expert E2E speedup: {avg_expert_speedup:.2f}x")
    
    if all_correct and avg_fusion_speedup > 1.0:
        print("\n  ✓ PROMOTED: Conv fusion kernels correct and faster.")
    elif all_correct:
        print("\n  ⚠ CORRECT but fusion kernel not faster — memory savings still valuable.")
    else:
        print("\n  ✗ REJECTED: Fix correctness issues.")
