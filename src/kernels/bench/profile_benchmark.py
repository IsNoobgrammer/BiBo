"""
Comprehensive profiling test suite for BiBo kernels.

Compares baseline vs Liger-only vs Liger+Triton kernels with deterministic profiling.
"""
import copy
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.kernels.patch import patch_bibo_with_triton, unpatch_bibo
from src.kernels.dense_mlp import patch_dense_mlp_with_triton
from src.kernels.moe_dispatch import patch_moe_with_triton


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_10M_model() -> BiBoForCausalLM:
    """Create a BiBo model with ~10M parameters."""
    config = BiBoConfig(
        vocab_size=5000,
        hidden_size=384,  # Reduced from 512 to get closer to 10M
        intermediate_size=1536,  # Reduced from 2048
        num_hidden_layers=4,
        num_attention_heads=6,  # Reduced from 8
        num_key_value_heads=2,
        max_position_embeddings=2048,
        polyglu_expert_multiplier=1,  # 1 group = 3 GLU experts
        special_expert_pairs=1,  # 1 pair = (Identity, Zero) = 2 experts
        num_experts_per_tok=4,  # Top-4 routing
        mlp_only_layers=[0, 3],  # First and last layers are dense MLP
        use_ssmax=True,
    )
    model = BiBoForCausalLM(config)
    
    # Verify parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {param_count:,} (~{param_count / 1e6:.2f}M)")
    
    if param_count > 12_000_000 or param_count < 8_000_000:
        print(f"Warning: Model parameter count {param_count:,} is outside target range (8-12M)")
    
    return model


def generate_input(batch_size: int = 4, seq_length: int = 2048, vocab_size: int = 5000) -> torch.Tensor:
    """Generate random input for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_length))


def profile_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    config_name: str,
    log_dir: str,
) -> Dict:
    """Profile a model with torch.profiler."""
    model.eval()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(5):  # wait + warmup + active
            with torch.no_grad():
                output = model(input_ids)
            prof.step()
    
    # Extract metrics
    key_averages = prof.key_averages()
    
    # Time metrics - use CPU time as fallback, try CUDA time if available
    total_cpu_time_ms = sum(evt.cpu_time_total for evt in key_averages if evt.cpu_time_total > 0)
    avg_time_ms = total_cpu_time_ms / 3  # Average over active steps
    
    # Memory metrics
    memory_events = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
    
    # Kernel counts
    cuda_kernels = sum(1 for evt in key_averages if evt.key.startswith('cuda'))
    cpu_ops = sum(1 for evt in key_averages if not evt.key.startswith('cuda'))
    
    # Top operators by time
    top_ops_time = sorted(
        [{'name': evt.key, 'time_ms': evt.cpu_time_total / 1000, 'pct': (evt.cpu_time_total / total_cpu_time_ms * 100) if total_cpu_time_ms > 0 else 0}
         for evt in key_averages if evt.cpu_time_total > 0],
        key=lambda x: x['time_ms'],
        reverse=True
    )[:10]
    
    # Top operators by memory
    top_ops_memory = sorted(
        [{'name': evt.key, 'memory_mb': evt.self_cuda_memory_usage / 1024 / 1024 if hasattr(evt, 'self_cuda_memory_usage') else 0}
         for evt in key_averages if hasattr(evt, 'self_cuda_memory_usage') and evt.self_cuda_memory_usage > 0],
        key=lambda x: x['memory_mb'],
        reverse=True
    )[:10]
    
    # Peak memory
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'avg_time_ms': avg_time_ms,
        'total_time_ms': total_cpu_time_ms,
        'peak_memory_mb': peak_memory_mb,
        'cuda_kernel_count': cuda_kernels,
        'cpu_operator_count': cpu_ops,
        'top_ops_time': top_ops_time,
        'top_ops_memory': top_ops_memory,
    }


def run_parameter_sweep(
    batch_sizes: List[int] = [1, 2, 4],
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    seeds: List[int] = [42, 123, 999],
):
    """Run parameter sweep across batch sizes and sequence lengths."""
    vocab_size = 5000
    
    # Results structure: {batch_size: {seq_length: {seed: {config: results}}}}
    all_results = {
        bs: {
            sl: {seed: {} for seed in seeds}
            for sl in seq_lengths
        }
        for bs in batch_sizes
    }
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            print(f"\n{'=' * 80}")
            print(f"BATCH SIZE: {batch_size}, SEQUENCE LENGTH: {seq_length}")
            print(f"{'=' * 80}")
            
            for seed in seeds:
                print(f"\n{'=' * 80}")
                print(f"SEED: {seed}")
                print(f"{'=' * 80}")
                
                set_seed(seed)
                
                # Initialize model on CPU
                model_cpu = create_10M_model()
                input_ids_cpu = generate_input(batch_size, seq_length, vocab_size)
                
                # Baseline Test (No Liger, No Triton)
                print("\n[1/3] Running baseline test (no optimizations)...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                model_baseline = copy.deepcopy(model_cpu).to('cuda')
                input_ids_baseline = input_ids_cpu.to('cuda')
                
                baseline_results = profile_model(
                    model_baseline,
                    input_ids_baseline,
                    'baseline',
                    f'./logs/baseline_bs{batch_size}_sl{seq_length}_seed{seed}'
                )
                all_results[batch_size][seq_length][seed]['baseline'] = baseline_results
                
                print(f"  Avg Time: {baseline_results['avg_time_ms']:.2f} ms")
                print(f"  Peak Memory: {baseline_results['peak_memory_mb']:.2f} MB")
                print(f"  CUDA Kernels: {baseline_results['cuda_kernel_count']}")
                
                # Reset CUDA
                del model_baseline, input_ids_baseline
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Liger-Only Test (RMSNorm, RoPE)
                print("\n[2/3] Running Liger-only test (RMSNorm, RoPE)...")
                model_liger = copy.deepcopy(model_cpu).to('cuda')
                input_ids_liger = input_ids_cpu.to('cuda')
                
                patch_bibo_with_triton(model_liger)
                
                liger_results = profile_model(
                    model_liger,
                    input_ids_liger,
                    'liger',
                    f'./logs/liger_bs{batch_size}_sl{seq_length}_seed{seed}'
                )
                all_results[batch_size][seq_length][seed]['liger'] = liger_results
                
                print(f"  Avg Time: {liger_results['avg_time_ms']:.2f} ms")
                print(f"  Peak Memory: {liger_results['peak_memory_mb']:.2f} MB")
                print(f"  CUDA Kernels: {liger_results['cuda_kernel_count']}")
                
                # Reset CUDA
                del model_liger, input_ids_liger
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Liger+Triton Test (All kernels)
                print("\n[3/3] Running Liger+Triton test (all kernels)...")
                model_full = copy.deepcopy(model_cpu).to('cuda')
                input_ids_full = input_ids_cpu.to('cuda')
                
                patch_bibo_with_triton(model_full)
                patch_moe_with_triton(model_full)
                patch_dense_mlp_with_triton(model_full)
                
                full_results = profile_model(
                    model_full,
                    input_ids_full,
                    'full',
                    f'./logs/full_bs{batch_size}_sl{seq_length}_seed{seed}'
                )
                all_results[batch_size][seq_length][seed]['full'] = full_results
                
                print(f"  Avg Time: {full_results['avg_time_ms']:.2f} ms")
                print(f"  Peak Memory: {full_results['peak_memory_mb']:.2f} MB")
                print(f"  CUDA Kernels: {full_results['cuda_kernel_count']}")
                
                # Reset CUDA
                del model_full, input_ids_full
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Cleanup CPU model
                del model_cpu, input_ids_cpu
    
    # Generate comprehensive sweep report
    print(f"\n{'=' * 80}")
    print("PARAMETER SWEEP SUMMARY")
    print(f"{'=' * 80}")
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            print(f"\n{'=' * 80}")
            print(f"BATCH SIZE: {batch_size}, SEQUENCE LENGTH: {seq_length}")
            print(f"{'=' * 80}")
            
            for config_name in ['baseline', 'liger', 'full']:
                times = [all_results[batch_size][seq_length][seed][config_name]['avg_time_ms'] for seed in seeds]
                memories = [all_results[batch_size][seq_length][seed][config_name]['peak_memory_mb'] for seed in seeds]
                
                time_mean, time_std = np.mean(times), np.std(times)
                memory_mean, memory_std = np.mean(memories), np.std(memories)
                
                print(f"\n{config_name.upper()}:")
                print(f"  Time: {time_mean:.2f} ± {time_std:.2f} ms (CV: {time_std/time_mean*100:.1f}%)")
                print(f"  Memory: {memory_mean:.2f} ± {memory_std:.2f} MB (CV: {memory_std/memory_mean*100:.1f}%)")
            
            # Comparison summary for this configuration
            baseline_time = np.mean([all_results[batch_size][seq_length][seed]['baseline']['avg_time_ms'] for seed in seeds])
            liger_time = np.mean([all_results[batch_size][seq_length][seed]['liger']['avg_time_ms'] for seed in seeds])
            full_time = np.mean([all_results[batch_size][seq_length][seed]['full']['avg_time_ms'] for seed in seeds])
            
            baseline_memory = np.mean([all_results[batch_size][seq_length][seed]['baseline']['peak_memory_mb'] for seed in seeds])
            liger_memory = np.mean([all_results[batch_size][seq_length][seed]['liger']['peak_memory_mb'] for seed in seeds])
            full_memory = np.mean([all_results[batch_size][seq_length][seed]['full']['peak_memory_mb'] for seed in seeds])
            
            print(f"\nSpeedup (Liger vs Baseline): {baseline_time/liger_time:.2f}x")
            print(f"Speedup (Full vs Baseline): {baseline_time/full_time:.2f}x")
            print(f"Memory Savings (Liger vs Baseline): {baseline_memory - liger_memory:.2f} MB")
            print(f"Memory Savings (Full vs Baseline): {baseline_memory - full_memory:.2f} MB")


def run_profiling_suite(seeds: List[int] = [42, 123, 999]):
    """Run comprehensive profiling suite with multiple seeds."""
    batch_size = 4
    seq_length = 2048
    vocab_size = 5000
    
    results = {seed: {} for seed in seeds}
    
    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"SEED: {seed}")
        print(f"{'=' * 80}")
        
        set_seed(seed)
        
        # Initialize model on CPU
        model_cpu = create_10M_model()
        input_ids_cpu = generate_input(batch_size, seq_length, vocab_size)
        
        # Baseline Test (No Liger, No Triton)
        print("\n[1/3] Running baseline test (no optimizations)...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model_baseline = copy.deepcopy(model_cpu).to('cuda')
        input_ids_baseline = input_ids_cpu.to('cuda')
        
        baseline_results = profile_model(
            model_baseline,
            input_ids_baseline,
            'baseline',
            f'./logs/baseline_seed_{seed}'
        )
        results[seed]['baseline'] = baseline_results
        
        print(f"  Avg Time: {baseline_results['avg_time_ms']:.2f} ms")
        print(f"  Peak Memory: {baseline_results['peak_memory_mb']:.2f} MB")
        print(f"  CUDA Kernels: {baseline_results['cuda_kernel_count']}")
        
        # Reset CUDA
        del model_baseline, input_ids_baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Liger-Only Test (RMSNorm, RoPE)
        print("\n[2/3] Running Liger-only test (RMSNorm, RoPE)...")
        model_liger = copy.deepcopy(model_cpu).to('cuda')
        input_ids_liger = input_ids_cpu.to('cuda')
        
        patch_bibo_with_triton(model_liger)
        
        liger_results = profile_model(
            model_liger,
            input_ids_liger,
            'liger',
            f'./logs/liger_seed_{seed}'
        )
        results[seed]['liger'] = liger_results
        
        print(f"  Avg Time: {liger_results['avg_time_ms']:.2f} ms")
        print(f"  Peak Memory: {liger_results['peak_memory_mb']:.2f} MB")
        print(f"  CUDA Kernels: {liger_results['cuda_kernel_count']}")
        
        # Reset CUDA
        del model_liger, input_ids_liger
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Liger+Triton Test (All kernels)
        print("\n[3/3] Running Liger+Triton test (all kernels)...")
        model_full = copy.deepcopy(model_cpu).to('cuda')
        input_ids_full = input_ids_cpu.to('cuda')
        
        patch_bibo_with_triton(model_full)
        patch_moe_with_triton(model_full)
        patch_dense_mlp_with_triton(model_full)
        
        full_results = profile_model(
            model_full,
            input_ids_full,
            'full',
            f'./logs/full_seed_{seed}'
        )
        results[seed]['full'] = full_results
        
        print(f"  Avg Time: {full_results['avg_time_ms']:.2f} ms")
        print(f"  Peak Memory: {full_results['peak_memory_mb']:.2f} MB")
        print(f"  CUDA Kernels: {full_results['cuda_kernel_count']}")
        
        # Reset CUDA
        del model_full, input_ids_full
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Cleanup CPU model
        del model_cpu, input_ids_cpu
    
    # Generate comprehensive report
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE REPORT")
    print(f"{'=' * 80}")
    
    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"SEED: {seed}")
        print(f"{'=' * 80}")
        
        for config_name in ['baseline', 'liger', 'full']:
            prof_data = results[seed][config_name]
            print(f"\n{config_name.upper()}")
            print("-" * 80)
            print(f"  Avg Time: {prof_data['avg_time_ms']:.2f} ms")
            print(f"  Peak Memory: {prof_data['peak_memory_mb']:.2f} MB")
            print(f"  CUDA Kernels: {prof_data['cuda_kernel_count']}")
            print(f"  CPU Operators: {prof_data['cpu_operator_count']}")
            print(f"\n  Top 10 Operators (by time):")
            for op in prof_data['top_ops_time'][:10]:
                print(f"    - {op['name']}: {op['time_ms']:.2f} ms ({op['pct']:.1f}%)")
            print(f"\n  Top 10 Operators (by memory):")
            for op in prof_data['top_ops_memory'][:10]:
                print(f"    - {op['name']}: {op['memory_mb']:.2f} MB")
    
    # Variance analysis
    print(f"\n{'=' * 80}")
    print("VARIANCE ANALYSIS (across 3 seeds)")
    print(f"{'=' * 80}")
    
    for config_name in ['baseline', 'liger', 'full']:
        times = [results[seed][config_name]['avg_time_ms'] for seed in seeds]
        memories = [results[seed][config_name]['peak_memory_mb'] for seed in seeds]
        kernels = [results[seed][config_name]['cuda_kernel_count'] for seed in seeds]
        
        time_mean, time_std = np.mean(times), np.std(times)
        memory_mean, memory_std = np.mean(memories), np.std(memories)
        kernel_mean, kernel_std = np.mean(kernels), np.std(kernels)
        
        print(f"\n{config_name.upper()}:")
        print(f"  Time: {time_mean:.2f} ± {time_std:.2f} ms (CV: {time_std/time_mean*100:.1f}%)")
        print(f"  Memory: {memory_mean:.2f} ± {memory_std:.2f} MB (CV: {memory_std/memory_mean*100:.1f}%)")
        print(f"  Kernels: {kernel_mean:.1f} ± {kernel_std:.1f} (CV: {kernel_std/kernel_mean*100:.1f}%)")
    
    # Comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    
    baseline_time = np.mean([results[seed]['baseline']['avg_time_ms'] for seed in seeds])
    liger_time = np.mean([results[seed]['liger']['avg_time_ms'] for seed in seeds])
    full_time = np.mean([results[seed]['full']['avg_time_ms'] for seed in seeds])
    
    baseline_memory = np.mean([results[seed]['baseline']['peak_memory_mb'] for seed in seeds])
    liger_memory = np.mean([results[seed]['liger']['peak_memory_mb'] for seed in seeds])
    full_memory = np.mean([results[seed]['full']['peak_memory_mb'] for seed in seeds])
    
    baseline_kernels = np.mean([results[seed]['baseline']['cuda_kernel_count'] for seed in seeds])
    liger_kernels = np.mean([results[seed]['liger']['cuda_kernel_count'] for seed in seeds])
    full_kernels = np.mean([results[seed]['full']['cuda_kernel_count'] for seed in seeds])
    
    print(f"\nSpeedup (Liger vs Baseline): {baseline_time/liger_time:.2f}x")
    print(f"Speedup (Full vs Baseline): {baseline_time/full_time:.2f}x")
    print(f"Memory Savings (Liger vs Baseline): {baseline_memory - liger_memory:.2f} MB")
    print(f"Memory Savings (Full vs Baseline): {baseline_memory - full_memory:.2f} MB")
    print(f"Kernel Count Reduction (Liger vs Baseline): {(1 - liger_kernels/baseline_kernels)*100:.1f}%")
    print(f"Kernel Count Reduction (Full vs Baseline): {(1 - full_kernels/baseline_kernels)*100:.1f}%")


if __name__ == "__main__":
    # Create logs directory
    os.makedirs('./logs', exist_ok=True)
    
    # Run parameter sweep
    run_parameter_sweep()
