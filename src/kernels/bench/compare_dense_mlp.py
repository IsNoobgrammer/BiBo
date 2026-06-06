"""
Comparison test for dense MLP kernel configurations.

Compares 4 configurations:
1. Baseline: PyTorch eager (no fusion)
2. Forward-only: Triton forward, PyTorch backward (current dense_mlp.py)
3. Forward+Backward: Separate Triton kernels (dense_mlp_fused.py)
4. Fully Fused: Single Triton kernel for both (dense_mlp_fused.py - research)
"""
import copy
import os
import sys
import time
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.kernels.dense_mlp import patch_dense_mlp_with_triton, unpatch_dense_mlp
from src.kernels.dense_mlp_fused import (
    patch_dense_mlp_separate_backward,
    patch_dense_mlp_full_fused,
    unpatch_dense_mlp_fused,
)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model() -> BiBoForCausalLM:
    """Create a BiBo model with dense MLP layers (smaller config)."""
    config = BiBoConfig(
        vocab_size=5000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=2,  # Reduced from 4 to 2
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        polyglu_expert_multiplier=1,
        special_expert_pairs=1,
        num_experts_per_tok=4,
        mlp_only_layers=[0, 1],  # Both layers are dense MLP
        use_ssmax=True,
    )
    model = BiBoForCausalLM(config)
    return model


def benchmark_forward_backward(model, input_ids, labels, num_iterations=10, warmup=3):
    """Benchmark forward and backward passes."""
    model.train()
    
    # Warmup
    for _ in range(warmup):
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    
    # Benchmark
    forward_times = []
    backward_times = []
    total_times = []
    
    for _ in range(num_iterations):
        # Forward
        torch.cuda.synchronize()
        start_fwd = time.perf_counter()
        output = model(input_ids, labels=labels)
        torch.cuda.synchronize()
        end_fwd = time.perf_counter()
        forward_times.append(end_fwd - start_fwd)
        
        # Backward
        torch.cuda.synchronize()
        start_bwd = time.perf_counter()
        output.loss.backward()
        torch.cuda.synchronize()
        end_bwd = time.perf_counter()
        backward_times.append(end_bwd - start_bwd)
        
        total_times.append(forward_times[-1] + backward_times[-1])
        
        model.zero_grad()
    
    return {
        'forward_ms': sum(forward_times) / len(forward_times) * 1000,
        'backward_ms': sum(backward_times) / len(backward_times) * 1000,
        'total_ms': sum(total_times) / len(total_times) * 1000,
    }


def verify_gradients(model_baseline, model_test, input_ids, labels):
    """Verify gradient equivalence between baseline and test model."""
    model_baseline.train()
    model_test.train()
    
    # Baseline forward + backward
    output_baseline = model_baseline(input_ids, labels=labels)
    loss_baseline = output_baseline.loss
    loss_baseline.backward()
    
    # Test forward + backward
    output_test = model_test(input_ids, labels=labels)
    loss_test = output_test.loss
    loss_test.backward()
    
    # Check loss difference
    loss_diff = abs(loss_baseline.item() - loss_test.item())
    print(f"  Loss difference: {loss_diff:.6f}")
    
    # Check gradient equivalence
    max_grad_diff = 0.0
    zero_grad_count = 0
    total_params = 0
    
    for (n1, p1), (n2, p2) in zip(
        model_baseline.named_parameters(),
        model_test.named_parameters()
    ):
        if p1.requires_grad and p2.requires_grad:
            total_params += 1
            if p1.grad is None or p2.grad is None:
                print(f"  WARNING: {n1} has None gradient")
                zero_grad_count += 1
                continue
            
            if p1.grad.norm() == 0 or p2.grad.norm() == 0:
                print(f"  WARNING: {n1} has zero gradient")
                zero_grad_count += 1
                continue
            
            diff = (p1.grad - p2.grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
    
    print(f"  Max gradient difference: {max_grad_diff:.6f}")
    print(f"  Parameters with zero/None gradients: {zero_grad_count}/{total_params}")
    
    # Tolerance: loss < 1e-2, grad diff < 5e-2 (fp16), no zero/None grads
    return loss_diff < 1e-2 and max_grad_diff < 5e-2 and zero_grad_count == 0


def run_comparison(seed=42):
    """Run comparison across all 4 configurations."""
    set_seed(seed)
    
    # Create models
    print("Creating models...")
    model_baseline_cpu = create_model()
    param_count = sum(p.numel() for p in model_baseline_cpu.parameters())
    print(f"Model parameter count: {param_count:,} (~{param_count / 1e6:.2f}M)")
    
    # Generate input
    batch_size = 4
    seq_length = 2048
    vocab_size = 5000
    input_ids_cpu = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels_cpu = input_ids_cpu.clone()
    
    # Test configurations
    configs = [
        ('baseline', 'PyTorch eager (no fusion)'),
        ('forward_only', 'Triton forward, PyTorch backward'),
        ('separate_backward', 'Separate Triton forward+backward'),
        ('full_fused', 'Fully fused (research)'),
    ]
    
    results = {}
    
    for config_name, config_desc in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name} - {config_desc}")
        print(f"{'=' * 80}")
        
        # Create fresh model copy
        model_cpu = copy.deepcopy(model_baseline_cpu)
        model = model_cpu.to('cuda')
        input_ids = input_ids_cpu.to('cuda')
        labels = labels_cpu.to('cuda')
        
        # Apply patches
        if config_name == 'forward_only':
            patch_dense_mlp_with_triton(model)
        elif config_name == 'separate_backward':
            patch_dense_mlp_separate_backward(model)
        elif config_name == 'full_fused':
            patch_dense_mlp_full_fused(model)
        # baseline: no patches
        
        # Benchmark
        print("Benchmarking forward + backward...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        perf = benchmark_forward_backward(model, input_ids, labels)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"  Forward: {perf['forward_ms']:.2f} ms")
        print(f"  Backward: {perf['backward_ms']:.2f} ms")
        print(f"  Total: {perf['total_ms']:.2f} ms")
        print(f"  Peak Memory: {peak_memory:.2f} MB")
        
        # Verify gradients (compare to baseline)
        if config_name != 'baseline':
            print("Verifying gradient equivalence...")
            model_baseline = copy.deepcopy(model_baseline_cpu).to('cuda')
            input_ids_baseline = input_ids_cpu.to('cuda')
            labels_baseline = labels_cpu.to('cuda')
            
            grad_ok = verify_gradients(model_baseline, model, input_ids, labels)
            print(f"  Gradient check: {'✓ PASS' if grad_ok else '✗ FAIL'}")
            
            del model_baseline, input_ids_baseline, labels_baseline
            torch.cuda.empty_cache()
        
        results[config_name] = {
            'description': config_desc,
            'forward_ms': perf['forward_ms'],
            'backward_ms': perf['backward_ms'],
            'total_ms': perf['total_ms'],
            'peak_memory_mb': peak_memory,
        }
        
        # Cleanup
        del model, input_ids, labels
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Print comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    
    baseline = results['baseline']
    
    for config_name, config_desc in configs:
        if config_name == 'baseline':
            continue
        
        res = results[config_name]
        fwd_speedup = baseline['forward_ms'] / res['forward_ms']
        bwd_speedup = baseline['backward_ms'] / res['backward_ms']
        total_speedup = baseline['total_ms'] / res['total_ms']
        mem_savings = baseline['peak_memory_mb'] - res['peak_memory_mb']
        
        print(f"\n{config_name} - {config_desc}")
        print(f"  Forward speedup: {fwd_speedup:.2f}x")
        print(f"  Backward speedup: {bwd_speedup:.2f}x")
        print(f"  Total speedup: {total_speedup:.2f}x")
        print(f"  Memory savings: {mem_savings:.2f} MB")
    
    return results


if __name__ == "__main__":
    results = run_comparison(seed=42)
