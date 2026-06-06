"""
Simple GLU layer test - compares 4 configurations on just the SwiGLU activation.

Tests 4 configurations:
1. Baseline: PyTorch eager (no fusion)
2. Forward-only: Triton forward, PyTorch backward (current dense_mlp.py)
3. Forward+Backward: Separate Triton kernels (dense_mlp_fused.py)
4. Fully Fused: Single Triton kernel for both (dense_mlp_fused.py - research)
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.kernels.dense_mlp import _TritonSwiGLUFunction
from src.kernels.dense_mlp_fused import _FusedSwiGLUSeparateBackward, _FusedSwiGLUFull


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleSwiGLULayer(nn.Module):
    """Simple SwiGLU layer for testing."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = F.silu(gate) * up
        out = self.down_proj(activated)
        return out


class SimpleSwiGLULayerForwardOnly(nn.Module):
    """SwiGLU layer with forward-only Triton."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate+up
        fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = F.linear(x, fused_weight)
        
        # Triton forward-only
        intermediate = _TritonSwiGLUFunction.apply(gate_up)
        
        # Down projection
        out = self.down_proj(intermediate)
        return out


class SimpleSwiGLULayerSeparateBackward(nn.Module):
    """SwiGLU layer with separate Triton forward+backward."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate+up
        fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = F.linear(x, fused_weight)
        
        # Triton separate forward+backward
        intermediate = _FusedSwiGLUSeparateBackward.apply(gate_up)
        
        # Down projection
        out = self.down_proj(intermediate)
        return out


class SimpleSwiGLULayerFullFused(nn.Module):
    """SwiGLU layer with fully fused Triton (research)."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gate+up
        fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        gate_up = F.linear(x, fused_weight)
        
        # Triton fully fused
        intermediate = _FusedSwiGLUFull.apply(gate_up)
        
        # Down projection
        out = self.down_proj(intermediate)
        return out


def benchmark_layer(model, x, num_iterations=50, warmup=10):
    """Benchmark forward and backward passes."""
    model.train()
    
    # Warmup
    for _ in range(warmup):
        output = model(x)
        loss = output.sum()
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
        output = model(x)
        torch.cuda.synchronize()
        end_fwd = time.perf_counter()
        forward_times.append(end_fwd - start_fwd)
        
        # Backward
        torch.cuda.synchronize()
        start_bwd = time.perf_counter()
        loss = output.sum()
        loss.backward()
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


def verify_gradients(model_baseline, model_test, x, intermediate_size):
    """Verify gradient equivalence."""
    # Zero gradients first
    for p in model_baseline.parameters():
        if p.grad is not None:
            p.grad.zero_()
    for p in model_test.parameters():
        if p.grad is not None:
            p.grad.zero_()
    
    model_baseline.train()
    model_test.train()
    
    # Baseline forward + backward
    output_baseline = model_baseline(x)
    loss_baseline = output_baseline.sum()
    loss_baseline.backward()
    
    # Test forward + backward
    output_test = model_test(x)
    loss_test = output_test.sum()
    loss_test.backward()
    
    # Check output difference
    output_diff = (output_baseline - output_test).abs().max().item()
    loss_diff = abs(loss_baseline.item() - loss_test.item())
    print(f"  Output difference: {output_diff:.6f}")
    print(f"  Loss difference: {loss_diff:.6f}")
    
    # Debug: print some values
    print(f"  Baseline output sample: {output_baseline[0, :5]}")
    print(f"  Test output sample: {output_test[0, :5]}")
    
    # Debug: check if gate_up is correct
    # Create a simple test to verify the kernel works
    print(f"  Debug: Testing kernel directly...")
    test_gate_up = torch.randn(10, 2 * intermediate_size, device='cuda', dtype=torch.float32)
    baseline_gate = test_gate_up[:, :intermediate_size]
    baseline_up = test_gate_up[:, intermediate_size:]
    baseline_result = F.silu(baseline_gate) * baseline_up
    
    test_result = _TritonSwiGLUFunction.apply(test_gate_up)
    kernel_diff = (baseline_result - test_result).abs().max().item()
    print(f"  Kernel-only diff: {kernel_diff:.6f}")
    print(f"  Baseline kernel result sample: {baseline_result[0, :5]}")
    print(f"  Test kernel result sample: {test_result[0, :5]}")
    
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
            print(f"  {n1}: grad diff = {diff:.6f}")
    
    print(f"  Max gradient difference: {max_grad_diff:.6f}")
    print(f"  Parameters with zero/None gradients: {zero_grad_count}/{total_params}")
    
    # Tolerance: output < 1e-3, loss < 1e-2, grad diff < 5e-2 (fp16), no zero/None grads
    return output_diff < 1e-3 and loss_diff < 1e-2 and max_grad_diff < 5e-2 and zero_grad_count == 0


def run_comparison(seed=42):
    """Run comparison across all 4 configurations."""
    set_seed(seed)
    
    # Test parameters
    batch_size = 4
    seq_length = 2048
    hidden_size = 384
    intermediate_size = 1536
    
    # Create input
    x = torch.randn(batch_size * seq_length, hidden_size, device='cuda', dtype=torch.float32)
    
    # Test configurations
    configs = [
        ('baseline', SimpleSwiGLULayer, 'PyTorch eager (no fusion)'),
        ('forward_only', SimpleSwiGLULayerForwardOnly, 'Triton forward, PyTorch backward'),
        ('separate_backward', SimpleSwiGLULayerSeparateBackward, 'Separate Triton forward+backward'),
        ('full_fused', SimpleSwiGLULayerFullFused, 'Fully fused (research)'),
    ]
    
    results = {}
    
    # Create baseline model once to copy weights from
    baseline_model = SimpleSwiGLULayer(hidden_size, intermediate_size).to('cuda')
    
    for config_name, model_class, config_desc in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name} - {config_desc}")
        print(f"{'=' * 80}")
        
        model = model_class(hidden_size, intermediate_size).to('cuda')
        
        # Copy weights from baseline to ensure fair comparison
        if config_name != 'baseline':
            model.gate_proj.weight.data.copy_(baseline_model.gate_proj.weight.data)
            model.up_proj.weight.data.copy_(baseline_model.up_proj.weight.data)
            model.down_proj.weight.data.copy_(baseline_model.down_proj.weight.data)
        
        # Benchmark
        print("Benchmarking forward + backward...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        perf = benchmark_layer(model, x)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"  Forward: {perf['forward_ms']:.2f} ms")
        print(f"  Backward: {perf['backward_ms']:.2f} ms")
        print(f"  Total: {perf['total_ms']:.2f} ms")
        print(f"  Peak Memory: {peak_memory:.2f} MB")
        
        # Verify gradients (compare to baseline)
        if config_name != 'baseline':
            print("Verifying gradient equivalence...")
            grad_ok = verify_gradients(baseline_model, model, x, intermediate_size)
            print(f"  Gradient check: {'✓ PASS' if grad_ok else '✗ FAIL'}")
        
        results[config_name] = {
            'description': config_desc,
            'forward_ms': perf['forward_ms'],
            'backward_ms': perf['backward_ms'],
            'total_ms': perf['total_ms'],
            'peak_memory_mb': peak_memory,
        }
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Print comparison summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    
    baseline = results['baseline']
    
    for config_name, model_class, config_desc in configs:
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
