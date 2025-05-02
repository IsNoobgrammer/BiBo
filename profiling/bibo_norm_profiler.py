import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import os
import random
from tqdm import tqdm
import statistics

# Add the parent directory to the path to import the BiBo modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_bibo import (
    BiBoRMSNorm,
    BiBoDyTNorm,
    BiBoarctanNorm,
    BiBoAlgebraicSigmoid,
    BiBoSoftSign,
    BiBoErf
)

class NormProfiler:
    """Profiler for BiBo normalization layers"""
    
    def __init__(self, hidden_sizes=None, batch_sizes=None, seq_lens=None):
        """
        Initialize the profiler with multiple sizes for more extensive testing.
        
        Args:
            hidden_sizes: List of hidden sizes to test
            batch_sizes: List of batch sizes to test
            seq_lens: List of sequence lengths to test
        """
        # Set default values if not provided
        self.hidden_sizes = hidden_sizes or [2048, 1536]
        self.batch_sizes = batch_sizes or [16, 8]
        self.seq_lens = seq_lens or [1024, 512]
        
        # Results storage
        self.timing_results = {}
        self.output_stats = {}
        self.gradient_stats = {}
        self.detailed_timing = {}
        
        # Generate random seeds for reproducibility
        self.seeds = [random.randint(0, 10000) for _ in range(2)]
        
        print(f"Initialized profiler with:")
        print(f"  Hidden sizes: {self.hidden_sizes}")
        print(f"  Batch sizes: {self.batch_sizes}")
        print(f"  Sequence lengths: {self.seq_lens}")
        print(f"  Random seeds: {self.seeds}")
        
    def _initialize_layers(self, hidden_size):
        """Initialize all norm layers with the given hidden size"""
        return {
            "RMSNorm": BiBoRMSNorm(hidden_size),
            "DyTNorm": BiBoDyTNorm(hidden_size),
            "ArctanNorm": BiBoarctanNorm(hidden_size),
            "AlgebraicSigmoid": BiBoAlgebraicSigmoid(hidden_size),
            "SoftSign": BiBoSoftSign(hidden_size),
            "Erf": BiBoErf(hidden_size)
        }
    
    def _generate_random_input(self, batch_size, seq_len, hidden_size, seed, magnitude=1.0):
        """Generate random input with the given seed and magnitude"""
        torch.manual_seed(seed)
        return torch.randn(batch_size, seq_len, hidden_size) * magnitude
    
    def profile_timing_extensive(self, warmup=20, runs=100):
        """
        Profile the execution time of each normalization layer extensively
        across different sizes and with multiple random inputs.
        """
        print("\nProfiling execution time extensively...")
        
        # Initialize detailed timing results
        for layer_name in ["RMSNorm", "DyTNorm", "ArctanNorm", "AlgebraicSigmoid", "SoftSign", "Erf"]:
            self.detailed_timing[layer_name] = []
        
        # Test across all combinations of sizes
        total_combinations = len(self.hidden_sizes) * len(self.batch_sizes) * len(self.seq_lens) * len(self.seeds)
        with tqdm(total=total_combinations, desc="Testing combinations") as pbar:
            for hidden_size in self.hidden_sizes:
                # Initialize layers for this hidden size
                norm_layers = self._initialize_layers(hidden_size)
                
                for batch_size in self.batch_sizes:
                    for seq_len in self.seq_lens:
                        for seed in self.seeds:
                            # Generate random input
                            input_tensor = self._generate_random_input(batch_size, seq_len, hidden_size, seed)
                            
                            # Profile each layer
                            for name, layer in norm_layers.items():
                                # Warmup
                                for _ in range(warmup):
                                    _ = layer(input_tensor)
                                
                                # Measure
                                start_time = time.time()
                                for _ in range(runs):
                                    _ = layer(input_tensor)
                                end_time = time.time()
                                
                                avg_time = (end_time - start_time) / runs * 1000  # ms
                                
                                # Store detailed timing
                                self.detailed_timing[name].append({
                                    "hidden_size": hidden_size,
                                    "batch_size": batch_size,
                                    "seq_len": seq_len,
                                    "seed": seed,
                                    "time_ms": avg_time,
                                    "elements": batch_size * seq_len * hidden_size
                                })
                            
                            pbar.update(1)
        
        # Calculate aggregate statistics
        for name in self.detailed_timing:
            times = [entry["time_ms"] for entry in self.detailed_timing[name]]
            elements = [entry["elements"] for entry in self.detailed_timing[name]]
            
            # Calculate time per million elements for fair comparison
            times_per_million = [time * 1e6 / elem for time, elem in zip(times, elements)]
            
            self.timing_results[name] = {
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "stddev_ms": statistics.stdev(times) if len(times) > 1 else 0,
                "mean_per_million": statistics.mean(times_per_million),
                "median_per_million": statistics.median(times_per_million)
            }
            
            print(f"\n{name} timing statistics:")
            print(f"  Mean: {self.timing_results[name]['mean_ms']:.3f} ms")
            print(f"  Median: {self.timing_results[name]['median_ms']:.3f} ms")
            print(f"  Min: {self.timing_results[name]['min_ms']:.3f} ms")
            print(f"  Max: {self.timing_results[name]['max_ms']:.3f} ms")
            print(f"  StdDev: {self.timing_results[name]['stddev_ms']:.3f} ms")
            print(f"  Mean per million elements: {self.timing_results[name]['mean_per_million']:.6f} ms")
    
    def profile_timing_by_operation(self, hidden_size=2048, batch_size=16, seq_len=1024, runs=20):
        """
        Profile the execution time of each operation within the normalization layers
        to understand why some layers might be faster than others.
        """
        print("\nProfiling execution time by operation...")
        
        # Initialize layers
        norm_layers = self._initialize_layers(hidden_size)
        input_tensor = self._generate_random_input(batch_size, seq_len, hidden_size, seed=42)
        
        operation_times = {}
        
        # Profile RMSNorm operations
        layer = norm_layers["RMSNorm"]
        
        # Measure variance calculation
        start_time = time.time()
        for _ in range(runs):
            _ = input_tensor.pow(2).mean(-1, keepdim=True)
        end_time = time.time()
        operation_times["RMSNorm_variance"] = (end_time - start_time) / runs * 1000
        
        # Measure rsqrt operation
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        start_time = time.time()
        for _ in range(runs):
            _ = torch.rsqrt(variance + layer.variance_epsilon)
        end_time = time.time()
        operation_times["RMSNorm_rsqrt"] = (end_time - start_time) / runs * 1000
        
        # Measure multiplication
        rsqrt_result = torch.rsqrt(variance + layer.variance_epsilon)
        start_time = time.time()
        for _ in range(runs):
            _ = input_tensor * rsqrt_result
        end_time = time.time()
        operation_times["RMSNorm_multiply"] = (end_time - start_time) / runs * 1000
        
        # Profile activation-based norms
        for name in ["DyTNorm", "ArctanNorm", "AlgebraicSigmoid", "SoftSign", "Erf"]:
            layer = norm_layers[name]
            
            # Measure alpha scaling
            start_time = time.time()
            for _ in range(runs):
                _ = layer.alpha * input_tensor
            end_time = time.time()
            operation_times[f"{name}_alpha_scaling"] = (end_time - start_time) / runs * 1000
            
            # Measure activation function
            scaled_input = layer.alpha * input_tensor
            
            if name == "DyTNorm":
                start_time = time.time()
                for _ in range(runs):
                    _ = torch.tanh(scaled_input)
                end_time = time.time()
            elif name == "ArctanNorm":
                start_time = time.time()
                for _ in range(runs):
                    _ = torch.arctan(scaled_input)
                end_time = time.time()
            elif name == "AlgebraicSigmoid":
                start_time = time.time()
                for _ in range(runs):
                    _ = scaled_input / torch.sqrt(1.0 + scaled_input**2 + 1e-6)
                end_time = time.time()
            elif name == "SoftSign":
                start_time = time.time()
                for _ in range(runs):
                    _ = scaled_input / (1.0 + torch.abs(scaled_input))
                end_time = time.time()
            elif name == "Erf":
                start_time = time.time()
                for _ in range(runs):
                    _ = torch.erf(scaled_input)
                end_time = time.time()
            
            operation_times[f"{name}_activation"] = (end_time - start_time) / runs * 1000
            
            # Measure weight multiplication
            if name == "DyTNorm":
                activated = torch.tanh(scaled_input)
            elif name == "ArctanNorm":
                activated = torch.arctan(scaled_input)
            elif name == "AlgebraicSigmoid":
                activated = scaled_input / torch.sqrt(1.0 + scaled_input**2 + 1e-6)
            elif name == "SoftSign":
                activated = scaled_input / (1.0 + torch.abs(scaled_input))
            elif name == "Erf":
                activated = torch.erf(scaled_input)
                
            start_time = time.time()
            for _ in range(runs):
                _ = layer.weight * activated
            end_time = time.time()
            operation_times[f"{name}_weight_multiply"] = (end_time - start_time) / runs * 1000
        
        # Print operation times
        print("\nOperation times (ms):")
        for op, time_ms in sorted(operation_times.items()):
            print(f"  {op}: {time_ms:.3f} ms")
        
        return operation_times
    
    def profile_output_characteristics(self):
        """Profile the statistical characteristics of each layer's output"""
        print("\nProfiling output characteristics...")
        
        # Use the largest hidden size for this test
        hidden_size = max(self.hidden_sizes)
        norm_layers = self._initialize_layers(hidden_size)
        
        # Test with different input magnitudes
        magnitudes = [0.01, 1.0, 10.0, 100.0]
        
        for magnitude in magnitudes:
            print(f"\nInput magnitude: {magnitude}")
            
            # Generate random input
            input_tensor = self._generate_random_input(16, 1024, hidden_size, seed=42, magnitude=magnitude)
            
            print(f"Input stats - Mean: {input_tensor.mean().item():.6f}, Std: {input_tensor.std().item():.6f}, "
                  f"Min: {input_tensor.min().item():.6f}, Max: {input_tensor.max().item():.6f}")
            
            for name, layer in norm_layers.items():
                output = layer(input_tensor)
                
                # Calculate statistics
                mean = output.mean().item()
                std = output.std().item()
                min_val = output.min().item()
                max_val = output.max().item()
                
                if name not in self.output_stats:
                    self.output_stats[name] = {}
                
                self.output_stats[name][magnitude] = {
                    "mean": mean,
                    "std": std,
                    "min": min_val,
                    "max": max_val
                }
                
                print(f"{name} - Mean: {mean:.6f}, Std: {std:.6f}, Min: {min_val:.6f}, Max: {max_val:.6f}")
    
    def profile_gradient_flow(self):
        """Profile gradient flow through each normalization layer"""
        print("\nProfiling gradient flow...")
        
        # Use the largest hidden size for this test
        hidden_size = max(self.hidden_sizes)
        norm_layers = self._initialize_layers(hidden_size)
        
        # Test with different input magnitudes
        magnitudes = [0.01, 1.0, 10.0, 100.0]
        
        for magnitude in magnitudes:
            print(f"\nInput magnitude: {magnitude}")
            
            for name, layer in norm_layers.items():
                # Reset gradients
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    layer.weight.grad.zero_()
                if hasattr(layer, 'alpha') and layer.alpha.grad is not None:
                    layer.alpha.grad.zero_()
                
                # Forward pass
                input_tensor = self._generate_random_input(16, 1024, hidden_size, seed=42, magnitude=magnitude)
                input_tensor.requires_grad_(True)
                output = layer(input_tensor)
                
                # Backward pass
                output.mean().backward()
                
                # Collect gradient statistics
                input_grad = input_tensor.grad
                
                if name not in self.gradient_stats:
                    self.gradient_stats[name] = {}
                
                self.gradient_stats[name][magnitude] = {
                    "input_grad_mean": input_grad.mean().item(),
                    "input_grad_std": input_grad.std().item(),
                    "input_grad_min": input_grad.min().item(),
                    "input_grad_max": input_grad.max().item()
                }
                
                if hasattr(layer, 'weight'):
                    self.gradient_stats[name][magnitude]["weight_grad_mean"] = layer.weight.grad.mean().item()
                    self.gradient_stats[name][magnitude]["weight_grad_std"] = layer.weight.grad.std().item()
                
                if hasattr(layer, 'alpha'):
                    self.gradient_stats[name][magnitude]["alpha_grad"] = layer.alpha.grad.item()
                
                print(f"{name} - Input grad mean: {input_grad.mean().item():.6f}, std: {input_grad.std().item():.6f}")
                if hasattr(layer, 'alpha'):
                    print(f"  Alpha grad: {layer.alpha.grad.item():.6f}")
    
    def plot_timing_results(self):
        """Plot timing results"""
        plt.figure(figsize=(12, 8))
        
        names = list(self.timing_results.keys())
        means = [stats["mean_ms"] for stats in self.timing_results.values()]
        stds = [stats["stddev_ms"] for stats in self.timing_results.values()]
        
        # Sort by mean time
        sorted_indices = np.argsort(means)
        sorted_names = [names[i] for i in sorted_indices]
        sorted_means = [means[i] for i in sorted_indices]
        sorted_stds = [stds[i] for i in sorted_indices]
        
        plt.bar(sorted_names, sorted_means, yerr=sorted_stds, capsize=5, color='skyblue')
        plt.title('Forward Pass Time by Normalization Type (Lower is Better)')
        plt.xlabel('Normalization Layer')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('norm_timing_comparison.png')
        plt.close()
        
        # Plot time per million elements
        plt.figure(figsize=(12, 8))
        
        means_per_million = [stats["mean_per_million"] for stats in self.timing_results.values()]
        
        # Sort by mean time per million
        sorted_indices = np.argsort(means_per_million)
        sorted_names = [names[i] for i in sorted_indices]
        sorted_means_per_million = [means_per_million[i] for i in sorted_indices]
        
        plt.bar(sorted_names, sorted_means_per_million, color='lightgreen')
        plt.title('Time per Million Elements by Normalization Type (Lower is Better)')
        plt.xlabel('Normalization Layer')
        plt.ylabel('Time per Million Elements (ms)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('norm_timing_per_million.png')
        plt.close()
        
        # Plot timing by hidden size
        plt.figure(figsize=(14, 8))
        
        for name in names:
            hidden_sizes = []
            times = []
            
            for entry in self.detailed_timing[name]:
                hidden_sizes.append(entry["hidden_size"])
                times.append(entry["time_ms"])
            
            # Group by hidden size
            unique_sizes = sorted(set(hidden_sizes))
            mean_times = []
            
            for size in unique_sizes:
                size_times = [times[i] for i in range(len(times)) if hidden_sizes[i] == size]
                mean_times.append(statistics.mean(size_times))
            
            plt.plot(unique_sizes, mean_times, marker='o', label=name)
        
        plt.title('Forward Pass Time by Hidden Size')
        plt.xlabel('Hidden Size')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('norm_timing_by_hidden_size.png')
        plt.close()
    
    def plot_operation_times(self, operation_times):
        """Plot operation times"""
        plt.figure(figsize=(14, 10))
        
        # Group operations by layer
        layer_operations = {}
        
        for op, time in operation_times.items():
            layer_name = op.split('_')[0]
            operation = '_'.join(op.split('_')[1:])
            
            if layer_name not in layer_operations:
                layer_operations[layer_name] = {}
            
            layer_operations[layer_name][operation] = time
        
        # Plot
        num_layers = len(layer_operations)
        num_ops_per_layer = max(len(ops) for ops in layer_operations.values())
        
        width = 0.8 / num_layers
        x = np.arange(num_ops_per_layer)
        
        for i, (layer_name, operations) in enumerate(sorted(layer_operations.items())):
            op_names = sorted(operations.keys())
            op_times = [operations[op] for op in op_names]
            
            plt.bar(x + i * width, op_times, width, label=layer_name)
            
            # Only set x-tick labels for the first layer
            if i == 0:
                plt.xticks(x + width * (num_layers - 1) / 2, op_names, rotation=45)
        
        plt.title('Operation Times by Layer')
        plt.xlabel('Operation')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('norm_operation_times.png')
        plt.close()
    
    def plot_output_distributions(self):
        """Plot output distributions for different input magnitudes"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 16))
        
        # Plot means
        magnitudes = sorted(next(iter(self.output_stats.values())).keys())
        
        for name in self.output_stats:
            means = [self.output_stats[name][mag]["mean"] for mag in magnitudes]
            axes[0].semilogx(magnitudes, means, marker='o', label=name)
        
        axes[0].set_title('Output Mean by Input Magnitude')
        axes[0].set_xlabel('Input Magnitude')
        axes[0].set_ylabel('Output Mean')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot standard deviations
        for name in self.output_stats:
            stds = [self.output_stats[name][mag]["std"] for mag in magnitudes]
            axes[1].semilogx(magnitudes, stds, marker='o', label=name)
        
        axes[1].set_title('Output Standard Deviation by Input Magnitude')
        axes[1].set_xlabel('Input Magnitude')
        axes[1].set_ylabel('Output Standard Deviation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('norm_output_distributions.png')
        plt.close()
    
    def plot_gradient_stats(self):
        """Plot gradient statistics"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 16))
        
        # Plot input gradient means
        magnitudes = sorted(next(iter(self.gradient_stats.values())).keys())
        
        for name in self.gradient_stats:
            grad_means = [self.gradient_stats[name][mag]["input_grad_mean"] for mag in magnitudes]
            axes[0].semilogx(magnitudes, grad_means, marker='o', label=name)
        
        axes[0].set_title('Input Gradient Mean by Input Magnitude')
        axes[0].set_xlabel('Input Magnitude')
        axes[0].set_ylabel('Input Gradient Mean')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot alpha gradients if available
        for name in self.gradient_stats:
            if "alpha_grad" in self.gradient_stats[name][magnitudes[0]]:
                alpha_grads = [self.gradient_stats[name][mag]["alpha_grad"] for mag in magnitudes]
                axes[1].semilogx(magnitudes, alpha_grads, marker='o', label=f"{name} Alpha")
        
        axes[1].set_title('Alpha Parameter Gradient by Input Magnitude')
        axes[1].set_xlabel('Input Magnitude')
        axes[1].set_ylabel('Alpha Gradient')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('norm_gradient_stats.png')
        plt.close()
    
    def plot_activation_functions(self):
        """Plot the activation functions for each normalization layer"""
        plt.figure(figsize=(12, 8))
        
        x = torch.linspace(-128, 128, 1000)
        
        for name in ["DyTNorm", "ArctanNorm", "AlgebraicSigmoid", "SoftSign", "Erf"]:
            # Create a temporary layer with hidden_size=1
            layer = self._initialize_layers(1)[name]
            
            # Create a single feature input
            x_input = x.view(-1, 1)
            y = layer(x_input).view(-1).detach().numpy()
            
            plt.plot(x.numpy(), y, label=name)
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Activation Functions of Different Normalization Layers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('norm_activation_functions.png')
        plt.close()
    
    def run_all_profiles(self):
        """Run all profiling tasks and generate plots"""
        self.profile_timing_extensive()
        operation_times = self.profile_timing_by_operation()
        self.profile_output_characteristics()
        self.profile_gradient_flow()
        
        self.plot_timing_results()
        self.plot_operation_times(operation_times)
        self.plot_output_distributions()
        self.plot_gradient_stats()
        self.plot_activation_functions()
        
        print("\nProfiling complete. Plots saved to current directory.")


if __name__ == "__main__":
    print("Starting enhanced BiBo normalization layers profiling...")
    
    # Force CPU usage as requested
    torch.set_num_threads(1)  # Use single thread for more consistent timing
    device = torch.device("cpu")
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Create and run profiler with extensive settings
    profiler = NormProfiler()
    profiler.run_all_profiles()