import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path to import the BiBo modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_bibo import (
    BiBoRMSNorm,
    BiBoarctanNorm,
    BiBoAlgebraicSigmoid,
    BiBoSoftSign
)

class NormStabilityTester:
    """Tests numerical stability of BiBo normalization layers"""
    
    def __init__(self, hidden_size=768):
        self.hidden_size = hidden_size
        
        # Initialize all norm layers
        self.norm_layers = {
            "RMSNorm": BiBoRMSNorm(hidden_size),
            "ArctanNorm": BiBoarctanNorm(hidden_size),
            "AlgebraicSigmoid": BiBoAlgebraicSigmoid(hidden_size),
            "SoftSign": BiBoSoftSign(hidden_size)
        }
    
    def test_extreme_values(self):
        """Test how each layer handles extreme input values"""
        print("Testing extreme input values...")
        
        # Create test cases with extreme values
        test_cases = {
            "zeros": torch.zeros(1, 10, self.hidden_size),
            "ones": torch.ones(1, 10, self.hidden_size),
            "large_positive": torch.ones(1, 10, self.hidden_size) * 1e6,
            "large_negative": torch.ones(1, 10, self.hidden_size) * -1e6,
            "mixed_large": torch.randn(1, 10, self.hidden_size) * 1e6,
            "small_positive": torch.ones(1, 10, self.hidden_size) * 1e-6,
            "small_negative": torch.ones(1, 10, self.hidden_size) * -1e-6,
            "mixed_small": torch.randn(1, 10, self.hidden_size) * 1e-6,
            "nan_values": torch.full((1, 10, self.hidden_size), float('nan')),
            "inf_values": torch.full((1, 10, self.hidden_size), float('inf')),
            "neg_inf_values": torch.full((1, 10, self.hidden_size), float('-inf')),
        }
        
        results = {}
        
        for case_name, input_tensor in test_cases.items():
            print(f"\nTesting case: {case_name}")
            results[case_name] = {}
            
            for layer_name, layer in self.norm_layers.items():
                try:
                    # Forward pass
                    output = layer(input_tensor)
                    
                    # Check for NaNs or Infs
                    has_nan = torch.isnan(output).any().item()
                    has_inf = torch.isinf(output).any().item()
                    
                    # Calculate statistics if no NaNs or Infs
                    if not has_nan and not has_inf:
                        mean = output.mean().item()
                        std = output.std().item()
                        min_val = output.min().item()
                        max_val = output.max().item()
                        
                        results[case_name][layer_name] = {
                            "status": "OK",
                            "mean": mean,
                            "std": std,
                            "min": min_val,
                            "max": max_val
                        }
                        
                        print(f"{layer_name}: OK - Mean: {mean:.6f}, Std: {std:.6f}, Min: {min_val:.6f}, Max: {max_val:.6f}")
                    else:
                        results[case_name][layer_name] = {
                            "status": "UNSTABLE",
                            "has_nan": has_nan,
                            "has_inf": has_inf
                        }
                        
                        print(f"{layer_name}: UNSTABLE - Has NaN: {has_nan}, Has Inf: {has_inf}")
                
                except Exception as e:
                    results[case_name][layer_name] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                    
                    print(f"{layer_name}: ERROR - {str(e)}")
        
        return results
    
    def test_gradient_stability(self):
        """Test gradient stability with different input magnitudes"""
        print("\nTesting gradient stability...")
        
        # Create test cases with different magnitudes
        magnitudes = [1e-10, 1e-6, 1e-3, 1, 1e3, 1e6, 1e10]
        results = {}
        
        for magnitude in magnitudes:
            print(f"\nTesting magnitude: {magnitude}")
            results[magnitude] = {}
            
            for layer_name, layer in self.norm_layers.items():
                try:
                    # Reset gradients
                    if hasattr(layer, 'weight') and layer.weight.grad is not None:
                        layer.weight.grad.zero_()
                    if hasattr(layer, 'alpha') and layer.alpha.grad is not None:
                        layer.alpha.grad.zero_()
                    
                    # Forward pass
                    input_tensor = torch.randn(1, 10, self.hidden_size) * magnitude
                    input_tensor.requires_grad_(True)
                    output = layer(input_tensor)
                    
                    # Backward pass
                    output.mean().backward()
                    
                    # Check for NaNs or Infs in gradients
                    has_nan_grad = torch.isnan(input_tensor.grad).any().item()
                    has_inf_grad = torch.isinf(input_tensor.grad).any().item()
                    
                    if not has_nan_grad and not has_inf_grad:
                        grad_mean = input_tensor.grad.mean().item()
                        grad_std = input_tensor.grad.std().item()
                        
                        results[magnitude][layer_name] = {
                            "status": "OK",
                            "grad_mean": grad_mean,
                            "grad_std": grad_std
                        }
                        
                        print(f"{layer_name}: OK - Grad Mean: {grad_mean:.6e}, Grad Std: {grad_std:.6e}")
                    else:
                        results[magnitude][layer_name] = {
                            "status": "UNSTABLE",
                            "has_nan_grad": has_nan_grad,
                            "has_inf_grad": has_inf_grad
                        }
                        
                        print(f"{layer_name}: UNSTABLE - Has NaN Grad: {has_nan_grad}, Has Inf Grad: {has_inf_grad}")
                
                except Exception as e:
                    results[magnitude][layer_name] = {
                        "status": "ERROR",
                        "error": str(e)
                    }
                    
                    print(f"{layer_name}: ERROR - {str(e)}")
        
        return results
    
    def plot_stability_heatmap(self, extreme_results):
        """Plot a heatmap of stability results for extreme values"""
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        layer_names = list(self.norm_layers.keys())
        case_names = list(extreme_results.keys())
        
        # Create a matrix for the heatmap
        # 0: ERROR, 1: UNSTABLE, 2: OK
        heatmap_data = np.zeros((len(case_names), len(layer_names)))
        
        for i, case in enumerate(case_names):
            for j, layer in enumerate(layer_names):
                if layer in extreme_results[case]:
                    status = extreme_results[case][layer]["status"]
                    if status == "OK":
                        heatmap_data[i, j] = 2
                    elif status == "UNSTABLE":
                        heatmap_data[i, j] = 1
                    else:  # ERROR
                        heatmap_data[i, j] = 0
        
        # Plot heatmap
        plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        plt.colorbar(ticks=[0, 1, 2], label='Status')
        plt.xticks(np.arange(len(layer_names)), layer_names, rotation=45)
        plt.yticks(np.arange(len(case_names)), case_names)
        plt.title('Normalization Layer Stability with Extreme Values')
        plt.tight_layout()
        plt.savefig('norm_stability_heatmap.png')
        plt.close()
    
    def plot_gradient_stability(self, gradient_results):
        """Plot gradient stability across different input magnitudes"""
        plt.figure(figsize=(12, 8))
        
        layer_names = list(self.norm_layers.keys())
        magnitudes = sorted(list(gradient_results.keys()))
        
        for layer_name in layer_names:
            x_values = []
            y_values = []
            
            for magnitude in magnitudes:
                if layer_name in gradient_results[magnitude] and gradient_results[magnitude][layer_name]["status"] == "OK":
                    x_values.append(magnitude)
                    y_values.append(abs(gradient_results[magnitude][layer_name]["grad_mean"]))
            
            if x_values:
                plt.loglog(x_values, y_values, marker='o', label=layer_name)
        
        plt.xlabel('Input Magnitude')
        plt.ylabel('Absolute Gradient Mean')
        plt.title('Gradient Stability Across Input Magnitudes')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig('norm_gradient_stability.png')
        plt.close()
    
    def run_all_tests(self):
        """Run all stability tests and generate plots"""
        extreme_results = self.test_extreme_values()
        gradient_results = self.test_gradient_stability()
        
        self.plot_stability_heatmap(extreme_results)
        self.plot_gradient_stability(gradient_results)
        
        print("\nStability testing complete. Plots saved to current directory.")


if __name__ == "__main__":
    print("Starting BiBo normalization layers stability testing...")
    
    # Force CPU usage as requested
    device = torch.device("cpu")
    
    # Create and run tester
    tester = NormStabilityTester()
    tester.run_all_tests()