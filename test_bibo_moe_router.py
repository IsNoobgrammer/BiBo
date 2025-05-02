import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoMoERouter

def test_router_correctness(router_type="mlp", batch_size=2, seq_len=512, hidden_size=1536):
    """
    Test the correctness of the BiBoMoERouter implementation.
    
    Args:
        router_type (str): Type of router to test ("mlp" or "conv")
        batch_size (int): Batch size for test input
        seq_len (int): Sequence length for test input
        hidden_size (int): Hidden dimension size for test input
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print(f"\n=== Testing {router_type.upper()} Router Correctness ===")
    
    # Create a config with the specified router type
    config = BiBoConfig(
        hidden_size=hidden_size,
        num_routed_experts=8,
        num_experts_per_tok=2,
        router_type=router_type,
        kernel_size=3 if router_type == "conv" else 1
    )
    
    # Initialize the router
    router = BiBoMoERouter(config)
    router.eval()  # Set to eval mode to disable noise
    
    # Create a test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run the router
    top_k_indices, norm_weights = router(hidden_states)
    
    # Check output shapes
    shape_test = (
        top_k_indices.shape == (batch_size, seq_len, config.num_experts_per_tok) and
        norm_weights.shape == (batch_size, seq_len, config.num_experts_per_tok)
    )
    print(f"✓ Output shapes correct: {shape_test}")
    
    # Check that indices are within the valid range
    indices_test = (
        torch.all(top_k_indices >= 0) and 
        torch.all(top_k_indices < config.num_routed_experts)
    )
    print(f"✓ Indices within valid range: {indices_test}")
    
    # Check that weights sum to approximately 1
    weights_sum = norm_weights.sum(dim=-1)
    weights_test = torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
    print(f"✓ Weights sum to 1: {weights_test}")
    
    # For conv router, check causality (earlier tokens shouldn't affect later tokens)
    if router_type == "conv":
        # Create a modified input where we only change a token in the middle
        modified_input = hidden_states.clone()
        mid_pos = seq_len // 2
        modified_input[:, mid_pos, :] = torch.randn_like(modified_input[:, mid_pos, :])
        
        # Run the router on the modified input
        mod_top_k_indices, mod_norm_weights = router(modified_input)
        
        # Check that tokens before the modified position have the same routing decisions
        causality_test = torch.all(top_k_indices[:, :mid_pos, :] == mod_top_k_indices[:, :mid_pos, :])
        print(f"✓ Causality preserved: {causality_test}")
    
    all_tests_passed = shape_test and indices_test and weights_test
    if router_type == "conv":
        all_tests_passed = all_tests_passed and causality_test
    
    print(f"All tests passed: {all_tests_passed}")
    return all_tests_passed

def profile_router_performance(router_types=["mlp", "conv"], 
                              batch_sizes=[1, 2, 4, 8], 
                              seq_lengths=[128, 256, 512, 1024, 2048],
                              hidden_size=1536,
                              num_runs=5):
    """
    Profile the performance of different router types with varying batch sizes and sequence lengths.
    
    Args:
        router_types (list): List of router types to test
        batch_sizes (list): List of batch sizes to test
        seq_lengths (list): List of sequence lengths to test
        hidden_size (int): Hidden dimension size
        num_runs (int): Number of runs for each configuration to average timing
        
    Returns:
        dict: Dictionary containing timing results
    """
    print("\n=== Profiling Router Performance ===")
    
    results = {
        router_type: {
            "batch_sizes": batch_sizes,
            "seq_lengths": seq_lengths,
            "times": np.zeros((len(batch_sizes), len(seq_lengths)))
        }
        for router_type in router_types
    }
    
    for r_idx, router_type in enumerate(router_types):
        print(f"\nProfiling {router_type.upper()} router...")
        
        for b_idx, batch_size in enumerate(batch_sizes):
            for s_idx, seq_len in enumerate(seq_lengths):
                # Create config
                config = BiBoConfig(
                    hidden_size=hidden_size,
                    num_routed_experts=8,
                    num_experts_per_tok=2,
                    router_type=router_type,
                    kernel_size=3 if router_type == "conv" else 1
                )
                
                # Initialize router
                router = BiBoMoERouter(config)
                router.eval()
                
                # Create input
                hidden_states = torch.randn(batch_size, seq_len, hidden_size)
                
                # Warm-up run
                _ = router(hidden_states)
                
                # Timed runs
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = router(hidden_states)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                results[router_type]["times"][b_idx, s_idx] = avg_time
                
                tokens_per_second = (batch_size * seq_len) / avg_time
                print(f"  Batch {batch_size}, Seq {seq_len}: {avg_time:.4f}s ({tokens_per_second:.0f} tokens/sec)")
    
    return results

def plot_performance_comparison(results):
    """
    Plot performance comparison between different router types.
    
    Args:
        results (dict): Results dictionary from profile_router_performance
    """
    router_types = list(results.keys())
    seq_lengths = results[router_types[0]]["seq_lengths"]
    
    plt.figure(figsize=(12, 8))
    
    # Plot for each batch size
    for b_idx, batch_size in enumerate(results[router_types[0]]["batch_sizes"]):
        plt.subplot(2, 2, b_idx + 1)
        
        for router_type in router_types:
            times = results[router_type]["times"][b_idx]
            tokens_per_second = [(batch_size * seq_len) / time for seq_len, time in zip(seq_lengths, times)]
            plt.plot(seq_lengths, tokens_per_second, marker='o', label=f"{router_type.upper()} Router")
        
        plt.title(f"Batch Size: {batch_size}")
        plt.xlabel("Sequence Length")
        plt.ylabel("Tokens per Second")
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("router_performance_comparison.png")
    plt.close()
    print("\nPerformance comparison plot saved as 'router_performance_comparison.png'")

def main():
    # Test correctness
    mlp_test_passed = test_router_correctness(router_type="mlp")
    conv_test_passed = test_router_correctness(router_type="conv")
    
    if not (mlp_test_passed and conv_test_passed):
        print("Correctness tests failed. Skipping performance profiling.")
        return
    
    # Profile performance
    results = profile_router_performance()
    
    # Plot results
    plot_performance_comparison(results)
    
    # Print summary
    print("\n=== Performance Summary ===")
    for router_type in results:
        avg_times = results[router_type]["times"].mean(axis=0)
        avg_tokens_per_second = [(4 * seq_len) / time for seq_len, time in 
                                zip(results[router_type]["seq_lengths"], avg_times)]
        
        print(f"\n{router_type.upper()} Router:")
        print(f"  Average tokens per second: {sum(avg_tokens_per_second)/len(avg_tokens_per_second):.0f}")
        
        # Find best and worst cases
        flat_idx = np.argmax(results[router_type]["times"])
        b_idx, s_idx = np.unravel_index(flat_idx, results[router_type]["times"].shape)
        worst_batch = results[router_type]["batch_sizes"][b_idx]
        worst_seq = results[router_type]["seq_lengths"][s_idx]
        worst_time = results[router_type]["times"][b_idx, s_idx]
        
        flat_idx = np.argmin(results[router_type]["times"])
        b_idx, s_idx = np.unravel_index(flat_idx, results[router_type]["times"].shape)
        best_batch = results[router_type]["batch_sizes"][b_idx]
        best_seq = results[router_type]["seq_lengths"][s_idx]
        best_time = results[router_type]["times"][b_idx, s_idx]
        
        print(f"  Slowest: Batch {worst_batch}, Seq {worst_seq}: {worst_time:.4f}s")
        print(f"  Fastest: Batch {best_batch}, Seq {best_seq}: {best_time:.4f}s")

if __name__ == "__main__":
    main()