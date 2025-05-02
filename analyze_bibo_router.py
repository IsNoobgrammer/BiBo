import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoMoERouter


@torch.no_grad
def analyze_expert_selection(router_type="mlp", batch_size=4, seq_len=1024, hidden_size=1536, 
                            num_experts=8, num_experts_per_tok=2, seed=42):
    """
    Analyze which experts are selected most frequently by the router.
    
    Args:
        router_type (str): Type of router to analyze ("mlp" or "conv")
        batch_size (int): Batch size for test input
        seq_len (int): Sequence length for test input
        hidden_size (int): Hidden dimension size for test input
        num_experts (int): Number of experts in the router
        num_experts_per_tok (int): Number of experts selected per token
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing expert selection statistics
    """
    print(f"\n=== Analyzing Expert Selection for {router_type.upper()} Router ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a config with the specified router type
    config = BiBoConfig(
        hidden_size=hidden_size,
        num_routed_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
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
    
    # Analyze expert selection
    total_tokens = batch_size * seq_len
    
    # Flatten the indices for analysis
    flat_indices = top_k_indices.view(-1).cpu().numpy()
    
    # Count occurrences of each expert
    expert_counts = Counter(flat_indices)
    
    # Calculate selection frequency for each expert
    expert_freq = {expert: count / (total_tokens * num_experts_per_tok) for expert, count in expert_counts.items()}
    
    # Calculate average weight for each expert
    expert_weights = {}
    for expert_idx in range(num_experts):
        # Find all occurrences of this expert
        mask = (top_k_indices == expert_idx)
        if mask.sum() > 0:
            # Get the average weight assigned to this expert
            avg_weight = norm_weights[mask].mean().item()
            expert_weights[expert_idx] = avg_weight
        else:
            expert_weights[expert_idx] = 0.0
    
    # Analyze position-based selection patterns
    position_patterns = torch.zeros(seq_len, num_experts)
    for pos in range(seq_len):
        for batch in range(batch_size):
            for k in range(num_experts_per_tok):
                expert_idx = top_k_indices[batch, pos, k].item()
                weight = norm_weights[batch, pos, k].item()
                position_patterns[pos, expert_idx] += weight / batch_size
    
    # Analyze weight distribution
    all_weights = norm_weights.view(-1).cpu().numpy()
    
    results = {
        "router_type": router_type,
        "expert_freq": expert_freq,
        "expert_weights": expert_weights,
        "position_patterns": position_patterns.cpu().numpy(),
        "all_weights": all_weights,
        "top_k_indices": top_k_indices.cpu().numpy(),
        "norm_weights": norm_weights.cpu().numpy()
    }
    
    # Print summary statistics
    print(f"Total tokens analyzed: {total_tokens}")
    print("\nExpert selection frequency:")
    for expert, freq in sorted(expert_freq.items()):
        print(f"  Expert {expert}: {freq:.4f} ({freq * 100:.2f}%)")
    
    print("\nAverage weight per expert:")
    for expert, weight in sorted(expert_weights.items()):
        print(f"  Expert {expert}: {weight:.4f}")
    
    return results

@torch.no_grad
def analyze_context_sensitivity(router_type="mlp", batch_size=4, seq_len=1024, hidden_size=1536, 
                               num_experts=8, num_experts_per_tok=2, seed=42):
    """
    Analyze how sensitive the router is to local context changes.
    
    Args:
        router_type (str): Type of router to analyze ("mlp" or "conv")
        batch_size (int): Batch size for test input
        seq_len (int): Sequence length for test input
        hidden_size (int): Hidden dimension size for test input
        num_experts (int): Number of experts in the router
        num_experts_per_tok (int): Number of experts selected per token
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing context sensitivity metrics
    """
    print(f"\n=== Analyzing Context Sensitivity for {router_type.upper()} Router ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a config with the specified router type
    config = BiBoConfig(
        hidden_size=hidden_size,
        num_routed_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        router_type=router_type,
        kernel_size=3 if router_type == "conv" else 1
    )
    
    # Initialize the router
    router = BiBoMoERouter(config)
    router.eval()  # Set to eval mode to disable noise
    
    # Create a base input
    base_input = torch.randn(batch_size, seq_len, hidden_size)
    
    # Run the router on the base input
    base_indices, base_weights = router(base_input)
    
    # Create a series of modified inputs with local perturbations
    num_perturbations = 10
    perturbation_results = []
    
    for i in range(num_perturbations):
        # Create a modified input with a local perturbation
        modified_input = base_input.clone()
        
        # Perturb a random position in each sequence
        for b in range(batch_size):
            pos = np.random.randint(1, seq_len - 1)  # Avoid first and last positions
            modified_input[b, pos, :] = torch.randn_like(modified_input[b, pos, :])
        
        # Run the router on the modified input
        mod_indices, mod_weights = router(modified_input)
        
        # Analyze how the perturbation affects routing decisions
        # For each batch, check how many positions changed their routing decisions
        changes_before = []  # Changes before the perturbation
        changes_at = []      # Changes at the perturbation
        changes_after = []   # Changes after the perturbation
        
        for b in range(batch_size):
            pos = np.random.randint(1, seq_len - 1)  # Same position as perturbed
            
            # Count positions before perturbation that changed routing
            before_changes = torch.sum(base_indices[b, :pos, :] != mod_indices[b, :pos, :]).item()
            changes_before.append(before_changes / (pos * num_experts_per_tok))
            
            # Check if the perturbed position changed routing
            at_change = torch.sum(base_indices[b, pos, :] != mod_indices[b, pos, :]).item()
            changes_at.append(at_change / num_experts_per_tok)
            
            # Count positions after perturbation that changed routing
            after_changes = torch.sum(base_indices[b, pos+1:, :] != mod_indices[b, pos+1:, :]).item()
            changes_after.append(after_changes / ((seq_len - pos - 1) * num_experts_per_tok))
        
        perturbation_results.append({
            "changes_before": np.mean(changes_before),
            "changes_at": np.mean(changes_at),
            "changes_after": np.mean(changes_after)
        })
    
    # Average results across all perturbations
    avg_changes_before = np.mean([r["changes_before"] for r in perturbation_results])
    avg_changes_at = np.mean([r["changes_at"] for r in perturbation_results])
    avg_changes_after = np.mean([r["changes_after"] for r in perturbation_results])
    
    results = {
        "router_type": router_type,
        "avg_changes_before": avg_changes_before,
        "avg_changes_at": avg_changes_at,
        "avg_changes_after": avg_changes_after,
        "perturbation_results": perturbation_results
    }
    
    # Print summary statistics
    print(f"Average proportion of routing changes:")
    print(f"  Before perturbation: {avg_changes_before:.4f} ({avg_changes_before * 100:.2f}%)")
    print(f"  At perturbation: {avg_changes_at:.4f} ({avg_changes_at * 100:.2f}%)")
    print(f"  After perturbation: {avg_changes_after:.4f} ({avg_changes_after * 100:.2f}%)")
    
    return results

def plot_expert_selection_heatmap(results, save_path="expert_selection_heatmap.png"):
    """
    Plot a heatmap of expert selection patterns across sequence positions.
    
    Args:
        results (dict): Results from analyze_expert_selection
        save_path (str): Path to save the plot
    """
    position_patterns = results["position_patterns"]
    router_type = results["router_type"]
    
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap of expert selection by position
    sns.heatmap(position_patterns.T, cmap="viridis", 
                xticklabels=100, yticklabels=True)
    
    plt.title(f"Expert Selection Patterns by Position ({router_type.upper()} Router)")
    plt.xlabel("Sequence Position")
    plt.ylabel("Expert Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Expert selection heatmap saved as '{save_path}'")

def plot_weight_distribution(results, save_path="weight_distribution.png"):
    """
    Plot the distribution of normalized weights.
    
    Args:
        results (dict): Results from analyze_expert_selection
        save_path (str): Path to save the plot
    """
    all_weights = results["all_weights"]
    router_type = results["router_type"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of normalized weights
    sns.histplot(all_weights, bins=50, kde=True)
    
    plt.title(f"Distribution of Normalized Weights ({router_type.upper()} Router)")
    plt.xlabel("Normalized Weight")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Weight distribution plot saved as '{save_path}'")

def plot_expert_frequency(mlp_results, conv_results, save_path="expert_frequency_comparison.png"):
    """
    Plot a comparison of expert selection frequency between MLP and Conv routers.
    
    Args:
        mlp_results (dict): Results from analyze_expert_selection for MLP router
        conv_results (dict): Results from analyze_expert_selection for Conv router
        save_path (str): Path to save the plot
    """
    mlp_freq = mlp_results["expert_freq"]
    conv_freq = conv_results["expert_freq"]
    
    # Ensure all experts are represented
    all_experts = sorted(set(list(mlp_freq.keys()) + list(conv_freq.keys())))
    
    mlp_values = [mlp_freq.get(expert, 0) for expert in all_experts]
    conv_values = [conv_freq.get(expert, 0) for expert in all_experts]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(all_experts))
    width = 0.35
    
    plt.bar(x - width/2, mlp_values, width, label="MLP Router")
    plt.bar(x + width/2, conv_values, width, label="Conv Router")
    
    plt.xlabel("Expert Index")
    plt.ylabel("Selection Frequency")
    plt.title("Expert Selection Frequency Comparison")
    plt.xticks(x, all_experts)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Expert frequency comparison plot saved as '{save_path}'")

def plot_context_sensitivity_comparison(mlp_results, conv_results, save_path="context_sensitivity_comparison.png"):
    """
    Plot a comparison of context sensitivity between MLP and Conv routers.
    
    Args:
        mlp_results (dict): Results from analyze_context_sensitivity for MLP router
        conv_results (dict): Results from analyze_context_sensitivity for Conv router
        save_path (str): Path to save the plot
    """
    categories = ["Before Perturbation", "At Perturbation", "After Perturbation"]
    
    mlp_values = [
        mlp_results["avg_changes_before"],
        mlp_results["avg_changes_at"],
        mlp_results["avg_changes_after"]
    ]
    
    conv_values = [
        conv_results["avg_changes_before"],
        conv_results["avg_changes_at"],
        conv_results["avg_changes_after"]
    ]
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, mlp_values, width, label="MLP Router")
    plt.bar(x + width/2, conv_values, width, label="Conv Router")
    
    plt.xlabel("Position Relative to Perturbation")
    plt.ylabel("Proportion of Routing Changes")
    plt.title("Context Sensitivity Comparison")
    plt.xticks(x, categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Context sensitivity comparison plot saved as '{save_path}'")

def plot_top_k_weight_distribution(results, save_path="top_k_weight_distribution.png"):
    """
    Plot the distribution of weights between the top-1 and top-2 experts.
    
    Args:
        results (dict): Results from analyze_expert_selection
        save_path (str): Path to save the plot
    """
    norm_weights = results["norm_weights"]
    router_type = results["router_type"]
    
    # For each token, get the ratio of weight assigned to top-1 vs top-2
    top1_weights = norm_weights[:, :, 0].flatten()
    top2_weights = norm_weights[:, :, 1].flatten() if norm_weights.shape[2] > 1 else np.zeros_like(top1_weights)
    
    weight_ratios = top1_weights / (top1_weights + top2_weights)
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of weight ratios
    sns.histplot(weight_ratios, bins=50, kde=True)
    
    plt.title(f"Distribution of Top-1 Weight Proportion ({router_type.upper()} Router)")
    plt.xlabel("Proportion of Weight Assigned to Top-1 Expert")
    plt.ylabel("Frequency")
    plt.axvline(x=0.5, color='r', linestyle='--', label="Equal Weighting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Top-k weight distribution plot saved as '{save_path}'")


@torch.no_grad
def main():
    # Set parameters
    batch_size = 4
    seq_len = 1024
    hidden_size = 1536
    num_experts = 8
    num_experts_per_tok = 2
    seed = 42
    
    # Analyze expert selection patterns
    mlp_selection_results = analyze_expert_selection(
        router_type="mlp", 
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        seed=seed
    )
    
    conv_selection_results = analyze_expert_selection(
        router_type="conv", 
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        seed=seed
    )
    
    # Analyze context sensitivity
    mlp_context_results = analyze_context_sensitivity(
        router_type="mlp", 
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        seed=seed
    )
    
    conv_context_results = analyze_context_sensitivity(
        router_type="conv", 
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        seed=seed
    )
    
    # Generate plots
    plot_expert_selection_heatmap(mlp_selection_results, "mlp_expert_selection_heatmap.png")
    plot_expert_selection_heatmap(conv_selection_results, "conv_expert_selection_heatmap.png")
    
    plot_weight_distribution(mlp_selection_results, "mlp_weight_distribution.png")
    plot_weight_distribution(conv_selection_results, "conv_weight_distribution.png")
    
    plot_expert_frequency(mlp_selection_results, conv_selection_results)
    
    plot_context_sensitivity_comparison(mlp_context_results, conv_context_results)
    
    plot_top_k_weight_distribution(mlp_selection_results, "mlp_top_k_weight_distribution.png")
    plot_top_k_weight_distribution(conv_selection_results, "conv_top_k_weight_distribution.png")
    
    # Print summary comparison
    print("\n=== Router Comparison Summary ===")
    
    # Compare expert utilization
    mlp_entropy = -sum(p * np.log2(p) for p in mlp_selection_results["expert_freq"].values() if p > 0)
    conv_entropy = -sum(p * np.log2(p) for p in conv_selection_results["expert_freq"].values() if p > 0)
    max_entropy = np.log2(num_experts)
    
    print(f"Expert utilization entropy (higher is more uniform):")
    print(f"  MLP Router: {mlp_entropy:.4f} ({mlp_entropy/max_entropy*100:.2f}% of max)")
    print(f"  Conv Router: {conv_entropy:.4f} ({conv_entropy/max_entropy*100:.2f}% of max)")
    
    # Compare context sensitivity
    print(f"\nContext sensitivity (% of routing changes after perturbation):")
    print(f"  MLP Router: {mlp_context_results['avg_changes_after']*100:.2f}%")
    print(f"  Conv Router: {conv_context_results['avg_changes_after']*100:.2f}%")
    
    # Compare weight distribution
    mlp_top1_ratio = np.mean(mlp_selection_results["norm_weights"][:,:,0])
    conv_top1_ratio = np.mean(conv_selection_results["norm_weights"][:,:,0])
    
    print(f"\nAverage weight assigned to top-1 expert:")
    print(f"  MLP Router: {mlp_top1_ratio:.4f} ({mlp_top1_ratio*100:.2f}%)")
    print(f"  Conv Router: {conv_top1_ratio:.4f} ({conv_top1_ratio*100:.2f}%)")
    
    # Conclusion
    print("\n=== Conclusion ===")
    if conv_entropy > mlp_entropy:
        print("The convolutional router distributes tokens more evenly across experts.")
    else:
        print("The MLP router distributes tokens more evenly across experts.")
        
    if conv_context_results['avg_changes_after'] > mlp_context_results['avg_changes_after']:
        print("The convolutional router is more sensitive to local context changes.")
    else:
        print("The MLP router is more sensitive to local context changes.")
    
    if abs(conv_top1_ratio - 0.5) < abs(mlp_top1_ratio - 0.5):
        print("The convolutional router assigns more balanced weights between top experts.")
    else:
        print("The MLP router assigns more balanced weights between top experts.")

if __name__ == "__main__":
    main()