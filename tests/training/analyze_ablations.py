"""
Analyze full ablation suite:
- Compare baseline vs bias+scaling vs bias+scaling+noise
- Noise sweep analysis
- MLP vs Conv router comparison
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import glob

def load_all_ablations():
    """Load all ablation results"""
    base_dir = Path('logs/router_ablation')
    results = {}
    
    for config_path in base_dir.glob('*/config.json'):
        with open(config_path) as f:
            config = json.load(f)
        
        ablation_name = config['ablation']
        logs_path = config_path.parent / 'logs.npz'
        
        if logs_path.exists():
            logs = np.load(logs_path)
            results[ablation_name] = {'config': config, 'logs': logs}
    
    return results

def plot_convergence_comparison(results):
    """Compare convergence speed across all ablations"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Comparison: MLP vs Conv Router', fontsize=16, fontweight='bold')
    
    router_types = ['mlp', 'conv']
    colors = {'baseline': 'gray', 'bias_scaling': 'blue', 
              '0.05': 'green', '0.10': 'orange', '0.20': 'red', '0.50': 'purple'}
    
    for idx, router_type in enumerate(router_types):
        ax = axes[idx]
        
        # Baseline
        key = f'{router_type}_baseline'
        if key in results:
            data = results[key]
            ax.plot(data['logs']['steps'], data['logs']['losses'], 
                   label='Baseline (no bias)', color=colors['baseline'], linewidth=2, linestyle='--')
        
        # Bias + Scaling
        key = f'{router_type}_bias_scaling'
        if key in results:
            data = results[key]
            ax.plot(data['logs']['steps'], data['logs']['losses'], 
                   label='Bias + Scaling', color=colors['bias_scaling'], linewidth=2)
        
        # Bias + Scaling + Noise
        for noise in [0.05, 0.10, 0.20, 0.50]:
            key = f'{router_type}_bias_scaling_noise_{noise:.2f}'
            if key in results:
                data = results[key]
                ax.plot(data['logs']['steps'], data['logs']['losses'], 
                       label=f'+ Noise {noise}', color=colors[f'{noise:.2f}'], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{router_type.upper()} Router', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/router_ablation/convergence_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Convergence comparison → logs/router_ablation/convergence_comparison.png")
    plt.close()

def plot_noise_effect(results):
    """Analyze noise effect on convergence and expert selection"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Noise Effect Analysis', fontsize=16, fontweight='bold')
    
    router_types = ['mlp', 'conv']
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.50]
    
    for col, router_type in enumerate(router_types):
        # Convergence speed vs noise
        steps_to_converge = []
        final_losses = []
        noise_vals = []
        
        for noise in noise_levels:
            if noise == 0.0:
                key = f'{router_type}_bias_scaling'
            else:
                key = f'{router_type}_bias_scaling_noise_{noise:.2f}'
            
            if key in results:
                data = results[key]
                steps_to_converge.append(data['config']['steps'])
                final_losses.append(data['config']['final_loss'])
                noise_vals.append(noise)
        
        axes[0, col].plot(noise_vals, steps_to_converge, 'o-', linewidth=2, markersize=10, color='steelblue')
        axes[0, col].set_xlabel('Noise Level', fontsize=12)
        axes[0, col].set_ylabel('Steps to Converge', fontsize=12)
        axes[0, col].set_title(f'{router_type.upper()}: Convergence Speed', fontsize=13, fontweight='bold')
        axes[0, col].grid(True, alpha=0.3)
        
        # Expert selection entropy vs noise
        entropies = []
        for noise in noise_levels:
            if noise == 0.0:
                key = f'{router_type}_bias_scaling'
            else:
                key = f'{router_type}_bias_scaling_noise_{noise:.2f}'
            
            if key in results:
                data = results[key]
                all_selections = np.concatenate([s.flatten() for s in data['logs']['expert_selections']])
                counts = np.bincount(all_selections, minlength=4)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
        
        axes[1, col].plot(noise_vals, entropies, 'o-', linewidth=2, markersize=10, color='orange')
        axes[1, col].axhline(np.log(4), color='red', linestyle='--', linewidth=2, label='Max entropy')
        axes[1, col].set_xlabel('Noise Level', fontsize=12)
        axes[1, col].set_ylabel('Expert Selection Entropy (nats)', fontsize=12)
        axes[1, col].set_title(f'{router_type.upper()}: Expert Diversity', fontsize=13, fontweight='bold')
        axes[1, col].legend(fontsize=10)
        axes[1, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/router_ablation/noise_effect_analysis.png', dpi=200, bbox_inches='tight')
    print("✓ Noise effect analysis → logs/router_ablation/noise_effect_analysis.png")
    plt.close()

def plot_bias_evolution_comparison(results):
    """Compare bias evolution across modes"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Bias Evolution: Baseline vs Bias+Scaling vs Bias+Scaling+Noise', fontsize=16, fontweight='bold')
    
    router_types = ['mlp', 'conv']
    
    for row, router_type in enumerate(router_types):
        # Baseline (no bias update)
        key = f'{router_type}_baseline'
        if key in results:
            data = results[key]
            bias_arr = data['logs']['bias_values']
            for i in range(bias_arr.shape[1]):
                axes[row, 0].plot(data['logs']['steps'], bias_arr[:, i], label=f'E{i}', linewidth=2)
            axes[row, 0].set_title(f'{router_type.upper()}: Baseline (frozen)', fontsize=12, fontweight='bold')
            axes[row, 0].set_xlabel('Step', fontsize=11)
            axes[row, 0].set_ylabel('Bias Value', fontsize=11)
            axes[row, 0].legend(fontsize=9)
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Bias + Scaling
        key = f'{router_type}_bias_scaling'
        if key in results:
            data = results[key]
            bias_arr = data['logs']['bias_values']
            for i in range(bias_arr.shape[1]):
                axes[row, 1].plot(data['logs']['steps'], bias_arr[:, i], label=f'E{i}', linewidth=2)
            axes[row, 1].set_title(f'{router_type.upper()}: Bias + Scaling', fontsize=12, fontweight='bold')
            axes[row, 1].set_xlabel('Step', fontsize=11)
            axes[row, 1].set_ylabel('Bias Value', fontsize=11)
            axes[row, 1].legend(fontsize=9)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Bias + Scaling + Noise (0.1)
        key = f'{router_type}_bias_scaling_noise_0.10'
        if key in results:
            data = results[key]
            bias_arr = data['logs']['bias_values']
            for i in range(bias_arr.shape[1]):
                axes[row, 2].plot(data['logs']['steps'], bias_arr[:, i], label=f'E{i}', linewidth=2)
            axes[row, 2].set_title(f'{router_type.upper()}: Bias + Scaling + Noise 0.1', fontsize=12, fontweight='bold')
            axes[row, 2].set_xlabel('Step', fontsize=11)
            axes[row, 2].set_ylabel('Bias Value', fontsize=11)
            axes[row, 2].legend(fontsize=9)
            axes[row, 2].grid(True, alpha=0.3)
            axes[row, 2].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('logs/router_ablation/bias_evolution_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Bias evolution comparison → logs/router_ablation/bias_evolution_comparison.png")
    plt.close()

def plot_mlp_vs_conv(results):
    """Direct comparison: MLP vs Conv router"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MLP vs Conv Router Comparison', fontsize=16, fontweight='bold')
    
    modes = [
        ('baseline', 'Baseline (no bias)'),
        ('bias_scaling', 'Bias + Scaling'),
        ('bias_scaling_noise_0.10', 'Bias + Scaling + Noise 0.1'),
        ('bias_scaling_noise_0.50', 'Bias + Scaling + Noise 0.5'),
    ]
    
    # Convergence speed
    mlp_steps = []
    conv_steps = []
    mode_labels = []
    
    for mode, label in modes:
        mlp_key = f'mlp_{mode}'
        conv_key = f'conv_{mode}'
        
        if mlp_key in results and conv_key in results:
            mlp_steps.append(results[mlp_key]['config']['steps'])
            conv_steps.append(results[conv_key]['config']['steps'])
            mode_labels.append(label)
    
    x = np.arange(len(mode_labels))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, mlp_steps, width, label='MLP', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, conv_steps, width, label='Conv', color='orange', alpha=0.8)
    axes[0, 0].set_xlabel('Mode', fontsize=12)
    axes[0, 0].set_ylabel('Steps to Converge', fontsize=12)
    axes[0, 0].set_title('Convergence Speed', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(mode_labels, rotation=15, ha='right', fontsize=9)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Final loss
    mlp_loss = []
    conv_loss = []
    
    for mode, label in modes:
        mlp_key = f'mlp_{mode}'
        conv_key = f'conv_{mode}'
        
        if mlp_key in results and conv_key in results:
            mlp_loss.append(results[mlp_key]['config']['final_loss'])
            conv_loss.append(results[conv_key]['config']['final_loss'])
    
    axes[0, 1].bar(x - width/2, mlp_loss, width, label='MLP', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x + width/2, conv_loss, width, label='Conv', color='orange', alpha=0.8)
    axes[0, 1].set_xlabel('Mode', fontsize=12)
    axes[0, 1].set_ylabel('Final Loss', fontsize=12)
    axes[0, 1].set_title('Final Loss', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(mode_labels, rotation=15, ha='right', fontsize=9)
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Expert selection entropy
    mlp_entropy = []
    conv_entropy = []
    
    for mode, label in modes:
        mlp_key = f'mlp_{mode}'
        conv_key = f'conv_{mode}'
        
        if mlp_key in results and conv_key in results:
            for key, entropy_list in [(mlp_key, mlp_entropy), (conv_key, conv_entropy)]:
                data = results[key]
                all_selections = np.concatenate([s.flatten() for s in data['logs']['expert_selections']])
                counts = np.bincount(all_selections, minlength=4)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropy_list.append(entropy)
    
    axes[1, 0].bar(x - width/2, mlp_entropy, width, label='MLP', color='steelblue', alpha=0.8)
    axes[1, 0].bar(x + width/2, conv_entropy, width, label='Conv', color='orange', alpha=0.8)
    axes[1, 0].axhline(np.log(4), color='red', linestyle='--', linewidth=2, label='Max entropy')
    axes[1, 0].set_xlabel('Mode', fontsize=12)
    axes[1, 0].set_ylabel('Entropy (nats)', fontsize=12)
    axes[1, 0].set_title('Expert Selection Entropy', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(mode_labels, rotation=15, ha='right', fontsize=9)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Bias range (final)
    mlp_bias_range = []
    conv_bias_range = []
    
    for mode, label in modes:
        mlp_key = f'mlp_{mode}'
        conv_key = f'conv_{mode}'
        
        if mlp_key in results and conv_key in results:
            for key, range_list in [(mlp_key, mlp_bias_range), (conv_key, conv_bias_range)]:
                data = results[key]
                final_bias = data['logs']['bias_values'][-1]
                bias_range = final_bias.max() - final_bias.min()
                range_list.append(bias_range)
    
    axes[1, 1].bar(x - width/2, mlp_bias_range, width, label='MLP', color='steelblue', alpha=0.8)
    axes[1, 1].bar(x + width/2, conv_bias_range, width, label='Conv', color='orange', alpha=0.8)
    axes[1, 1].set_xlabel('Mode', fontsize=12)
    axes[1, 1].set_ylabel('Bias Range (max - min)', fontsize=12)
    axes[1, 1].set_title('Final Bias Spread', fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(mode_labels, rotation=15, ha='right', fontsize=9)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('logs/router_ablation/mlp_vs_conv.png', dpi=200, bbox_inches='tight')
    print("✓ MLP vs Conv comparison → logs/router_ablation/mlp_vs_conv.png")
    plt.close()

def print_summary_table(results):
    """Print summary table of all ablations"""
    print("\n" + "="*120)
    print("ABLATION SUMMARY")
    print("="*120)
    print(f"{'Ablation':<40} {'Router':<8} {'Noise':<8} {'Bias':<8} {'Steps':<10} {'Final Loss':<15} {'Converged':<10}")
    print("-"*120)
    
    for name in sorted(results.keys()):
        config = results[name]['config']
        print(f"{name:<40} {config['router_type']:<8} {config['router_noise']:<8.2f} "
              f"{'Yes' if config['update_bias'] else 'No':<8} {config['steps']:<10} "
              f"{config['final_loss']:<15.2e} {'Yes' if config.get('converged', False) else 'No':<10}")
    
    print("="*120)

def main():
    """Run all analyses"""
    print("Loading ablation results...")
    results = load_all_ablations()
    print(f"Loaded {len(results)} ablations\n")
    
    if len(results) == 0:
        print("No ablation results found. Run train_router_ablation.py first.")
        return
    
    print_summary_table(results)
    
    print("\nGenerating visualizations...")
    plot_convergence_comparison(results)
    plot_noise_effect(results)
    plot_bias_evolution_comparison(results)
    plot_mlp_vs_conv(results)
    
    print("\n" + "="*80)
    print("Analysis complete. Check logs/router_ablation/ for plots.")
    print("="*80)

if __name__ == '__main__':
    main()
