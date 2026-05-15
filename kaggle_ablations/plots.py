"""
Generate all plots from training + post-training metrics.
Run AFTER extract_metrics.py.

Usage: python kaggle_ablations/plots.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)
COLORS = {'bibo': '#2196F3', 'qwen3moe': '#F44336'}


def load_training_metrics():
    bibo = json.load(open(os.path.join(METRICS_DIR, 'bibo_metrics.json')))
    qwen = json.load(open(os.path.join(METRICS_DIR, 'qwen3moe_metrics.json')))
    return bibo, qwen


def load_post_metrics():
    return json.load(open(os.path.join(METRICS_DIR, 'post_training_metrics.json')))


# ============================================================
# Training Plots
# ============================================================

def plot_loss_curves(bibo, qwen):
    """Train + val loss per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    b_epochs = [e['epoch'] for e in bibo['epochs']]
    q_epochs = [e['epoch'] for e in qwen['epochs']]
    
    axes[0].plot(b_epochs, [e['train_loss'] for e in bibo['epochs']], '-o', color=COLORS['bibo'], label='BiBo', markersize=4)
    axes[0].plot(q_epochs, [e['train_loss'] for e in qwen['epochs']], '-s', color=COLORS['qwen3moe'], label='Qwen3MoE', markersize=4)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss')
    axes[0].legend()
    
    axes[1].plot(b_epochs, [e['val_loss'] for e in bibo['epochs']], '-o', color=COLORS['bibo'], label='BiBo', markersize=4)
    axes[1].plot(q_epochs, [e['val_loss'] for e in qwen['epochs']], '-s', color=COLORS['qwen3moe'], label='Qwen3MoE', markersize=4)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].set_title('Validation Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_loss_curves.png")


def plot_accuracy(bibo, qwen):
    """Val accuracy per epoch."""
    fig, ax = plt.subplots(figsize=(10, 5))
    b_epochs = [e['epoch'] for e in bibo['epochs']]
    q_epochs = [e['epoch'] for e in qwen['epochs']]
    
    ax.plot(b_epochs, [e['val_acc']*100 for e in bibo['epochs']], '-o', color=COLORS['bibo'], label='BiBo', linewidth=2, markersize=5)
    ax.plot(q_epochs, [e['val_acc']*100 for e in qwen['epochs']], '-s', color=COLORS['qwen3moe'], label='Qwen3MoE', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Validation Accuracy')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_accuracy.png")


def plot_step_loss(bibo, qwen):
    """Smoothed step loss."""
    fig, ax = plt.subplots(figsize=(14, 5))
    window = 25
    
    b_losses = [s['loss'] for s in bibo['steps']]
    q_losses = [s['loss'] for s in qwen['steps']]
    
    b_smooth = np.convolve(b_losses, np.ones(window)/window, mode='valid')
    q_smooth = np.convolve(q_losses, np.ones(window)/window, mode='valid')
    
    ax.plot(b_smooth, color=COLORS['bibo'], alpha=0.85, label='BiBo', linewidth=1.2)
    ax.plot(q_smooth, color=COLORS['qwen3moe'], alpha=0.85, label='Qwen3MoE', linewidth=1.2)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title(f'Step Loss (smoothed, window={window})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_step_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_step_loss.png")


def plot_lr_schedule(bibo, qwen):
    """LR schedule over training."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    b_lrs = [s['lr'] for s in bibo['steps']]
    q_lrs = [s['lr'] for s in qwen['steps']]
    
    ax.plot(b_lrs, color=COLORS['bibo'], label='BiBo', linewidth=1.5)
    ax.plot(q_lrs, color=COLORS['qwen3moe'], label='Qwen3MoE', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Step'); ax.set_ylabel('Learning Rate'); ax.set_title('LR Schedule (Cosine + Warmup)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_lr_schedule.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_lr_schedule.png")


# ============================================================
# Router Plots
# ============================================================

def plot_router_heatmap(post_metrics):
    """Expert selection distribution heatmap per layer."""
    for model_name in ['bibo', 'qwen3moe']:
        router = post_metrics[model_name]['router']
        layers = sorted(router.keys())
        n_experts = len(router[layers[0]]['expert_distribution'])
        
        dist_matrix = np.array([router[l]['expert_distribution'] for l in layers])
        
        fig, ax = plt.subplots(figsize=(max(8, n_experts * 0.9), max(4, len(layers) * 0.7)))
        sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'E{i}' for i in range(n_experts)],
                    yticklabels=layers, ax=ax, vmin=0, linewidths=0.5)
        ax.set_xlabel('Expert'); ax.set_ylabel('Layer')
        ax.set_title(f'{model_name.upper()} — Expert Selection Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'05_router_heatmap_{model_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print("Saved: 05_router_heatmap_bibo.png, 05_router_heatmap_qwen3moe.png")


def plot_router_confidence(post_metrics):
    """Confidence (inverse entropy) per layer — both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model_name in enumerate(['bibo', 'qwen3moe']):
        router = post_metrics[model_name]['router']
        layers = sorted(router.keys())
        
        confidences = [router[l]['inverse_entropy'] for l in layers]
        entropies = [router[l]['normalized_entropy'] for l in layers]
        
        x = np.arange(len(layers))
        bars = axes[idx].bar(x, confidences, color=COLORS[model_name], alpha=0.8, edgecolor='black', linewidth=0.5)
        axes[idx].set_xticks(x); axes[idx].set_xticklabels(layers, rotation=30, fontsize=8)
        axes[idx].set_ylabel('Confidence (1 - norm_entropy)')
        axes[idx].set_title(f'{model_name.upper()} — Router Confidence per Layer')
        axes[idx].set_ylim(0, 1)
        axes[idx].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% confidence')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_router_confidence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_router_confidence.png")


def plot_load_balance(post_metrics):
    """Load balance comparison — std of expert distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for model_name in ['bibo', 'qwen3moe']:
        router = post_metrics[model_name]['router']
        layers = sorted(router.keys())
        stds = [router[l]['load_balance_std'] for l in layers]
        ax.plot(range(len(layers)), stds, '-o', color=COLORS[model_name], label=model_name, markersize=6, linewidth=2)
    
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(sorted(post_metrics['bibo']['router'].keys()), rotation=30, fontsize=8)
    ax.set_ylabel('Std of Expert Distribution (lower = more balanced)')
    ax.set_title('Load Balance Across Layers')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_load_balance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_load_balance.png")


# ============================================================
# Timing Plot
# ============================================================

def plot_timing(post_metrics):
    """Forward pass latency comparison."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    models = ['BiBo', 'Qwen3MoE']
    avgs = [post_metrics['bibo']['timing']['avg_ms'], post_metrics['qwen3moe']['timing']['avg_ms']]
    stds = [post_metrics['bibo']['timing']['std_ms'], post_metrics['qwen3moe']['timing']['std_ms']]
    tps = [post_metrics['bibo']['timing']['tok_per_sec'], post_metrics['qwen3moe']['timing']['tok_per_sec']]
    
    bars = ax.bar(models, avgs, yerr=stds, capsize=8,
                  color=[COLORS['bibo'], COLORS['qwen3moe']], edgecolor='black', linewidth=0.5)
    for bar, avg, t in zip(bars, avgs, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[models.index(bar.get_label() if hasattr(bar, 'get_label') else '')] + 1,
                f'{avg:.1f}ms\n{t:,.0f} tok/s', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Forward Pass (ms)'); ax.set_title('Inference Latency (lower = better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_timing.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_timing.png")


# ============================================================
# Summary Plot
# ============================================================

def plot_summary(bibo_train, qwen_train, post_metrics):
    """Single summary figure with key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Final metrics
    b_final = bibo_train['epochs'][-1]
    q_final = qwen_train['epochs'][-1]
    
    # Top-left: final val loss + acc
    metrics = ['Val Loss', 'Val Acc (%)']
    bibo_vals = [b_final['val_loss'], b_final['val_acc'] * 100]
    qwen_vals = [q_final['val_loss'], q_final['val_acc'] * 100]
    
    x = np.arange(len(metrics))
    w = 0.35
    axes[0, 0].bar(x - w/2, bibo_vals, w, label='BiBo', color=COLORS['bibo'])
    axes[0, 0].bar(x + w/2, qwen_vals, w, label='Qwen3MoE', color=COLORS['qwen3moe'])
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend(); axes[0, 0].set_title('Final Metrics')
    
    # Top-right: timing
    axes[0, 1].bar(['BiBo', 'Qwen3MoE'],
                   [post_metrics['bibo']['timing']['avg_ms'], post_metrics['qwen3moe']['timing']['avg_ms']],
                   color=[COLORS['bibo'], COLORS['qwen3moe']])
    axes[0, 1].set_ylabel('ms'); axes[0, 1].set_title('Forward Pass Latency')
    
    # Bottom-left: val loss curve
    axes[1, 0].plot([e['epoch'] for e in bibo_train['epochs']], [e['val_loss'] for e in bibo_train['epochs']],
                    '-o', color=COLORS['bibo'], label='BiBo', markersize=3)
    axes[1, 0].plot([e['epoch'] for e in qwen_train['epochs']], [e['val_loss'] for e in qwen_train['epochs']],
                    '-s', color=COLORS['qwen3moe'], label='Qwen3MoE', markersize=3)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss'); axes[1, 0].set_title('Val Loss Curve')
    axes[1, 0].legend()
    
    # Bottom-right: router confidence (BiBo only)
    bibo_router = post_metrics['bibo']['router']
    layers = sorted(bibo_router.keys())
    confs = [bibo_router[l]['inverse_entropy'] for l in layers]
    axes[1, 1].bar(range(len(layers)), confs, color=COLORS['bibo'], alpha=0.8)
    axes[1, 1].set_xticks(range(len(layers))); axes[1, 1].set_xticklabels(layers, rotation=30, fontsize=7)
    axes[1, 1].set_ylabel('Confidence'); axes[1, 1].set_title('BiBo Router Confidence')
    axes[1, 1].set_ylim(0, 1)
    
    plt.suptitle('BiBo vs Qwen3MoE — Ablation Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '00_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 00_summary.png")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading metrics...")
    bibo_train, qwen_train = load_training_metrics()
    post_metrics = load_post_metrics()
    
    print("\nGenerating plots...")
    plot_loss_curves(bibo_train, qwen_train)
    plot_accuracy(bibo_train, qwen_train)
    plot_step_loss(bibo_train, qwen_train)
    plot_lr_schedule(bibo_train, qwen_train)
    plot_router_heatmap(post_metrics)
    plot_router_confidence(post_metrics)
    plot_load_balance(post_metrics)
    plot_timing(post_metrics)
    plot_summary(bibo_train, qwen_train, post_metrics)
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
