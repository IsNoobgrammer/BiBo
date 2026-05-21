"""
Sorting task plots — model quality comparison visualizations.
"""
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import PLOTS_DIR, BIBO_COLOR, QWEN_COLOR, setup_style

__all__ = ['plot_results']


def plot_results(all_results):
    """Generate comparison plots from collected results."""
    setup_style()

    seq_lens = sorted(all_results['bibo'].keys())

    # --- Plot 1: Loss vs Sequence Length ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bibo_losses = [all_results['bibo'][sl]['loss'] for sl in seq_lens]
    qwen_losses = [all_results['qwen3moe'][sl]['loss'] for sl in seq_lens]

    ax = axes[0]
    ax.plot(seq_lens, bibo_losses, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(seq_lens, qwen_losses, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Loss vs Sequence Length', fontweight='bold')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)

    # --- Plot 2: Token Accuracy vs Sequence Length ---
    bibo_acc = [all_results['bibo'][sl]['token_accuracy'] for sl in seq_lens]
    qwen_acc = [all_results['qwen3moe'][sl]['token_accuracy'] for sl in seq_lens]

    ax = axes[1]
    ax.plot(seq_lens, bibo_acc, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(seq_lens, qwen_acc, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Token Accuracy vs Sequence Length', fontweight='bold')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)
    ax.set_ylim(0, 1.05)

    # --- Plot 3: Full Sequence Accuracy ---
    bibo_full = [all_results['bibo'][sl]['full_sequence_accuracy'] for sl in seq_lens]
    qwen_full = [all_results['qwen3moe'][sl]['full_sequence_accuracy'] for sl in seq_lens]

    ax = axes[2]
    ax.plot(seq_lens, bibo_full, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(seq_lens, qwen_full, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Full Sequence Accuracy')
    ax.set_title('Full Sequence Accuracy\n(entire sorted output correct)', fontweight='bold')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)
    ax.set_ylim(0, 1.05)

    plt.suptitle('Model Quality — BiBo vs Qwen3MoE (Sorting Task)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_quality_comparison.png'))
    plt.close()
    print(f"  [OK] model_quality_comparison.png")

    # --- Plot 4: Confidence Calibration ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bibo_conf_correct = [all_results['bibo'][sl]['mean_confidence_correct'] for sl in seq_lens]
    bibo_conf_wrong = [all_results['bibo'][sl]['mean_confidence_wrong'] for sl in seq_lens]
    qwen_conf_correct = [all_results['qwen3moe'][sl]['mean_confidence_correct'] for sl in seq_lens]
    qwen_conf_wrong = [all_results['qwen3moe'][sl]['mean_confidence_wrong'] for sl in seq_lens]

    ax = axes[0]
    ax.plot(seq_lens, bibo_conf_correct, 'o-', color=BIBO_COLOR, linewidth=2, label='BiBo (correct)')
    ax.plot(seq_lens, bibo_conf_wrong, 'o--', color=BIBO_COLOR, linewidth=1.5, alpha=0.6, label='BiBo (wrong)')
    ax.plot(seq_lens, qwen_conf_correct, 's-', color=QWEN_COLOR, linewidth=2, label='Qwen (correct)')
    ax.plot(seq_lens, qwen_conf_wrong, 's--', color=QWEN_COLOR, linewidth=1.5, alpha=0.6, label='Qwen (wrong)')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mean Top-1 Probability')
    ax.set_title('Confidence: Correct vs Wrong Predictions', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)

    # --- Plot 5: Position-wise accuracy for select seq_lens ---
    ax = axes[1]
    for sl in [64, 96, 128, 192]:
        if sl in all_results['bibo'] and all_results['bibo'][sl]['position_accuracy']:
            pos_acc = all_results['bibo'][sl]['position_accuracy']
            positions = np.linspace(0, 1, len(pos_acc))
            ax.plot(positions, pos_acc, '-', linewidth=1.5, label=f'BiBo seq={sl}')
    for sl in [64, 96, 128, 192]:
        if sl in all_results['qwen3moe'] and all_results['qwen3moe'][sl]['position_accuracy']:
            pos_acc = all_results['qwen3moe'][sl]['position_accuracy']
            positions = np.linspace(0, 1, len(pos_acc))
            ax.plot(positions, pos_acc, '--', linewidth=1.5, label=f'Qwen seq={sl}')
    ax.set_xlabel('Relative Position in Sorted Output')
    ax.set_ylabel('Accuracy')
    ax.set_title('Position-wise Accuracy\n(0=first sorted token, 1=last)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_confidence_position.png'))
    plt.close()
    print(f"  [OK] model_confidence_position.png")

    # --- Plot 6: Length generalization ---
    fig, ax = plt.subplots(figsize=(10, 5))
    trained_lens = [2, 8, 32, 64, 128, 256]

    x = np.arange(len(seq_lens))
    width = 0.35

    bibo_bars = [all_results['bibo'][sl]['token_accuracy'] for sl in seq_lens]
    qwen_bars = [all_results['qwen3moe'][sl]['token_accuracy'] for sl in seq_lens]

    bars1 = ax.bar(x - width/2, bibo_bars, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    bars2 = ax.bar(x + width/2, qwen_bars, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')

    for i, sl in enumerate(seq_lens):
        if sl not in trained_lens:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Length Generalization\n(red background = NOT in training data)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_length_generalization.png'))
    plt.close()
    print(f"  [OK] model_length_generalization.png")
