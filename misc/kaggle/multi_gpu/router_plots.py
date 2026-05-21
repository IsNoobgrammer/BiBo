"""
All plot functions for router analysis — BiBo vs Qwen3MoE.

Imports shared utilities from plot_utils.py and metrics from router_metrics.py.
"""
import os
import torch
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from plot_utils import (
    PLOTS_DIR,
    BIBO_COLOR, QWEN_COLOR, BIBO_CMAP, QWEN_CMAP,
    POLYGLU_COLORS, SPECIAL_COLORS,
    get_expert_layout, get_expert_colors, get_expert_labels, get_expert_types,
)
from model_utils import CFG, extract_routing_data
from router_metrics import (
    compute_gini,
    compute_load_balance_metrics,
    compute_expert_coselection,
    compute_specialization_score,
)

__all__ = [
    'plot_expert_usage_sweep',
    'plot_comparative_usage',
    'plot_confidence_distribution',
    'plot_coselection_matrix',
    'plot_position_type_routing',
    'plot_specialization_radar',
    'plot_token_expert_heatmap_v2',
    'plot_confidence_evolution_comparative',
    'plot_entropy_evolution_comparative',
    'plot_load_balance_summary',
    'plot_weight_rank_distribution',
    'plot_routing_diversity',
    'plot_routing_stability',
    'plot_expert_type_analysis',
    'plot_special_expert_analysis',
    'plot_expert_switching_rate',
    'plot_grand_summary',
    'plot_per_layer_weight_kde',
]


# ============================================================
# PLOT 1: Expert Usage Heatmap — Multi-batch sweep
# ============================================================

def plot_expert_usage_sweep(all_data, model_name, n_experts, batch_sizes, seq_lens, bibo_cfg=None):
    """Grid plot: rows=seq_lens, cols=batch_sizes. Each cell = expert usage bar chart."""
    labels = get_expert_labels(n_experts, model_name, bibo_cfg)
    colors = get_expert_colors(n_experts, model_name, bibo_cfg)

    n_rows = len(seq_lens)
    n_cols = len(batch_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows), squeeze=False)

    for row, sl in enumerate(seq_lens):
        for col, bs in enumerate(batch_sizes):
            ax = axes[row, col]
            key = (sl, bs)
            if key not in all_data:
                ax.set_visible(False)
                continue

            data = all_data[key]
            counts = np.zeros(n_experts)
            for layer_idx, ld in data.items():
                idx = ld['indices'].numpy().flatten()
                counts += np.bincount(idx, minlength=n_experts)

            dist = counts / (counts.sum() + 1e-10)
            ax.bar(range(n_experts), dist, color=colors, edgecolor='white', linewidth=0.5)
            ax.axhline(y=1/n_experts, color='red', linestyle='--', alpha=0.5, linewidth=1)

            metrics = compute_load_balance_metrics(
                np.concatenate([ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1])
                               for ld in data.values()]),
                n_experts, CFG['bibo']['num_experts_per_tok'] if model_name == 'BiBo' else CFG['qwen3moe']['num_experts_per_tok']
            )
            ax.text(0.98, 0.95, f"Gini={metrics['gini']:.3f}\nH={metrics['normalized_entropy']:.3f}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            if row == 0:
                ax.set_title(f'batch={bs}', fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'seq={sl}\nFraction')
            ax.set_xticks(range(n_experts))
            ax.set_xticklabels(labels, rotation=45, fontsize=6)
            ax.set_ylim(0, max(dist.max() * 1.3, 2/n_experts))

    fig.suptitle(f'{model_name} — Expert Usage Across Batch Sizes & Sequence Lengths',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'usage_sweep_{model_name}.png'))
    plt.close()
    print(f"  ✓ usage_sweep_{model_name}.png")


# ============================================================
# PLOT 2: Side-by-side comparative expert usage
# ============================================================

def plot_comparative_usage(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len, batch_size, bibo_cfg=None):
    """Side-by-side bar charts comparing BiBo vs Qwen expert usage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bibo_labels = get_expert_labels(n_exp_bibo, 'BiBo', bibo_cfg)
    bibo_colors = get_expert_colors(n_exp_bibo, 'BiBo', bibo_cfg)
    bibo_counts = np.zeros(n_exp_bibo)
    for ld in bibo_data.values():
        bibo_counts += np.bincount(ld['indices'].numpy().flatten(), minlength=n_exp_bibo)
    bibo_dist = bibo_counts / (bibo_counts.sum() + 1e-10)

    ax1.bar(range(n_exp_bibo), bibo_dist, color=bibo_colors, edgecolor='white', linewidth=0.8)
    ax1.axhline(y=1/n_exp_bibo, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax1.set_xticks(range(n_exp_bibo))
    ax1.set_xticklabels(bibo_labels, rotation=45)
    ax1.set_ylabel('Token Fraction')
    ax1.set_title('BiBo (PolyGLU + Conv Router)', color=BIBO_COLOR, fontweight='bold')
    ax1.legend()

    qwen_labels = get_expert_labels(n_exp_qwen, 'Qwen3MoE')
    qwen_colors = get_expert_colors(n_exp_qwen, 'Qwen3MoE')
    qwen_counts = np.zeros(n_exp_qwen)
    for ld in qwen_data.values():
        qwen_counts += np.bincount(ld['indices'].numpy().flatten(), minlength=n_exp_qwen)
    qwen_dist = qwen_counts / (qwen_counts.sum() + 1e-10)

    ax2.bar(range(n_exp_qwen), qwen_dist, color=qwen_colors, edgecolor='white', linewidth=0.8)
    ax2.axhline(y=1/n_exp_qwen, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax2.set_xticks(range(n_exp_qwen))
    ax2.set_xticklabels(qwen_labels, rotation=45)
    ax2.set_ylabel('Token Fraction')
    ax2.set_title('Qwen3MoE (Softmax Router)', color=QWEN_COLOR, fontweight='bold')
    ax2.legend()

    bibo_metrics = compute_load_balance_metrics(
        np.concatenate([ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in bibo_data.values()]),
        n_exp_bibo, CFG['bibo']['num_experts_per_tok']
    )
    qwen_metrics = compute_load_balance_metrics(
        np.concatenate([ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in qwen_data.values()]),
        n_exp_qwen, CFG['qwen3moe']['num_experts_per_tok']
    )

    fig.text(0.5, -0.02,
             f"BiBo: Gini={bibo_metrics['gini']:.3f} | H_norm={bibo_metrics['normalized_entropy']:.3f} | CV={bibo_metrics['cv']:.3f}    "
             f"Qwen: Gini={qwen_metrics['gini']:.3f} | H_norm={qwen_metrics['normalized_entropy']:.3f} | CV={qwen_metrics['cv']:.3f}",
             ha='center', fontsize=10, style='italic')

    plt.suptitle(f'Expert Usage Comparison — seq={seq_len}, batch={batch_size}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'comparative_usage_seq{seq_len}_bs{batch_size}.png'))
    plt.close()
    print(f"  ✓ comparative_usage_seq{seq_len}_bs{batch_size}.png")


# ============================================================
# PLOT 3: Confidence distribution (violin + box)
# ============================================================

def plot_confidence_distribution(bibo_data, qwen_data, seq_len):
    """Violin plots of top-1 and top-k weight distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bibo_top1, qwen_top1 = [], []
    bibo_std, qwen_std = [], []

    for ld in bibo_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        bibo_top1.extend(w[:, 0].tolist())
        bibo_std.extend(w.std(axis=1).tolist())

    for ld in qwen_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        qwen_top1.extend(w[:, 0].tolist())
        qwen_std.extend(w.std(axis=1).tolist())

    ax = axes[0]
    parts = ax.violinplot([bibo_top1, qwen_top1], positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([BIBO_COLOR, QWEN_COLOR][i])
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BiBo', 'Qwen3MoE'])
    ax.set_ylabel('Top-1 Routing Weight')
    ax.set_title('Top-1 Expert Confidence', fontweight='bold')
    ax.axhline(y=1/CFG['bibo']['num_experts_per_tok'], color='gray', linestyle='--', alpha=0.5,
               label=f'Uniform (1/top_k)')
    ax.legend()

    ax = axes[1]
    ax.hist(bibo_std, bins=50, alpha=0.7, color=BIBO_COLOR, label='BiBo', density=True)
    ax.hist(qwen_std, bins=50, alpha=0.7, color=QWEN_COLOR, label='Qwen3MoE', density=True)
    ax.set_xlabel('Std of Top-K Weights (per token)')
    ax.set_ylabel('Density')
    ax.set_title('Weight Spread — Lower = More Uniform Expert Contribution', fontweight='bold')
    ax.legend()

    plt.suptitle(f'Routing Confidence Analysis — seq={seq_len}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'confidence_distribution_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ confidence_distribution_seq{seq_len}.png")


# ============================================================
# PLOT 4: Expert co-selection heatmap
# ============================================================

def plot_coselection_matrix(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len, bibo_cfg=None):
    """Heatmap showing which expert pairs are frequently co-selected."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bibo_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in bibo_data.values()
    ])
    bibo_cosel = compute_expert_coselection(bibo_indices, n_exp_bibo)
    bibo_labels = get_expert_labels(n_exp_bibo, 'BiBo', bibo_cfg)

    mask = np.triu(np.ones_like(bibo_cosel, dtype=bool), k=1)
    sns.heatmap(bibo_cosel, mask=~mask, ax=ax1, cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=bibo_labels, yticklabels=bibo_labels, square=True,
                cbar_kws={'label': 'Co-selection freq', 'shrink': 0.8})
    ax1.set_title('BiBo — Expert Co-Selection', color=BIBO_COLOR, fontweight='bold')

    qwen_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in qwen_data.values()
    ])
    qwen_cosel = compute_expert_coselection(qwen_indices, n_exp_qwen)
    qwen_labels = get_expert_labels(n_exp_qwen, 'Qwen3MoE')

    mask = np.triu(np.ones_like(qwen_cosel, dtype=bool), k=1)
    sns.heatmap(qwen_cosel, mask=~mask, ax=ax2, cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=qwen_labels, yticklabels=qwen_labels, square=True,
                cbar_kws={'label': 'Co-selection freq', 'shrink': 0.8})
    ax2.set_title('Qwen3MoE — Expert Co-Selection', color=QWEN_COLOR, fontweight='bold')

    plt.suptitle(f'Expert Co-Selection Patterns — seq={seq_len}\n'
                 '(Which experts are picked together for same token)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'coselection_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ coselection_seq{seq_len}.png")


# ============================================================
# PLOT 5: Per-position-type routing (unsorted vs sorted tokens)
# ============================================================

def plot_position_type_routing(layer_data, model_name, n_experts, seq_len, bibo_cfg=None):
    """Compare routing for different token regions (input vs output)."""
    labels = get_expert_labels(n_experts, model_name, bibo_cfg)
    colors = get_expert_colors(n_experts, model_name, bibo_cfg)

    task = CFG['training'].get('task', 'sort')
    half = seq_len
    region_names = ('Unsorted (input)', 'Sorted (output)') if task == 'sort' else ('Phase1 (input)', 'Phase2+3 (output)')

    layers = sorted(layer_data.keys())
    fig, axes = plt.subplots(len(layers), 2, figsize=(12, 3*len(layers)), squeeze=False)

    for row, l in enumerate(layers):
        idx = layer_data[l]['indices'].numpy()
        if idx.ndim == 3:
            unsorted_idx = idx[:, :half, :].reshape(-1)
            sorted_idx = idx[:, half:, :].reshape(-1)
        else:
            unsorted_idx = idx[:half, :].flatten()
            sorted_idx = idx[half:, :].flatten()

        ax = axes[row, 0]
        counts_u = np.bincount(unsorted_idx, minlength=n_experts).astype(float)
        dist_u = counts_u / (counts_u.sum() + 1e-10)
        ax.bar(range(n_experts), dist_u, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=1/n_experts, color='red', linestyle='--', alpha=0.4)
        ax.set_title(f'L{l} — {region_names[0]}' if row == 0 else f'L{l}')
        ax.set_ylim(0, max(dist_u.max() * 1.3, 2/n_experts))
        if row == len(layers) - 1:
            ax.set_xticks(range(n_experts))
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
        else:
            ax.set_xticks([])

        ax = axes[row, 1]
        counts_s = np.bincount(sorted_idx, minlength=n_experts).astype(float)
        dist_s = counts_s / (counts_s.sum() + 1e-10)
        ax.bar(range(n_experts), dist_s, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=1/n_experts, color='red', linestyle='--', alpha=0.4)
        ax.set_title(f'L{l} — {region_names[1]}' if row == 0 else f'L{l}')
        ax.set_ylim(0, max(dist_s.max() * 1.3, 2/n_experts))
        if row == len(layers) - 1:
            ax.set_xticks(range(n_experts))
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
        else:
            ax.set_xticks([])

    fig.suptitle(f'{model_name} — Routing by Token Type ({region_names[0]} vs {region_names[1]})\nseq_len={seq_len}',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'position_type_routing_{model_name}_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ position_type_routing_{model_name}_seq{seq_len}.png")


# ============================================================
# PLOT 6: Expert specialization radar chart
# ============================================================

def plot_specialization_radar(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len, bibo_cfg=None):
    """Radar chart showing per-expert position specialization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(polar=True))

    bibo_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in bibo_data.values()
    ])
    first_layer = list(bibo_data.values())[0]
    sl = first_layer['indices'].shape[1] if first_layer['indices'].ndim == 3 else first_layer['indices'].shape[0]
    bibo_spec = compute_specialization_score(bibo_indices, n_exp_bibo, sl)
    bibo_labels = get_expert_labels(n_exp_bibo, 'BiBo', bibo_cfg)

    angles = np.linspace(0, 2*np.pi, n_exp_bibo, endpoint=False).tolist()
    angles += angles[:1]
    values = bibo_spec.tolist() + [bibo_spec[0]]

    ax1.plot(angles, values, 'o-', color=BIBO_COLOR, linewidth=2)
    ax1.fill(angles, values, alpha=0.25, color=BIBO_COLOR)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(bibo_labels, fontsize=7)
    ax1.set_title('BiBo — Position Specialization\n(KL from uniform)', fontsize=10, fontweight='bold', pad=20)

    qwen_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in qwen_data.values()
    ])
    first_layer_q = list(qwen_data.values())[0]
    sl_q = first_layer_q['indices'].shape[1] if first_layer_q['indices'].ndim == 3 else first_layer_q['indices'].shape[0]
    qwen_spec = compute_specialization_score(qwen_indices, n_exp_qwen, sl_q)
    qwen_labels = get_expert_labels(n_exp_qwen, 'Qwen3MoE')

    angles_q = np.linspace(0, 2*np.pi, n_exp_qwen, endpoint=False).tolist()
    angles_q += angles_q[:1]
    values_q = qwen_spec.tolist() + [qwen_spec[0]]

    ax2.plot(angles_q, values_q, 'o-', color=QWEN_COLOR, linewidth=2)
    ax2.fill(angles_q, values_q, alpha=0.25, color=QWEN_COLOR)
    ax2.set_xticks(angles_q[:-1])
    ax2.set_xticklabels(qwen_labels, fontsize=7)
    ax2.set_title('Qwen3MoE — Position Specialization\n(KL from uniform)', fontsize=10, fontweight='bold', pad=20)

    plt.suptitle(f'Expert Position Specialization — seq={seq_len}\n'
                 'Higher = expert prefers specific positions (more specialized)',
                 fontsize=11, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'specialization_radar_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ specialization_radar_seq{seq_len}.png")


# ============================================================
# PLOT 7: Token-level expert heatmap
# ============================================================

def plot_token_expert_heatmap_v2(layer_data, model_name, n_experts, seq_len_label, bibo_cfg=None):
    """Heatmap: x=token position, y=layer. Color = primary expert."""
    layers = sorted(layer_data.keys())
    n_layers = len(layers)
    first_ld = layer_data[layers[0]]

    if first_ld['indices'].ndim == 3:
        actual_seq = first_ld['indices'].shape[1]
        heatmap = np.zeros((n_layers, actual_seq))
        for i, l in enumerate(layers):
            heatmap[i] = layer_data[l]['indices'][0, :, 0].numpy()
    else:
        actual_seq = first_ld['indices'].shape[0]
        heatmap = np.zeros((n_layers, actual_seq))
        for i, l in enumerate(layers):
            heatmap[i] = layer_data[l]['indices'][:, 0].numpy()

    cmap = plt.cm.get_cmap('tab10', n_experts)

    fig, ax = plt.subplots(figsize=(min(18, max(10, actual_seq//8)), max(3, n_layers*0.8)))
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=-0.5, vmax=n_experts-0.5,
                   interpolation='nearest')

    half = int(seq_len_label)
    task = CFG['training'].get('task', 'sort')
    if task == 'arithmetic':
        sep_token = CFG['training']['vocab_size'] - 1
        third = actual_seq // 3
        ax.axvline(x=third, color='white', linewidth=2, linestyle='-')
        ax.axvline(x=2*third, color='white', linewidth=2, linestyle='--')
        ax.text(third/2, -0.8, 'Phase1 (input)', ha='center', fontsize=8, color='gray')
        ax.text(third + third/2, -0.8, 'Phase2 (resolved)', ha='center', fontsize=8, color='gray')
        ax.text(2*third + (actual_seq-2*third)/2, -0.8, 'P3', ha='center', fontsize=8, color='gray')
    elif half < actual_seq:
        ax.axvline(x=half, color='white', linewidth=2, linestyle='-')
        ax.text(half/2, -0.8, 'Unsorted', ha='center', fontsize=9, color='gray')
        ax.text(half + (actual_seq-half)/2, -0.8, 'Sorted', ha='center', fontsize=9, color='gray')

    ax.set_xlabel('Token Position')
    ax.set_ylabel('MoE Layer')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{l}' for l in layers])

    cbar = plt.colorbar(im, ax=ax, ticks=range(n_experts))
    labels = get_expert_labels(n_experts, model_name, bibo_cfg)
    cbar.set_ticklabels(labels)
    cbar.set_label('Expert')

    ax.set_title(f'{model_name} — Top-1 Expert per Token (seq_len={seq_len_label})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'expert_heatmap_v2_{model_name}_seq{seq_len_label}.png'))
    plt.close()
    print(f"  ✓ expert_heatmap_v2_{model_name}_seq{seq_len_label}.png")


# ============================================================
# PLOT 8: Confidence evolution (smoothed, comparative)
# ============================================================

def plot_confidence_evolution_comparative(bibo_data, qwen_data, seq_len):
    """Side-by-side smoothed confidence evolution over token positions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    bibo_layers = sorted(bibo_data.keys())
    for l in bibo_layers:
        w = bibo_data[l]['weights'].numpy()
        if w.ndim == 3:
            max_w = w[:, :, 0].mean(axis=0)
        else:
            max_w = w[:, 0]
        window = max(3, len(max_w) // 15)
        smoothed = np.convolve(max_w, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=f'L{l}', linewidth=1.8)

    ax1.axhline(y=1/CFG['bibo']['num_experts_per_tok'], color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Top-1 Weight')
    ax1.set_title('BiBo — Router Confidence Over Position', color=BIBO_COLOR, fontweight='bold')
    ax1.legend(ncol=2, fontsize=8)
    ax1.set_ylim(0, 1.05)

    qwen_layers = sorted(qwen_data.keys())
    for l in qwen_layers:
        w = qwen_data[l]['weights'].numpy()
        if w.ndim == 3:
            max_w = w[:, :, 0].mean(axis=0)
        else:
            max_w = w[:, 0]
        window = max(3, len(max_w) // 15)
        smoothed = np.convolve(max_w, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, label=f'L{l}', linewidth=1.8)

    ax2.axhline(y=1/CFG['qwen3moe']['num_experts_per_tok'], color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Top-1 Weight')
    ax2.set_title('Qwen3MoE — Router Confidence Over Position', color=QWEN_COLOR, fontweight='bold')
    ax2.legend(ncol=2, fontsize=8)
    ax2.set_ylim(0, 1.05)

    half = seq_len
    task = CFG['training'].get('task', 'sort')
    for ax in [ax1, ax2]:
        if task == 'arithmetic':
            first_ld = list(bibo_data.values())[0]
            actual_seq = first_ld['weights'].shape[1] if first_ld['weights'].ndim == 3 else first_ld['weights'].shape[0]
            third = actual_seq // 3
            ax.axvline(x=third, color='green', linestyle=':', alpha=0.6, label='SEP1')
            ax.axvline(x=2*third, color='orange', linestyle=':', alpha=0.6, label='SEP2')
        else:
            ax.axvline(x=half, color='green', linestyle=':', alpha=0.6, label='SEP')

    plt.suptitle(f'Confidence Evolution — seq={seq_len}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'confidence_evolution_comparative_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ confidence_evolution_comparative_seq{seq_len}.png")


# ============================================================
# PLOT 9: Entropy evolution (comparative)
# ============================================================

def plot_entropy_evolution_comparative(bibo_data, qwen_data, seq_len):
    """Per-token routing entropy over positions — BiBo vs Qwen."""
    fig, ax = plt.subplots(figsize=(14, 5))

    bibo_entropies = []
    for l, ld in sorted(bibo_data.items()):
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.mean(axis=0)
        w_norm = w / (w.sum(axis=1, keepdims=True) + 1e-10)
        entropy = -(w_norm * np.log(w_norm + 1e-10)).sum(axis=1)
        bibo_entropies.append(entropy)
    bibo_avg_entropy = np.mean(bibo_entropies, axis=0)

    qwen_entropies = []
    for l, ld in sorted(qwen_data.items()):
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.mean(axis=0)
        w_norm = w / (w.sum(axis=1, keepdims=True) + 1e-10)
        entropy = -(w_norm * np.log(w_norm + 1e-10)).sum(axis=1)
        qwen_entropies.append(entropy)
    qwen_avg_entropy = np.mean(qwen_entropies, axis=0)

    window = max(3, len(bibo_avg_entropy) // 15)
    bibo_smooth = np.convolve(bibo_avg_entropy, np.ones(window)/window, mode='valid')
    qwen_smooth = np.convolve(qwen_avg_entropy, np.ones(window)/window, mode='valid')

    ax.plot(bibo_smooth, color=BIBO_COLOR, linewidth=2.5, label='BiBo (avg across layers)')
    ax.plot(qwen_smooth, color=QWEN_COLOR, linewidth=2.5, label='Qwen3MoE (avg across layers)')

    top_k_bibo = CFG['bibo']['num_experts_per_tok']
    ax.axhline(y=np.log(top_k_bibo), color=BIBO_COLOR, linestyle='--', alpha=0.4,
               label=f'Max entropy (top-{top_k_bibo})')
    task = CFG['training'].get('task', 'sort')
    if task == 'arithmetic':
        actual_len = len(bibo_smooth)
        third = actual_len // 3
        ax.axvline(x=third, color='green', linestyle=':', alpha=0.6, label='SEP1')
        ax.axvline(x=2*third, color='orange', linestyle=':', alpha=0.6, label='SEP2')
    else:
        ax.axvline(x=seq_len, color='green', linestyle=':', alpha=0.6, label='SEP position')

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Routing Entropy (nats)')
    ax.set_title(f'Routing Entropy — Higher = More Uniform Expert Weighting (seq={seq_len})',
                 fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'entropy_comparative_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ entropy_comparative_seq{seq_len}.png")


# ============================================================
# PLOT 10: Load balance metrics summary
# ============================================================

def plot_load_balance_summary(all_metrics):
    """Summary bar chart comparing Gini, normalized entropy, CV."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = ['gini', 'normalized_entropy', 'cv']
    metric_labels = ['Gini Coefficient\n(lower = more balanced)',
                     'Normalized Entropy\n(higher = more balanced)',
                     'Coefficient of Variation\n(lower = more balanced)']

    for ax, mname, mlabel in zip(axes, metric_names, metric_labels):
        bibo_vals, qwen_vals, x_labels = [], [], []

        for key, metrics in sorted(all_metrics.items()):
            model, sl, bs = key
            if model == 'BiBo':
                bibo_vals.append(metrics[mname])
                x_labels.append(f's{sl}\nb{bs}')
            else:
                qwen_vals.append(metrics[mname])

        x = np.arange(len(bibo_vals))
        width = 0.35
        ax.bar(x - width/2, bibo_vals, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
        ax.bar(x + width/2, qwen_vals, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels[:len(bibo_vals)], fontsize=8)
        ax.set_ylabel(mlabel)
        ax.legend()

    plt.suptitle('Load Balance Metrics — BiBo vs Qwen3MoE', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'load_balance_summary.png'))
    plt.close()
    print(f"  ✓ load_balance_summary.png")


# ============================================================
# PLOT 11: Weight rank distribution
# ============================================================

def plot_weight_rank_distribution(bibo_data, qwen_data, seq_len):
    """How much weight goes to rank-1, rank-2, ..., rank-K expert?"""
    top_k = CFG['bibo']['num_experts_per_tok']

    fig, ax = plt.subplots(figsize=(10, 5))

    bibo_rank_weights = np.zeros(top_k)
    bibo_count = 0
    for ld in bibo_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        bibo_rank_weights += w.mean(axis=0)
        bibo_count += 1
    bibo_rank_weights /= bibo_count

    qwen_rank_weights = np.zeros(top_k)
    qwen_count = 0
    for ld in qwen_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        qwen_rank_weights += w.mean(axis=0)
        qwen_count += 1
    qwen_rank_weights /= qwen_count

    x = np.arange(top_k)
    width = 0.35
    bars1 = ax.bar(x - width/2, bibo_rank_weights, width, color=BIBO_COLOR, alpha=0.85, label='BiBo')
    bars2 = ax.bar(x + width/2, qwen_rank_weights, width, color=QWEN_COLOR, alpha=0.85, label='Qwen3MoE')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=8, color=BIBO_COLOR)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=8, color=QWEN_COLOR)

    ax.axhline(y=1/top_k, color='gray', linestyle='--', alpha=0.5, label=f'Uniform (1/{top_k})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank-{i+1}' for i in range(top_k)])
    ax.set_xlabel('Expert Rank (by routing weight)')
    ax.set_ylabel('Average Weight')
    ax.set_title(f'Weight Distribution by Expert Rank — seq={seq_len}\n'
                 'Flatter = better expert utilization', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(bibo_rank_weights.max(), qwen_rank_weights.max()) * 1.25)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'weight_rank_distribution_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ weight_rank_distribution_seq{seq_len}.png")


# ============================================================
# PLOT 12: Per-layer routing diversity
# ============================================================

def plot_routing_diversity(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len, batch_size):
    """Per-layer: effective number of experts and fraction used."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bibo_layers = sorted(bibo_data.keys())
    bibo_effective, bibo_used_frac = [], []
    for l in bibo_layers:
        idx = bibo_data[l]['indices'].numpy().flatten()
        counts = np.bincount(idx, minlength=n_exp_bibo).astype(float)
        dist = counts / (counts.sum() + 1e-10)
        entropy = -np.sum(dist * np.log(dist + 1e-10))
        bibo_effective.append(np.exp(entropy))
        bibo_used_frac.append((counts > 0).sum() / n_exp_bibo)

    qwen_layers = sorted(qwen_data.keys())
    qwen_effective, qwen_used_frac = [], []
    for l in qwen_layers:
        idx = qwen_data[l]['indices'].numpy().flatten()
        counts = np.bincount(idx, minlength=n_exp_qwen).astype(float)
        dist = counts / (counts.sum() + 1e-10)
        entropy = -np.sum(dist * np.log(dist + 1e-10))
        qwen_effective.append(np.exp(entropy))
        qwen_used_frac.append((counts > 0).sum() / n_exp_qwen)

    all_layer_ids = sorted(set(bibo_layers) | set(qwen_layers))
    layer_to_pos = {l: i for i, l in enumerate(all_layer_ids)}
    bibo_pos = np.array([layer_to_pos[l] for l in bibo_layers])
    qwen_pos = np.array([layer_to_pos[l] for l in qwen_layers])

    ax1.bar(bibo_pos - 0.2, bibo_effective, 0.4, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax1.bar(qwen_pos + 0.2, qwen_effective, 0.4, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    ax1.axhline(y=n_exp_bibo, color='gray', linestyle='--', alpha=0.4, label=f'Max ({n_exp_bibo})')
    ax1.set_xticks(np.arange(len(all_layer_ids)))
    ax1.set_xticklabels([f'L{l}' for l in all_layer_ids])
    ax1.set_ylabel('Effective # Experts (exp(H))')
    ax1.set_title('Effective Expert Count per Layer', fontweight='bold')
    ax1.legend()

    ax2.bar(bibo_pos - 0.2, bibo_used_frac, 0.4, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax2.bar(qwen_pos + 0.2, qwen_used_frac, 0.4, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax2.set_xticks(np.arange(len(all_layer_ids)))
    ax2.set_xticklabels([f'L{l}' for l in all_layer_ids])
    ax2.set_ylabel('Fraction of Experts Used')
    ax2.set_title('Expert Utilization per Layer', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend()

    plt.suptitle(f'Routing Diversity — seq={seq_len}, batch={batch_size}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'routing_diversity_seq{seq_len}_bs{batch_size}.png'))
    plt.close()
    print(f"  ✓ routing_diversity_seq{seq_len}_bs{batch_size}.png")


# ============================================================
# PLOT 13: Routing stability
# ============================================================

def plot_routing_stability(model, device, model_type, model_name, n_experts, val_data, seq_len):
    """Measure routing agreement across similar inputs (Jaccard similarity)."""
    n_samples = min(20, len(val_data))
    samples = torch.tensor(val_data[:n_samples, :-1], dtype=torch.long)

    all_indices = []
    for i in range(n_samples):
        ld = extract_routing_data(model, samples[i], device, model_type)
        first_layer = list(ld.values())[0]
        idx = first_layer['indices']
        if idx.dim() == 3:
            idx = idx[0]
        all_indices.append(idx.numpy())

    n_tokens = all_indices[0].shape[0]
    jaccard_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            jaccards = []
            for t in range(n_tokens):
                set_i = set(all_indices[i][t])
                set_j = set(all_indices[j][t])
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccards.append(intersection / union if union > 0 else 1.0)
            avg_jaccard = np.mean(jaccards)
            jaccard_matrix[i, j] = avg_jaccard
            jaccard_matrix[j, i] = avg_jaccard

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(jaccard_matrix, ax=ax, cmap='YlGnBu', vmin=0, vmax=1,
                annot=True, fmt='.2f', square=True,
                cbar_kws={'label': 'Avg Jaccard Similarity'})
    ax.set_title(f'{model_name} — Routing Stability (seq={seq_len})\n'
                 'Higher = more consistent routing across different inputs',
                 fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')

    mask = np.triu(np.ones_like(jaccard_matrix, dtype=bool), k=1)
    avg_stability = jaccard_matrix[mask].mean()
    ax.text(0.5, -0.08, f'Average pairwise stability: {avg_stability:.3f}',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'routing_stability_{model_name}_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ routing_stability_{model_name}_seq{seq_len}.png")

    return avg_stability


# ============================================================
# PLOT 14: PolyGLU Expert-Type Deep Dive (BiBo-specific)
# ============================================================

def plot_expert_type_analysis(bibo_data_dict, bibo_cfg, seq_lens):
    """
    BiBo-specific deep analysis of expert types:
    - Aggregate usage by activation type (SiLU vs ReLU² vs Tanh vs Identity vs Zero)
    - Per-layer activation preference
    - Activation preference by token position (unsorted vs sorted)
    - Weight magnitude by expert type (are some types weighted higher?)
    """
    expert_types = get_expert_types(bibo_cfg)
    type_order = ['silu', 'relu2', 'tanh', 'identity', 'zero']
    type_colors = [POLYGLU_COLORS.get(t, SPECIAL_COLORS.get(t, '#000')) for t in type_order]
    type_labels = ['SiLU (SwiGLU)', 'ReLU² (ReGLU²)', 'Tanh (TanhGLU)', 'Identity', 'Zero']

    n_experts = len(expert_types)
    expert_to_type = {i: expert_types[i] for i in range(n_experts)}

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Aggregate usage by type
    ax = fig.add_subplot(gs[0, 0])
    type_counts = {t: 0 for t in type_order}
    total_tokens = 0

    for sl, data in bibo_data_dict.items():
        for layer_idx, ld in data.items():
            idx = ld['indices'].numpy().flatten()
            total_tokens += len(idx)
            for exp_id in idx:
                t = expert_to_type[exp_id]
                type_counts[t] += 1

    fracs = [type_counts[t] / (total_tokens + 1e-10) for t in type_order]
    bars = ax.bar(range(len(type_order)), fracs, color=type_colors, edgecolor='white', linewidth=1)
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_labels, rotation=20, fontsize=9)
    ax.set_ylabel('Fraction of All Routing Decisions')
    ax.set_title('A) Aggregate Expert-Type Usage', fontweight='bold')
    type_expert_count = {t: sum(1 for et in expert_types if et == t) for t in type_order}
    for i, (t, frac) in enumerate(zip(type_order, fracs)):
        n_of_type = type_expert_count[t]
        expected = n_of_type / n_experts
        ax.axhline(y=expected, xmin=(i-0.3)/len(type_order), xmax=(i+0.7)/len(type_order),
                   color=type_colors[i], linestyle='--', alpha=0.4)
        ax.text(i, frac + 0.005, f'×{n_of_type}', ha='center', fontsize=8, color='gray')

    # Panel B: Per-layer activation preference
    ax = fig.add_subplot(gs[0, 1])
    all_layers = set()
    for data in bibo_data_dict.values():
        all_layers.update(data.keys())
    all_layers = sorted(all_layers)

    layer_type_usage = {l: {t: 0 for t in type_order} for l in all_layers}
    layer_totals = {l: 0 for l in all_layers}

    for sl, data in bibo_data_dict.items():
        for l, ld in data.items():
            idx = ld['indices'].numpy().flatten()
            layer_totals[l] += len(idx)
            for exp_id in idx:
                layer_type_usage[l][expert_to_type[exp_id]] += 1

    x = np.arange(len(all_layers))
    bottom = np.zeros(len(all_layers))
    for t_idx, t in enumerate(type_order):
        vals = [layer_type_usage[l][t] / (layer_totals[l] + 1e-10) for l in all_layers]
        ax.bar(x, vals, bottom=bottom, color=type_colors[t_idx], label=type_labels[t_idx],
               edgecolor='white', linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in all_layers])
    ax.set_ylabel('Fraction')
    ax.set_title('B) Per-Layer Expert-Type Preference', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 1.05)

    # Panel C: Activation preference by position
    ax = fig.add_subplot(gs[1, 0])
    unsorted_type_counts = {t: 0 for t in type_order}
    sorted_type_counts = {t: 0 for t in type_order}
    unsorted_total = 0
    sorted_total = 0

    for sl, data in bibo_data_dict.items():
        half = sl
        for l, ld in data.items():
            idx = ld['indices'].numpy()
            if idx.ndim == 3:
                actual_seq = idx.shape[1]
                unsorted_idx = idx[:, :min(half, actual_seq), :].flatten()
                sorted_idx = idx[:, min(half, actual_seq):, :].flatten()
            else:
                unsorted_idx = idx[:half, :].flatten()
                sorted_idx = idx[half:, :].flatten()

            unsorted_total += len(unsorted_idx)
            sorted_total += len(sorted_idx)
            for exp_id in unsorted_idx:
                unsorted_type_counts[expert_to_type[exp_id]] += 1
            for exp_id in sorted_idx:
                sorted_type_counts[expert_to_type[exp_id]] += 1

    task = CFG['training'].get('task', 'sort')
    input_label = 'Phase1 (input)' if task == 'arithmetic' else 'Unsorted (input)'
    output_label = 'Phase2+3 (output)' if task == 'arithmetic' else 'Sorted (output)'

    x = np.arange(len(type_order))
    width = 0.35
    unsorted_fracs = [unsorted_type_counts[t] / (unsorted_total + 1e-10) for t in type_order]
    sorted_fracs = [sorted_type_counts[t] / (sorted_total + 1e-10) for t in type_order]

    ax.bar(x - width/2, unsorted_fracs, width, color='#FFA726', alpha=0.8, label=input_label)
    ax.bar(x + width/2, sorted_fracs, width, color='#66BB6A', alpha=0.8, label=output_label)
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=20, fontsize=9)
    ax.set_ylabel('Fraction')
    ax.set_title(f'C) Expert-Type Preference: {input_label} vs {output_label}', fontweight='bold')
    ax.legend()

    # Panel D: Average routing weight by expert type
    ax = fig.add_subplot(gs[1, 1])
    type_weight_sums = {t: 0.0 for t in type_order}
    type_weight_counts = {t: 0 for t in type_order}

    for sl, data in bibo_data_dict.items():
        for l, ld in data.items():
            idx = ld['indices'].numpy()
            weights = ld['weights'].numpy()
            if idx.ndim == 3:
                idx_flat = idx.reshape(-1)
                weights_flat = weights.reshape(-1)
            else:
                idx_flat = idx.flatten()
                weights_flat = weights.flatten()

            for exp_id, w in zip(idx_flat, weights_flat):
                t = expert_to_type[exp_id]
                type_weight_sums[t] += w
                type_weight_counts[t] += 1

    avg_weights = [type_weight_sums[t] / (type_weight_counts[t] + 1e-10) for t in type_order]
    bars = ax.bar(range(len(type_order)), avg_weights, color=type_colors, edgecolor='white', linewidth=1)
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_labels, rotation=20, fontsize=9)
    ax.set_ylabel('Average Routing Weight')
    ax.set_title('D) Avg Weight When Selected (confidence per type)', fontweight='bold')
    for i, v in enumerate(avg_weights):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    # Panel E: Expert-type co-selection patterns
    ax = fig.add_subplot(gs[2, 0])
    type_cosel = np.zeros((len(type_order), len(type_order)))
    type_to_idx = {t: i for i, t in enumerate(type_order)}
    total_pairs = 0

    for sl, data in bibo_data_dict.items():
        for l, ld in data.items():
            idx = ld['indices'].numpy()
            if idx.ndim == 3:
                idx = idx.reshape(-1, idx.shape[-1])
            for row in idx:
                types_in_row = [expert_to_type[e] for e in row]
                for a, b in combinations(types_in_row, 2):
                    type_cosel[type_to_idx[a], type_to_idx[b]] += 1
                    type_cosel[type_to_idx[b], type_to_idx[a]] += 1
                    total_pairs += 1

    if total_pairs > 0:
        type_cosel /= total_pairs

    mask = np.triu(np.ones_like(type_cosel, dtype=bool), k=0)
    sns.heatmap(type_cosel, mask=mask, ax=ax, cmap='YlOrRd', annot=True, fmt='.4f',
                xticklabels=type_labels, yticklabels=type_labels, square=True,
                cbar_kws={'label': 'Co-selection freq', 'shrink': 0.8})
    ax.set_title('E) Expert-Type Co-Selection\n(which activation types pair together)', fontweight='bold')

    # Panel F: Per-seq-len type preference evolution
    ax = fig.add_subplot(gs[2, 1])
    for t_idx, t in enumerate(type_order):
        per_sl_fracs = []
        for sl in sorted(bibo_data_dict.keys()):
            data = bibo_data_dict[sl]
            t_count = 0
            total = 0
            for l, ld in data.items():
                idx = ld['indices'].numpy().flatten()
                total += len(idx)
                t_count += sum(1 for e in idx if expert_to_type[e] == t)
            per_sl_fracs.append(t_count / (total + 1e-10))
        ax.plot(sorted(bibo_data_dict.keys()), per_sl_fracs, 'o-',
                color=type_colors[t_idx], linewidth=2, markersize=8, label=type_labels[t_idx])

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Fraction of Routing Decisions')
    ax.set_title('F) Expert-Type Usage vs Sequence Length', fontweight='bold')
    ax.legend(fontsize=8)

    fig.suptitle('BiBo PolyGLU Expert-Type Deep Dive\n'
                 'How does the router utilize diverse activation functions?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(PLOTS_DIR, 'polyglu_expert_type_analysis.png'))
    plt.close()
    print(f"  ✓ polyglu_expert_type_analysis.png")


# ============================================================
# PLOT 15: Special expert analysis (Identity + Zero only)
# ============================================================

def plot_special_expert_analysis(bibo_data_dict, bibo_cfg, seq_lens):
    """
    BiBo-specific: how are Identity and Zero experts used?
    Across seq_lens and layers.
    """
    expert_types = get_expert_types(bibo_cfg)
    n_experts = len(expert_types)

    identity_ids = [i for i, t in enumerate(expert_types) if t == 'identity']
    zero_ids = [i for i, t in enumerate(expert_types) if t == 'zero']
    special_groups = [('Identity', identity_ids, SPECIAL_COLORS['identity']),
                      ('Zero', zero_ids, SPECIAL_COLORS['zero'])]

    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5*len(seq_lens), 5), squeeze=False)

    for col, sl in enumerate(seq_lens):
        ax = axes[0, col]
        if sl not in bibo_data_dict:
            ax.set_visible(False)
            continue

        data = bibo_data_dict[sl]
        layers = sorted(data.keys())

        usage_matrix = np.zeros((len(special_groups), len(layers)))
        for i, l in enumerate(layers):
            idx = data[l]['indices'].numpy().flatten()
            total = len(idx)
            for j, (name, ids, color) in enumerate(special_groups):
                count = sum((idx == eid).sum() for eid in ids)
                usage_matrix[j, i] = count / total

        x = np.arange(len(layers))
        bottom = np.zeros(len(layers))
        for j, (name, ids, color) in enumerate(special_groups):
            ax.bar(x, usage_matrix[j], bottom=bottom, color=color,
                   label=f'{name} (×{len(ids)})', edgecolor='white', linewidth=0.5)
            bottom += usage_matrix[j]

        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_ylabel('Fraction of Tokens')
        ax.set_title(f'seq={sl}', fontweight='bold')
        if col == 0:
            ax.legend()

    plt.suptitle('BiBo — Special Expert Usage (Identity + Zero)\n'
                 'These provide residual bypass and gradient dampening',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'special_expert_analysis.png'))
    plt.close()
    print(f"  ✓ special_expert_analysis.png")


# ============================================================
# PLOT 16: Expert switching rate
# ============================================================

def plot_expert_switching_rate(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len):
    """How often does the top-1 expert change between consecutive tokens?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bibo_layers = sorted(bibo_data.keys())
    bibo_switch_rates = []
    for l in bibo_layers:
        idx = bibo_data[l]['indices'].numpy()
        if idx.ndim == 3:
            switches = []
            for b in range(idx.shape[0]):
                top1 = idx[b, :, 0]
                switches.append((top1[1:] != top1[:-1]).mean())
            bibo_switch_rates.append(np.mean(switches))
        else:
            top1 = idx[:, 0]
            bibo_switch_rates.append((top1[1:] != top1[:-1]).mean())

    qwen_layers = sorted(qwen_data.keys())
    qwen_switch_rates = []
    for l in qwen_layers:
        idx = qwen_data[l]['indices'].numpy()
        if idx.ndim == 3:
            switches = []
            for b in range(idx.shape[0]):
                top1 = idx[b, :, 0]
                switches.append((top1[1:] != top1[:-1]).mean())
            qwen_switch_rates.append(np.mean(switches))
        else:
            top1 = idx[:, 0]
            qwen_switch_rates.append((top1[1:] != top1[:-1]).mean())

    all_layer_ids = sorted(set(bibo_layers) | set(qwen_layers))
    layer_to_pos = {l: i for i, l in enumerate(all_layer_ids)}
    bibo_pos = np.array([layer_to_pos[l] for l in bibo_layers])
    qwen_pos = np.array([layer_to_pos[l] for l in qwen_layers])
    width = 0.35

    ax1.bar(bibo_pos - width/2, bibo_switch_rates, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax1.bar(qwen_pos + width/2, qwen_switch_rates, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    ax1.set_xticks(np.arange(len(all_layer_ids)))
    ax1.set_xticklabels([f'L{l}' for l in all_layer_ids])
    ax1.set_ylabel('Switch Rate')
    ax1.set_title('Top-1 Expert Switch Rate per Layer', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1)

    expected_random_bibo = 1 - 1/n_exp_bibo
    expected_random_qwen = 1 - 1/n_exp_qwen
    ax1.axhline(y=expected_random_bibo, color=BIBO_COLOR, linestyle='--', alpha=0.4)
    ax1.axhline(y=expected_random_qwen, color=QWEN_COLOR, linestyle='--', alpha=0.4)

    # Per-position switch visualization
    ax2_layer = bibo_layers[0]
    bibo_idx = bibo_data[ax2_layer]['indices'].numpy()
    qwen_idx = qwen_data[qwen_layers[0]]['indices'].numpy()

    if bibo_idx.ndim == 3:
        bibo_top1 = bibo_idx[0, :, 0]
    else:
        bibo_top1 = bibo_idx[:, 0]

    if qwen_idx.ndim == 3:
        qwen_top1 = qwen_idx[0, :, 0]
    else:
        seq_len_actual = len(bibo_top1)
        qwen_top1 = qwen_idx[:seq_len_actual, 0]

    bibo_switches = np.where(bibo_top1[1:] != bibo_top1[:-1])[0]
    qwen_switches = np.where(qwen_top1[1:] != qwen_top1[:-1])[0]

    ax2.eventplot([bibo_switches, qwen_switches], lineoffsets=[1, 0],
                  linelengths=0.8, colors=[BIBO_COLOR, QWEN_COLOR])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Qwen3MoE', 'BiBo'])
    ax2.set_xlabel('Token Position')
    ax2.set_title(f'Switch Events (L{ax2_layer}, first sample)', fontweight='bold')
    task = CFG['training'].get('task', 'sort')
    if task == 'arithmetic':
        actual_len = ax2.get_xlim()[1]
        third = int(actual_len) // 3
        ax2.axvline(x=third, color='green', linestyle=':', alpha=0.6, label='SEP1')
        ax2.axvline(x=2*third, color='orange', linestyle=':', alpha=0.6, label='SEP2')
    else:
        ax2.axvline(x=seq_len, color='green', linestyle=':', alpha=0.6, label='SEP')
    ax2.legend()

    plt.suptitle(f'Expert Switching Analysis — seq={seq_len}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'switching_rate_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ switching_rate_seq{seq_len}.png")


# ============================================================
# PLOT 17: Grand summary dashboard
# ============================================================

def plot_grand_summary(all_metrics, stability_scores, bibo_cfg):
    """Single-page dashboard summarizing key findings."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Key metrics comparison
    ax = fig.add_subplot(gs[0, :])
    metrics_to_show = ['gini', 'normalized_entropy', 'cv']
    labels_show = ['Gini ↓', 'Norm. Entropy ↑', 'CV ↓']

    bibo_avgs = {m: [] for m in metrics_to_show}
    qwen_avgs = {m: [] for m in metrics_to_show}
    for key, metrics in all_metrics.items():
        model, sl, bs = key
        for m in metrics_to_show:
            if model == 'BiBo':
                bibo_avgs[m].append(metrics[m])
            else:
                qwen_avgs[m].append(metrics[m])

    x = np.arange(len(metrics_to_show))
    width = 0.3
    bibo_means = [np.mean(bibo_avgs[m]) for m in metrics_to_show]
    qwen_means = [np.mean(qwen_avgs[m]) for m in metrics_to_show]

    bars1 = ax.bar(x - width/2, bibo_means, width, color=BIBO_COLOR, alpha=0.85, label='BiBo (avg)')
    bars2 = ax.bar(x + width/2, qwen_means, width, color=QWEN_COLOR, alpha=0.85, label='Qwen3MoE (avg)')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_show)
    ax.set_title('Average Load Balance Metrics', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)

    # Panel 2: Stability scores
    ax2 = fig.add_subplot(gs[1, 0])
    bibo_stab = [v for k, v in stability_scores.items() if 'BiBo' in k]
    qwen_stab = [v for k, v in stability_scores.items() if 'Qwen' in k]

    if bibo_stab and qwen_stab:
        ax2.bar(['BiBo', 'Qwen3MoE'], [np.mean(bibo_stab), np.mean(qwen_stab)],
                color=[BIBO_COLOR, QWEN_COLOR], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Avg Jaccard Similarity')
        ax2.set_title('Routing Stability', fontweight='bold')
        ax2.set_ylim(0, 1)

    # Panel 3: Findings
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.axis('off')

    findings = []
    if np.mean(bibo_avgs['gini']) < np.mean(qwen_avgs['gini']):
        findings.append("✓ BiBo has LOWER Gini → more balanced load")
    else:
        findings.append("✗ Qwen has lower Gini")

    if np.mean(bibo_avgs['normalized_entropy']) > np.mean(qwen_avgs['normalized_entropy']):
        findings.append("✓ BiBo has HIGHER entropy → better expert utilization")
    else:
        findings.append("✗ Qwen has higher entropy")

    if bibo_stab and qwen_stab:
        if np.mean(bibo_stab) > np.mean(qwen_stab):
            findings.append("✓ BiBo routing is MORE stable across inputs")
        else:
            findings.append("~ Qwen routing is more stable (expected: simpler router)")

    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_routed = poly_mult * 3 + special_pairs * 2
    findings.append(f"\nBiBo: PolyGLU ({poly_mult}×[SiLU,ReLU²,Tanh]) + {special_pairs}×[Id,Zero] = {n_routed} experts")
    findings.append(f"Router: Skywork-MoE norm (λ={bibo_cfg.get('router_lambda', 1.0)}) + Conv")
    findings.append(f"Qwen: {CFG['qwen3moe']['num_experts']} homogeneous MLP experts + linear router")
    findings.append(f"\nKey insight: PolyGLU diversity + logit norm → all experts contribute")

    text = '\n'.join(findings)
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Key Findings', fontweight='bold', fontsize=12)

    # Panel 4: Config comparison
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    expert_desc = f'PolyGLU({poly_mult}×3) + Special({special_pairs}×2)'
    config_text = (
        f"{'='*80}\n"
        f"{'Config':<20} {'BiBo':<35} {'Qwen3MoE':<30}\n"
        f"{'='*80}\n"
        f"{'Experts':<20} {n_routed:<35} {CFG['qwen3moe']['num_experts']:<30}\n"
        f"{'Top-K':<20} {bibo_cfg['num_experts_per_tok']:<35} {CFG['qwen3moe']['num_experts_per_tok']:<30}\n"
        f"{'Router':<20} {'Conv (k=3) + Skywork norm':<35} {'Linear + softmax':<30}\n"
        f"{'Expert types':<20} {expert_desc:<35} {'MLP×8 (homogeneous)':<30}\n"
        f"{'Shared expert':<20} {'CausalConv1D (always on)':<35} {'MLP (always on)':<30}\n"
        f"{'Load balance':<20} {'Heuristic bias update':<35} {'Aux loss (Switch)':<30}\n"
        f"{'='*80}"
    )
    ax4.text(0.02, 0.95, config_text, transform=ax4.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle('BiBo vs Qwen3MoE — Router Analysis Dashboard',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(PLOTS_DIR, 'grand_summary_dashboard.png'))
    plt.close()
    print(f"  ✓ grand_summary_dashboard.png")


# ============================================================
# PLOT 18: Per-layer weight KDE
# ============================================================

def plot_per_layer_weight_kde(bibo_data, qwen_data, seq_len):
    """KDE of routing weights per layer."""
    bibo_layers = sorted(bibo_data.keys())
    qwen_layers = sorted(qwen_data.keys())
    n_layers = max(len(bibo_layers), len(qwen_layers))

    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 3*n_layers), squeeze=False)

    for i in range(n_layers):
        ax = axes[i, 0]
        if i < len(bibo_layers):
            bl = bibo_layers[i]
            w = bibo_data[bl]['weights'].numpy()
            if w.ndim == 3:
                w = w.reshape(-1, w.shape[-1])
            for k in range(w.shape[1]):
                sns.kdeplot(w[:, k], ax=ax, label=f'Rank-{k+1}', linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_title(f'BiBo L{bl}', color=BIBO_COLOR, fontweight='bold')
            if i == 0:
                ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

        ax = axes[i, 1]
        if i < len(qwen_layers):
            ql = qwen_layers[i]
            w = qwen_data[ql]['weights'].numpy()
            if w.ndim == 3:
                w = w.reshape(-1, w.shape[-1])
            for k in range(w.shape[1]):
                sns.kdeplot(w[:, k], ax=ax, label=f'Rank-{k+1}', linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_title(f'Qwen L{ql}', color=QWEN_COLOR, fontweight='bold')
            if i == 0:
                ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

    plt.suptitle(f'Per-Layer Weight Distribution (KDE) — seq={seq_len}',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'weight_kde_per_layer_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ weight_kde_per_layer_seq{seq_len}.png")
