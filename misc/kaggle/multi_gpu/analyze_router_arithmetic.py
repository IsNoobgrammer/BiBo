"""
Arithmetic-Specialized Router Analysis — BiBo vs Qwen3MoE
==========================================================

Analyzes routing behavior specifically for arithmetic tasks:
- Per-operator expert affinity (which experts handle +, -, *, //)
- Per-phase routing (input expression vs intermediate vs final answer)
- Number magnitude routing (do big numbers route differently?)
- Load balance convergence: mean deviation of expert usage as batch size
  scales from 1 to 100 at fixed seq_len ~128 (the "hard" bucket, ~30 terms)

Key insight: batch_size=64 at seq_len=42 is NOT optimal for measuring load
balance — you need to see how deviation converges as sample count grows.

Usage:
    python misc/kaggle/multi_gpu/analyze_router_arithmetic.py

Produces plots in misc/kaggle/multi_gpu/plots/
"""
import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_utils import (
    setup_style, save_figure, PLOTS_DIR, METRICS_DIR,
    BIBO_COLOR, QWEN_COLOR, get_expert_layout,
    POLYGLU_COLORS, SPECIAL_COLORS,
)
from model_utils import CFG, load_bibo, load_qwen, extract_routing_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
METRICS_OUT = os.path.join(METRICS_DIR, 'router_arithmetic_analysis.json')

setup_style()

# ── Special token constants (must match data_arithmetic.py) ──
VOCAB_SIZE = CFG['training']['vocab_size']
SEP_TOKEN = VOCAB_SIZE - 1
ADD_TOKEN = VOCAB_SIZE - 2
SUB_TOKEN = VOCAB_SIZE - 3
MUL_TOKEN = VOCAB_SIZE - 4
DIV_TOKEN = VOCAB_SIZE - 5
PAD_TOKEN = 0

OP_TOKENS = {ADD_TOKEN: '+', SUB_TOKEN: '-', MUL_TOKEN: '*', DIV_TOKEN: '//'}
ALL_OP_TOKENS = set(OP_TOKENS.keys())


def classify_tokens(sequence):
    """
    Classify each token position in an arithmetic sequence.

    Returns array of labels per position:
      'num_small'  : number in [1, 50]
      'num_med'    : number in [51, 200]
      'num_large'  : number in [201, 500]
      'op_add'     : + operator
      'op_sub'     : - operator
      'op_mul'     : * operator
      'op_div'     : // operator
      'sep'        : phase separator
      'pad'        : padding
      'phase1'     : (alternative) position is in phase 1 (input)
      'phase2'     : position is in phase 2 (intermediate)
      'phase3'     : position is in phase 3 (answer)
    """
    labels = []
    phase = 1
    for t in sequence:
        if t == PAD_TOKEN:
            labels.append('pad')
        elif t == SEP_TOKEN:
            labels.append('sep')
            phase += 1
        elif t == ADD_TOKEN:
            labels.append('op_add')
        elif t == SUB_TOKEN:
            labels.append('op_sub')
        elif t == MUL_TOKEN:
            labels.append('op_mul')
        elif t == DIV_TOKEN:
            labels.append('op_div')
        elif 1 <= t <= 50:
            labels.append('num_small')
        elif 51 <= t <= 200:
            labels.append('num_med')
        elif 201 <= t <= 506:
            labels.append('num_large')
        else:
            labels.append('unknown')
    return labels, phase


def get_phase_masks(sequence):
    """
    Return boolean masks for each phase in the sequence.
    Phase 1 = input expression, Phase 2 = intermediate, Phase 3 = answer.
    """
    seq = np.array(sequence)
    sep_positions = np.where(seq == SEP_TOKEN)[0]

    masks = {}
    if len(sep_positions) >= 2:
        masks['phase1'] = np.zeros(len(seq), dtype=bool)
        masks['phase1'][:sep_positions[0]] = True
        masks['phase2'] = np.zeros(len(seq), dtype=bool)
        masks['phase2'][sep_positions[0]+1:sep_positions[1]] = True
        masks['phase3'] = np.zeros(len(seq), dtype=bool)
        masks['phase3'][sep_positions[1]+1:] = True
    else:
        masks['phase1'] = np.ones(len(seq), dtype=bool)
        masks['phase2'] = np.zeros(len(seq), dtype=bool)
        masks['phase3'] = np.zeros(len(seq), dtype=bool)

    # Non-pad mask
    masks['valid'] = seq != PAD_TOKEN
    return masks


def compute_expert_deviation(indices, n_experts):
    """
    Compute mean absolute deviation of expert usage from uniform.

    Perfect balance = 0. Higher = more imbalanced.
    Returns: mean_abs_deviation (fraction), std across experts.
    """
    counts = np.bincount(indices.flatten(), minlength=n_experts).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0
    fracs = counts / total
    uniform = 1.0 / n_experts
    mad = np.mean(np.abs(fracs - uniform))
    std = np.std(fracs)
    return float(mad), float(std)


def plot_load_balance_vs_batch_size(bibo_model, qwen_model, val_data,
                                    device_bibo, device_qwen, bibo_cfg):
    """
    THE KEY PLOT: Mean deviation of expert usage as batch size scales 1→100.

    Fixed seq_len ~128 (hard bucket). Shows how many samples you need for
    stable load balance measurement. batch=64 at seq=42 is suboptimal —
    this reveals the true convergence curve.
    """
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2
    n_exp_qwen = CFG['qwen3moe']['num_experts']

    # Use the longest available bucket (closest to seq_len ~128)
    # The "hard" bucket (19-30 terms) produces sequences ~80-130 tokens
    batch_sizes = list(range(1, 129))  # 1 to 128

    # We need enough data — use up to 128 samples
    max_samples = min(128, len(val_data))
    data_pool = val_data[:max_samples]

    bibo_mads = []
    bibo_stds = []
    qwen_mads = []
    qwen_stds = []

    print(f"\n  Sweeping batch size 1→{max_samples} (seq_len from data)...")

    for bs in batch_sizes:
        if bs > max_samples:
            break

        # Take first `bs` samples (excluding last token for input)
        batch = torch.tensor(data_pool[:bs, :-1], dtype=torch.long)

        # BiBo
        bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
        all_indices_bibo = []
        for layer_idx, ld in bibo_ld.items():
            all_indices_bibo.append(ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]))
        if all_indices_bibo:
            combined = np.concatenate(all_indices_bibo)
            mad, std = compute_expert_deviation(combined, n_exp_bibo)
            bibo_mads.append(mad)
            bibo_stds.append(std)
        else:
            bibo_mads.append(0)
            bibo_stds.append(0)

        # Qwen
        qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')
        all_indices_qwen = []
        for layer_idx, ld in qwen_ld.items():
            all_indices_qwen.append(ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]))
        if all_indices_qwen:
            combined = np.concatenate(all_indices_qwen)
            mad, std = compute_expert_deviation(combined, n_exp_qwen)
            qwen_mads.append(mad)
            qwen_stds.append(std)
        else:
            qwen_mads.append(0)
            qwen_stds.append(0)

    actual_sizes = batch_sizes[:len(bibo_mads)]

    # ── Plot: Mean Absolute Deviation vs Batch Size ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MAD (mean absolute deviation from uniform)
    ax1.plot(actual_sizes, bibo_mads, color=BIBO_COLOR, linewidth=2,
             label=f'BiBo ({n_exp_bibo} experts)', zorder=5)
    ax1.plot(actual_sizes, qwen_mads, color=QWEN_COLOR, linewidth=2,
             label=f'Qwen3MoE ({n_exp_qwen} experts)', zorder=5)

    # Add ideal uniform line
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect balance')

    # Mark the "old" batch size of 64
    if 64 <= max_samples:
        ax1.axvline(x=64, color='gray', linestyle=':', alpha=0.7)
        ax1.annotate('bs=64', xy=(64, ax1.get_ylim()[1] * 0.9 if bibo_mads else 0),
                     fontsize=8, color='gray', ha='center')

    ax1.set_xlabel('Batch Size (number of sequences)')
    ax1.set_ylabel('Mean Absolute Deviation from Uniform')
    ax1.set_title('Load Balance Convergence\n(lower = more balanced)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(1, max_samples)

    # Right: Std of expert fractions
    ax2.plot(actual_sizes, bibo_stds, color=BIBO_COLOR, linewidth=2,
             label=f'BiBo ({n_exp_bibo} experts)')
    ax2.plot(actual_sizes, qwen_stds, color=QWEN_COLOR, linewidth=2,
             label=f'Qwen3MoE ({n_exp_qwen} experts)')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Perfect balance')

    if 64 <= max_samples:
        ax2.axvline(x=64, color='gray', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Batch Size (number of sequences)')
    ax2.set_ylabel('Std of Expert Usage Fractions')
    ax2.set_title('Expert Usage Variance\n(lower = more uniform)')
    ax2.legend(loc='upper right')
    ax2.set_xlim(1, max_samples)

    fig.suptitle('Expert Load Balance vs Sample Count — Arithmetic Task (seq≈128)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_load_balance_vs_batch_size')

    return {
        'batch_sizes': actual_sizes,
        'bibo_mad': bibo_mads,
        'bibo_std': bibo_stds,
        'qwen_mad': qwen_mads,
        'qwen_std': qwen_stds,
    }


def plot_per_layer_deviation_vs_batch_size(bibo_model, qwen_model, val_data,
                                           device_bibo, device_qwen, bibo_cfg):
    """
    Per-layer view: how each MoE layer's balance converges with batch size.
    Shows whether some layers are inherently more imbalanced.
    """
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2
    n_exp_qwen = CFG['qwen3moe']['num_experts']

    batch_sizes = [1, 5, 10, 20, 30, 50, 75, 100, 128]
    max_samples = min(128, len(val_data))

    # Collect per-layer MAD for each batch size
    bibo_layer_mads = {}  # {layer_idx: [mad_per_bs]}
    qwen_layer_mads = {}

    for bs in batch_sizes:
        if bs > max_samples:
            break
        batch = torch.tensor(val_data[:bs, :-1], dtype=torch.long)

        bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
        for layer_idx, ld in bibo_ld.items():
            indices = ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1])
            mad, _ = compute_expert_deviation(indices, n_exp_bibo)
            bibo_layer_mads.setdefault(layer_idx, []).append(mad)

        qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')
        for layer_idx, ld in qwen_ld.items():
            indices = ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1])
            mad, _ = compute_expert_deviation(indices, n_exp_qwen)
            qwen_layer_mads.setdefault(layer_idx, []).append(mad)

    # Plot
    n_bibo_layers = len(bibo_layer_mads)
    n_qwen_layers = len(qwen_layer_mads)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap_b = plt.cm.Blues(np.linspace(0.4, 0.9, n_bibo_layers))
    for i, (layer_idx, mads) in enumerate(sorted(bibo_layer_mads.items())):
        axes[0].plot(batch_sizes[:len(mads)], mads, color=cmap_b[i],
                     linewidth=1.5, marker='o', markersize=3,
                     label=f'Layer {layer_idx}')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Mean Abs Deviation')
    axes[0].set_title(f'BiBo — Per-Layer Load Balance ({n_exp_bibo} experts)')
    axes[0].legend(fontsize=8)

    cmap_q = plt.cm.Oranges(np.linspace(0.4, 0.9, n_qwen_layers))
    for i, (layer_idx, mads) in enumerate(sorted(qwen_layer_mads.items())):
        axes[1].plot(batch_sizes[:len(mads)], mads, color=cmap_q[i],
                     linewidth=1.5, marker='o', markersize=3,
                     label=f'Layer {layer_idx}')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Mean Abs Deviation')
    axes[1].set_title(f'Qwen3MoE — Per-Layer Load Balance ({n_exp_qwen} experts)')
    axes[1].legend(fontsize=8)

    fig.suptitle('Per-Layer Balance Convergence — Arithmetic (seq≈128)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_per_layer_deviation_vs_bs')


def plot_per_layer_topk_weights(bibo_model, qwen_model, val_data,
                                device_bibo, device_qwen, bibo_cfg):
    """
    4-panel plot: per-layer mean top-1 and top-2 router weights for BiBo and Qwen.

    Shows how much weight the router assigns to its top choices per layer.
    High top-1 = router is decisive (one expert dominates).
    Low gap between top-1 and top-2 = router spreads load more evenly.
    """
    bs = min(64, len(val_data))
    batch = torch.tensor(val_data[:bs, :-1], dtype=torch.long)

    bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
    qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')

    def _get_topk_weights_per_layer(layer_data):
        """Return dict: layer_idx → (mean_top1, std_top1, mean_top2, std_top2)."""
        results = {}
        for layer_idx, ld in sorted(layer_data.items()):
            weights = ld['weights'].numpy()
            if weights.ndim == 2:
                weights = weights[:, :, np.newaxis]
            # weights shape: (bs, seq_len, top_k)
            # top-1 is index 0, top-2 is index 1 (already sorted by router)
            top1 = weights[:, :, 0].flatten()
            top2 = weights[:, :, 1].flatten() if weights.shape[2] > 1 else np.zeros_like(top1)
            results[layer_idx] = {
                'top1_mean': float(np.mean(top1)),
                'top1_std': float(np.std(top1)),
                'top2_mean': float(np.mean(top2)),
                'top2_std': float(np.std(top2)),
            }
        return results

    bibo_weights = _get_topk_weights_per_layer(bibo_ld)
    qwen_weights = _get_topk_weights_per_layer(qwen_ld)

    # 4-panel plot: BiBo top-1, BiBo top-2, Qwen top-1, Qwen top-2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # BiBo Top-1
    layers_b = sorted(bibo_weights.keys())
    top1_means_b = [bibo_weights[l]['top1_mean'] for l in layers_b]
    top1_stds_b = [bibo_weights[l]['top1_std'] for l in layers_b]
    axes[0, 0].bar(range(len(layers_b)), top1_means_b, yerr=top1_stds_b,
                   color=BIBO_COLOR, alpha=0.8, capsize=4, edgecolor='white')
    axes[0, 0].set_xticks(range(len(layers_b)))
    axes[0, 0].set_xticklabels([f'L{l}' for l in layers_b])
    axes[0, 0].set_ylabel('Mean Weight')
    axes[0, 0].set_title('BiBo — Top-1 Router Weight per Layer')
    axes[0, 0].set_ylim(0, 1)

    # BiBo Top-2
    top2_means_b = [bibo_weights[l]['top2_mean'] for l in layers_b]
    top2_stds_b = [bibo_weights[l]['top2_std'] for l in layers_b]
    axes[0, 1].bar(range(len(layers_b)), top2_means_b, yerr=top2_stds_b,
                   color=BIBO_COLOR, alpha=0.6, capsize=4, edgecolor='white')
    axes[0, 1].set_xticks(range(len(layers_b)))
    axes[0, 1].set_xticklabels([f'L{l}' for l in layers_b])
    axes[0, 1].set_ylabel('Mean Weight')
    axes[0, 1].set_title('BiBo — Top-2 Router Weight per Layer')
    axes[0, 1].set_ylim(0, 1)

    # Qwen Top-1
    layers_q = sorted(qwen_weights.keys())
    top1_means_q = [qwen_weights[l]['top1_mean'] for l in layers_q]
    top1_stds_q = [qwen_weights[l]['top1_std'] for l in layers_q]
    axes[1, 0].bar(range(len(layers_q)), top1_means_q, yerr=top1_stds_q,
                   color=QWEN_COLOR, alpha=0.8, capsize=4, edgecolor='white')
    axes[1, 0].set_xticks(range(len(layers_q)))
    axes[1, 0].set_xticklabels([f'L{l}' for l in layers_q])
    axes[1, 0].set_ylabel('Mean Weight')
    axes[1, 0].set_title('Qwen3MoE — Top-1 Router Weight per Layer')
    axes[1, 0].set_ylim(0, 1)

    # Qwen Top-2
    top2_means_q = [qwen_weights[l]['top2_mean'] for l in layers_q]
    top2_stds_q = [qwen_weights[l]['top2_std'] for l in layers_q]
    axes[1, 1].bar(range(len(layers_q)), top2_means_q, yerr=top2_stds_q,
                   color=QWEN_COLOR, alpha=0.6, capsize=4, edgecolor='white')
    axes[1, 1].set_xticks(range(len(layers_q)))
    axes[1, 1].set_xticklabels([f'L{l}' for l in layers_q])
    axes[1, 1].set_ylabel('Mean Weight')
    axes[1, 1].set_title('Qwen3MoE — Top-2 Router Weight per Layer')
    axes[1, 1].set_ylim(0, 1)

    fig.suptitle('Per-Layer Top-K Router Weights — Arithmetic Task\n'
                 '(High top-1 = decisive routing, Low gap = better expert utilization)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_per_layer_topk_weights')

    return bibo_weights, qwen_weights


def plot_operator_expert_affinity(bibo_model, qwen_model, val_data,
                                  device_bibo, device_qwen, bibo_cfg):
    """
    Which experts handle which operators?
    For each operator token (+, -, *, //), show the expert selection distribution.
    Also shows number-magnitude routing and phase-based routing.
    """
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2
    n_exp_qwen = CFG['qwen3moe']['num_experts']

    # Use a decent batch for stable statistics
    bs = min(64, len(val_data))
    batch_np = val_data[:bs]
    batch = torch.tensor(batch_np[:, :-1], dtype=torch.long)

    # Extract routing
    bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
    qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')

    # Classify all tokens
    token_categories = {
        'op_add': [], 'op_sub': [], 'op_mul': [], 'op_div': [],
        'num_small': [], 'num_med': [], 'num_large': [],
        'phase1': [], 'phase2': [], 'phase3': [],
    }

    # Build position→category mapping for the batch
    seq_len = batch_np.shape[1] - 1  # input is [:-1]
    position_categories = np.full((bs, seq_len), '', dtype=object)

    for sample_idx in range(bs):
        seq = batch_np[sample_idx, :-1]
        labels, _ = classify_tokens(seq)
        for pos, label in enumerate(labels):
            if label != 'pad':
                position_categories[sample_idx, pos] = label

    # Also compute phase masks
    position_phases = np.full((bs, seq_len), '', dtype=object)
    for sample_idx in range(bs):
        seq = batch_np[sample_idx, :-1]
        masks = get_phase_masks(seq)
        for pos in range(len(seq)):
            if masks['phase1'][pos]:
                position_phases[sample_idx, pos] = 'phase1'
            elif masks['phase2'][pos]:
                position_phases[sample_idx, pos] = 'phase2'
            elif masks['phase3'][pos]:
                position_phases[sample_idx, pos] = 'phase3'

    def _ensure_3d(indices):
        """Ensure indices are (bs, seq_len, top_k) — add dim if 2D."""
        if indices.ndim == 2:
            return indices[:, :, np.newaxis]
        return indices

    def _gather_category_experts(layer_data, n_experts, categories_map):
        """For each token category, collect which experts were selected."""
        cat_experts = {cat: np.zeros(n_experts) for cat in
                       ['op_add', 'op_sub', 'op_mul', 'op_div',
                        'num_small', 'num_med', 'num_large']}
        n_samples, n_positions = categories_map.shape

        for layer_idx, ld in layer_data.items():
            indices = _ensure_3d(ld['indices'].numpy())  # (bs, seq_len, top_k)
            for s in range(min(indices.shape[0], n_samples)):
                for p in range(min(indices.shape[1], n_positions)):
                    cat = categories_map[s, p]
                    if cat in cat_experts:
                        for k in range(indices.shape[2]):
                            cat_experts[cat][indices[s, p, k]] += 1
        return cat_experts

    def _gather_phase_experts(layer_data, n_experts, phases_map):
        """For each phase, collect expert selection distribution."""
        phase_experts = {ph: np.zeros(n_experts) for ph in ['phase1', 'phase2', 'phase3']}
        n_samples, n_positions = phases_map.shape

        for layer_idx, ld in layer_data.items():
            indices = _ensure_3d(ld['indices'].numpy())
            for s in range(min(indices.shape[0], n_samples)):
                for p in range(min(indices.shape[1], n_positions)):
                    ph = phases_map[s, p]
                    if ph in phase_experts:
                        for k in range(indices.shape[2]):
                            phase_experts[ph][indices[s, p, k]] += 1
        return phase_experts

    # Gather data
    bibo_cat = _gather_category_experts(bibo_ld, n_exp_bibo, position_categories)
    qwen_cat = _gather_category_experts(qwen_ld, n_exp_qwen, position_categories)
    bibo_phase = _gather_phase_experts(bibo_ld, n_exp_bibo, position_phases)
    qwen_phase = _gather_phase_experts(qwen_ld, n_exp_qwen, position_phases)

    # ── Plot 1: Operator → Expert Affinity (heatmap) ──
    op_cats = ['op_add', 'op_sub', 'op_mul', 'op_div']
    op_labels = ['+', '−', '×', '÷']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BiBo
    bibo_op_matrix = np.array([bibo_cat[c] for c in op_cats])
    row_sums = bibo_op_matrix.sum(axis=1, keepdims=True)
    bibo_op_norm = np.divide(bibo_op_matrix, row_sums, where=row_sums > 0,
                             out=np.zeros_like(bibo_op_matrix))

    layout = get_expert_layout(bibo_cfg)
    expert_labels_bibo = [l for l, _, _ in layout[:n_exp_bibo]]

    sns.heatmap(bibo_op_norm, ax=axes[0], cmap='Blues', vmin=0,
                xticklabels=expert_labels_bibo, yticklabels=op_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Selection Probability'})
    axes[0].set_title(f'BiBo — Operator→Expert Affinity\n({n_exp_bibo} PolyGLU experts)')
    axes[0].set_xlabel('Expert')
    axes[0].set_ylabel('Operator')

    # Qwen
    qwen_op_matrix = np.array([qwen_cat[c] for c in op_cats])
    row_sums = qwen_op_matrix.sum(axis=1, keepdims=True)
    qwen_op_norm = np.divide(qwen_op_matrix, row_sums, where=row_sums > 0,
                             out=np.zeros_like(qwen_op_matrix))

    expert_labels_qwen = [f'E{i}' for i in range(n_exp_qwen)]
    sns.heatmap(qwen_op_norm, ax=axes[1], cmap='Oranges', vmin=0,
                xticklabels=expert_labels_qwen, yticklabels=op_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Selection Probability'})
    axes[1].set_title(f'Qwen3MoE — Operator→Expert Affinity\n({n_exp_qwen} homogeneous experts)')
    axes[1].set_xlabel('Expert')
    axes[1].set_ylabel('Operator')

    fig.suptitle('Which Experts Handle Which Operators? — Arithmetic Task',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_operator_expert_affinity')

    # ── Plot 2: Phase → Expert Routing ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    phase_labels = ['Phase 1\n(Input)', 'Phase 2\n(Intermediate)', 'Phase 3\n(Answer)']
    phases = ['phase1', 'phase2', 'phase3']

    # BiBo
    bibo_ph_matrix = np.array([bibo_phase[p] for p in phases])
    row_sums = bibo_ph_matrix.sum(axis=1, keepdims=True)
    bibo_ph_norm = np.divide(bibo_ph_matrix, row_sums, where=row_sums > 0,
                             out=np.zeros_like(bibo_ph_matrix))

    sns.heatmap(bibo_ph_norm, ax=axes[0], cmap='Blues', vmin=0,
                xticklabels=expert_labels_bibo, yticklabels=phase_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Selection Probability'})
    axes[0].set_title('BiBo — Phase→Expert Routing')
    axes[0].set_xlabel('Expert')

    # Qwen
    qwen_ph_matrix = np.array([qwen_phase[p] for p in phases])
    row_sums = qwen_ph_matrix.sum(axis=1, keepdims=True)
    qwen_ph_norm = np.divide(qwen_ph_matrix, row_sums, where=row_sums > 0,
                             out=np.zeros_like(qwen_ph_matrix))

    sns.heatmap(qwen_ph_norm, ax=axes[1], cmap='Oranges', vmin=0,
                xticklabels=expert_labels_qwen, yticklabels=phase_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Selection Probability'})
    axes[1].set_title('Qwen3MoE — Phase→Expert Routing')
    axes[1].set_xlabel('Expert')

    fig.suptitle('Expert Routing by Computation Phase — Arithmetic Task',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_phase_expert_routing')

    # ── Plot 3: Number Magnitude → Expert Routing ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mag_cats = ['num_small', 'num_med', 'num_large']
    mag_labels = ['Small [1-50]', 'Medium [51-200]', 'Large [201-500]']

    # BiBo
    bibo_mag_matrix = np.array([bibo_cat[c] for c in mag_cats])
    row_sums = bibo_mag_matrix.sum(axis=1, keepdims=True)
    bibo_mag_norm = np.divide(bibo_mag_matrix, row_sums, where=row_sums > 0,
                              out=np.zeros_like(bibo_mag_matrix))

    sns.heatmap(bibo_mag_norm, ax=axes[0], cmap='Blues', vmin=0,
                xticklabels=expert_labels_bibo, yticklabels=mag_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Selection Probability'})
    axes[0].set_title('BiBo — Number Magnitude→Expert')
    axes[0].set_xlabel('Expert')
    axes[0].set_ylabel('Number Range')

    # Qwen
    qwen_mag_matrix = np.array([qwen_cat[c] for c in mag_cats])
    row_sums = qwen_mag_matrix.sum(axis=1, keepdims=True)
    qwen_mag_norm = np.divide(qwen_mag_matrix, row_sums, where=row_sums > 0,
                              out=np.zeros_like(qwen_mag_matrix))

    sns.heatmap(qwen_mag_norm, ax=axes[1], cmap='Oranges', vmin=0,
                xticklabels=expert_labels_qwen, yticklabels=mag_labels,
                annot=True, fmt='.2f', cbar_kws={'label': 'Selection Probability'})
    axes[1].set_title('Qwen3MoE — Number Magnitude→Expert')
    axes[1].set_xlabel('Expert')
    axes[1].set_ylabel('Number Range')

    fig.suptitle('Do Experts Specialize by Number Size? — Arithmetic Task',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_magnitude_expert_routing')

    return bibo_cat, qwen_cat, bibo_phase, qwen_phase


def plot_expert_confidence_by_token_type(bibo_model, qwen_model, val_data,
                                         device_bibo, device_qwen, bibo_cfg):
    """
    How confident is the router for different token types?
    Operators might get high-confidence routing (specialized experts),
    while numbers might be more spread out.
    """
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2

    bs = min(64, len(val_data))
    batch_np = val_data[:bs]
    batch = torch.tensor(batch_np[:, :-1], dtype=torch.long)
    seq_len = batch.shape[1]

    bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
    qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')

    # Classify positions
    categories = ['op_add', 'op_sub', 'op_mul', 'op_div',
                  'num_small', 'num_med', 'num_large', 'sep']
    cat_labels = ['+', '−', '×', '÷', 'Num\n[1-50]', 'Num\n[51-200]',
                  'Num\n[201-500]', 'SEP']

    position_categories = np.full((bs, seq_len), '', dtype=object)
    for sample_idx in range(bs):
        seq = batch_np[sample_idx, :-1]
        labels, _ = classify_tokens(seq)
        for pos, label in enumerate(labels[:seq_len]):
            position_categories[sample_idx, pos] = label

    def _gather_confidence(layer_data, cats_map):
        """Get top-1 weight (confidence) per token category."""
        cat_weights = {c: [] for c in categories}
        n_samples, n_positions = cats_map.shape
        for layer_idx, ld in layer_data.items():
            weights = ld['weights'].numpy()  # (bs, seq_len, top_k) or (bs, seq_len)
            if weights.ndim == 2:
                weights = weights[:, :, np.newaxis]
            for s in range(min(weights.shape[0], n_samples)):
                for p in range(min(weights.shape[1], n_positions)):
                    cat = cats_map[s, p]
                    if cat in cat_weights:
                        cat_weights[cat].append(weights[s, p, 0])  # top-1 weight
        return cat_weights

    bibo_conf = _gather_confidence(bibo_ld, position_categories)
    qwen_conf = _gather_confidence(qwen_ld, position_categories)

    # Box plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bibo_data = [bibo_conf[c] for c in categories if bibo_conf[c]]
    valid_cats_b = [cat_labels[i] for i, c in enumerate(categories) if bibo_conf[c]]
    if bibo_data:
        bp = axes[0].boxplot(bibo_data, labels=valid_cats_b, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(BIBO_COLOR)
            patch.set_alpha(0.6)
    axes[0].set_ylabel('Top-1 Router Weight (Confidence)')
    axes[0].set_title('BiBo — Router Confidence by Token Type')
    axes[0].tick_params(axis='x', rotation=0)

    qwen_data = [qwen_conf[c] for c in categories if qwen_conf[c]]
    valid_cats_q = [cat_labels[i] for i, c in enumerate(categories) if qwen_conf[c]]
    if qwen_data:
        bp = axes[1].boxplot(qwen_data, labels=valid_cats_q, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(QWEN_COLOR)
            patch.set_alpha(0.6)
    axes[1].set_ylabel('Top-1 Router Weight (Confidence)')
    axes[1].set_title('Qwen3MoE — Router Confidence by Token Type')
    axes[1].tick_params(axis='x', rotation=0)

    fig.suptitle('Router Confidence per Token Type — Arithmetic Task',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_confidence_by_token_type')


def plot_polyglue_activation_specialization(bibo_model, val_data,
                                            device_bibo, bibo_cfg):
    """
    BiBo-specific: Do different activation types (SiLU, ReLU², Tanh)
    specialize for different arithmetic operations?

    This is the key question for PolyGLU — does activation diversity
    actually lead to functional specialization on arithmetic?
    """
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2

    bs = min(64, len(val_data))
    batch_np = val_data[:bs]
    batch = torch.tensor(batch_np[:, :-1], dtype=torch.long)
    seq_len = batch.shape[1]

    bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')

    # Map expert index → activation type
    layout = get_expert_layout(bibo_cfg)
    expert_types = [t for _, t, _ in layout[:n_exp_bibo]]
    act_types = ['silu', 'relu2', 'tanh', 'identity', 'zero']

    # For each token category, count selections per activation type
    token_cats = ['op_add', 'op_sub', 'op_mul', 'op_div',
                  'num_small', 'num_med', 'num_large']
    cat_display = ['+', '−', '×', '÷', 'Small\nNums', 'Med\nNums', 'Large\nNums']

    position_categories = np.full((bs, seq_len), '', dtype=object)
    for sample_idx in range(bs):
        seq = batch_np[sample_idx, :-1]
        labels, _ = classify_tokens(seq)
        for pos, label in enumerate(labels[:seq_len]):
            position_categories[sample_idx, pos] = label

    # Count: cat → act_type → count
    cat_act_counts = {c: {a: 0 for a in act_types} for c in token_cats}

    for layer_idx, ld in bibo_ld.items():
        indices = ld['indices'].numpy()  # (bs, seq_len, top_k) or (bs, seq_len)
        if indices.ndim == 2:
            indices = indices[:, :, np.newaxis]
        for s in range(min(indices.shape[0], bs)):
            for p in range(min(indices.shape[1], seq_len)):
                cat = position_categories[s, p]
                if cat in cat_act_counts:
                    for k in range(indices.shape[2]):
                        exp_idx = indices[s, p, k]
                        if exp_idx < len(expert_types):
                            act = expert_types[exp_idx]
                            cat_act_counts[cat][act] += 1

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(token_cats))
    width = 0.7

    # Normalize per category
    bottoms = np.zeros(len(token_cats))
    act_colors = {
        'silu': POLYGLU_COLORS['silu'],
        'relu2': POLYGLU_COLORS['relu2'],
        'tanh': POLYGLU_COLORS['tanh'],
        'identity': SPECIAL_COLORS['identity'],
        'zero': SPECIAL_COLORS['zero'],
    }

    for act in act_types:
        values = []
        for cat in token_cats:
            total = sum(cat_act_counts[cat].values())
            values.append(cat_act_counts[cat][act] / total if total > 0 else 0)
        ax.bar(x, values, width, bottom=bottoms, label=act.upper(),
               color=act_colors[act], edgecolor='white', linewidth=0.5)
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_display)
    ax.set_ylabel('Fraction of Expert Selections')
    ax.set_xlabel('Token Category')
    ax.set_title('PolyGLU Activation Specialization — Arithmetic Task\n'
                 'Do SiLU/ReLU²/Tanh experts specialize for different operations?')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_figure(fig, 'arithmetic_polyglu_specialization')


def plot_deviation_heatmap_bs_vs_layer(bibo_model, qwen_model, val_data,
                                       device_bibo, device_qwen, bibo_cfg):
    """
    2D heatmap: batch_size (y) × layer (x) → deviation.
    Shows at a glance which layers need more samples to stabilize.
    """
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2
    n_exp_qwen = CFG['qwen3moe']['num_experts']

    batch_sizes = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 128]
    max_samples = min(128, len(val_data))

    bibo_grid = []  # rows=batch_sizes, cols=layers
    qwen_grid = []

    for bs in batch_sizes:
        if bs > max_samples:
            break
        batch = torch.tensor(val_data[:bs, :-1], dtype=torch.long)

        bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
        row_b = []
        for layer_idx in sorted(bibo_ld.keys()):
            indices = bibo_ld[layer_idx]['indices'].numpy().reshape(-1, bibo_ld[layer_idx]['indices'].shape[-1])
            mad, _ = compute_expert_deviation(indices, n_exp_bibo)
            row_b.append(mad)
        bibo_grid.append(row_b)

        qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')
        row_q = []
        for layer_idx in sorted(qwen_ld.keys()):
            indices = qwen_ld[layer_idx]['indices'].numpy().reshape(-1, qwen_ld[layer_idx]['indices'].shape[-1])
            mad, _ = compute_expert_deviation(indices, n_exp_qwen)
            row_q.append(mad)
        qwen_grid.append(row_q)

    actual_bs = batch_sizes[:len(bibo_grid)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if bibo_grid and bibo_grid[0]:
        bibo_arr = np.array(bibo_grid)
        n_layers_b = bibo_arr.shape[1]
        sns.heatmap(bibo_arr, ax=axes[0], cmap='YlOrRd',
                    xticklabels=[f'L{sorted(bibo_ld.keys())[i]}' for i in range(n_layers_b)],
                    yticklabels=actual_bs,
                    annot=True, fmt='.3f', cbar_kws={'label': 'MAD'})
        axes[0].set_xlabel('MoE Layer')
        axes[0].set_ylabel('Batch Size')
        axes[0].set_title(f'BiBo — Deviation Heatmap\n({n_exp_bibo} experts)')

    if qwen_grid and qwen_grid[0]:
        qwen_arr = np.array(qwen_grid)
        n_layers_q = qwen_arr.shape[1]
        sns.heatmap(qwen_arr, ax=axes[1], cmap='YlOrRd',
                    xticklabels=[f'L{sorted(qwen_ld.keys())[i]}' for i in range(n_layers_q)],
                    yticklabels=actual_bs,
                    annot=True, fmt='.3f', cbar_kws={'label': 'MAD'})
        axes[1].set_xlabel('MoE Layer')
        axes[1].set_ylabel('Batch Size')
        axes[1].set_title(f'Qwen3MoE — Deviation Heatmap\n({n_exp_qwen} experts)')

    fig.suptitle('Load Balance Deviation: Batch Size × Layer — Arithmetic (seq≈128)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'arithmetic_deviation_heatmap_bs_layer')


def main():
    print("=" * 70)
    print("  ARITHMETIC-SPECIALIZED ROUTER ANALYSIS — BiBo vs Qwen3MoE")
    print("=" * 70)

    bibo_cfg = CFG['bibo']
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2
    n_exp_qwen = CFG['qwen3moe']['num_experts']
    top_k = bibo_cfg['num_experts_per_tok']

    print(f"\n  BiBo: {poly_mult}×[SiLU,ReLU²,Tanh] + {special_pairs}×[Identity,Zero] = {n_exp_bibo} experts")
    print(f"  Qwen: {n_exp_qwen} homogeneous experts")
    print(f"  Top-K: {top_k}")

    # ── Load data ──
    # Use the "hard" bucket (19-30 terms → seq_len ~80-130, avg ~128)
    print("\n[1/8] Loading arithmetic data...")
    arith_cfg = CFG['training'].get('arithmetic', {})
    buckets = arith_cfg.get('buckets', [[3, 7], [9, 16], [19, 30], [35, 50]])

    # Find the bucket closest to seq_len 128 (hard bucket: 19-30 terms)
    target_bucket = None
    val_data = None
    for min_t, max_t in buckets:
        bucket_name = f'arith_{min_t}_{max_t}'
        path = os.path.join(DATA_DIR, f'val_{bucket_name}.npy')
        if os.path.exists(path):
            data = np.load(path)
            avg_len = data.shape[1]
            print(f"  Found: val_{bucket_name} — shape={data.shape}, seq_len={avg_len}")
            # Prefer bucket with seq_len closest to 128
            if avg_len >= 80 and (val_data is None or abs(avg_len - 128) < abs(val_data.shape[1] - 128)):
                val_data = data
                target_bucket = bucket_name
                print(f"    → Selected as primary (closest to seq≈128)")

    # Fallback: use largest available bucket
    if val_data is None:
        for min_t, max_t in reversed(buckets):
            bucket_name = f'arith_{min_t}_{max_t}'
            path = os.path.join(DATA_DIR, f'val_{bucket_name}.npy')
            if os.path.exists(path):
                val_data = np.load(path)
                target_bucket = bucket_name
                print(f"  Fallback: using {bucket_name} (shape={val_data.shape})")
                break

    if val_data is None:
        print("ERROR: No arithmetic validation data found. Run data_arithmetic.py first.")
        sys.exit(1)

    print(f"\n  Using bucket: {target_bucket}")
    print(f"  Samples available: {len(val_data)}")
    print(f"  Sequence length: {val_data.shape[1]}")

    # ── Load models ──
    print("\n[2/8] Loading models...")
    device_bibo = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_qwen = 'cuda:1' if torch.cuda.device_count() > 1 else device_bibo

    bibo_model = load_bibo(device_bibo, bibo_cfg)
    qwen_model = load_qwen(device_qwen)

    # ── Main analysis: Load balance vs batch size (1→128) ──
    print("\n[3/8] Load balance convergence (batch size 1→128)...")
    balance_metrics = plot_load_balance_vs_batch_size(
        bibo_model, qwen_model, val_data, device_bibo, device_qwen, bibo_cfg
    )

    # ── Per-layer deviation ──
    print("\n[4/8] Per-layer deviation vs batch size...")
    plot_per_layer_deviation_vs_batch_size(
        bibo_model, qwen_model, val_data, device_bibo, device_qwen, bibo_cfg
    )

    # ── Per-layer top-1 and top-2 weights ──
    print("\n[5/8] Per-layer top-1 & top-2 router weights...")
    bibo_topk_weights, qwen_topk_weights = plot_per_layer_topk_weights(
        bibo_model, qwen_model, val_data, device_bibo, device_qwen, bibo_cfg
    )

    # ── Operator/Phase/Magnitude affinity ──
    print("\n[6/8] Operator → Expert affinity analysis...")
    bibo_cat, qwen_cat, bibo_phase, qwen_phase = plot_operator_expert_affinity(
        bibo_model, qwen_model, val_data, device_bibo, device_qwen, bibo_cfg
    )

    # ── Confidence by token type ──
    print("\n[7/8] Router confidence by token type...")
    plot_expert_confidence_by_token_type(
        bibo_model, qwen_model, val_data, device_bibo, device_qwen, bibo_cfg
    )

    # ── PolyGLU specialization ──
    print("\n[8/8] PolyGLU activation specialization...")
    plot_polyglue_activation_specialization(
        bibo_model, val_data, device_bibo, bibo_cfg
    )

    # ── Bonus: Deviation heatmap ──
    print("\n[BONUS] Deviation heatmap (batch_size × layer)...")
    plot_deviation_heatmap_bs_vs_layer(
        bibo_model, qwen_model, val_data, device_bibo, device_qwen, bibo_cfg
    )

    # ── Save metrics ──
    print("\n  Saving metrics...")
    metrics_out = {
        'task': 'arithmetic',
        'target_bucket': target_bucket,
        'seq_len': int(val_data.shape[1]),
        'config': {
            'bibo': {
                'polyglu_expert_multiplier': poly_mult,
                'special_expert_pairs': special_pairs,
                'num_routed_experts': n_exp_bibo,
                'num_experts_per_tok': top_k,
                'router_type': bibo_cfg.get('router_type', 'mlp'),
                'router_lambda': bibo_cfg.get('router_lambda', 1.0),
            },
            'qwen3moe': {
                'num_experts': n_exp_qwen,
                'num_experts_per_tok': CFG['qwen3moe']['num_experts_per_tok'],
            }
        },
        'load_balance_convergence': balance_metrics,
        'operator_affinity': {
            'bibo': {k: v.tolist() if hasattr(v, 'tolist') else v
                     for k, v in bibo_cat.items()},
            'qwen': {k: v.tolist() if hasattr(v, 'tolist') else v
                     for k, v in qwen_cat.items()},
        },
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"  Saved → {METRICS_OUT}")

    print("\n" + "=" * 70)
    print("  ARITHMETIC ROUTER ANALYSIS COMPLETE")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Metrics: {METRICS_OUT}")
    print("=" * 70)
    print("\n  Key plots generated:")
    print("    • arithmetic_load_balance_vs_batch_size.png  ← THE MAIN ONE")
    print("    • arithmetic_per_layer_deviation_vs_bs.png")
    print("    • arithmetic_per_layer_topk_weights.png      ← TOP-1/TOP-2 WEIGHTS")
    print("    • arithmetic_operator_expert_affinity.png")
    print("    • arithmetic_phase_expert_routing.png")
    print("    • arithmetic_magnitude_expert_routing.png")
    print("    • arithmetic_confidence_by_token_type.png")
    print("    • arithmetic_polyglu_specialization.png")
    print("    • arithmetic_deviation_heatmap_bs_layer.png")


if __name__ == '__main__':
    main()
