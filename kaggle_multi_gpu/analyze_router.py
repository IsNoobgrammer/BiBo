"""
Comprehensive Router Analysis — BiBo vs Qwen3MoE
=================================================

Multi-dimensional comparison of routing behavior across:
- Multiple batch sizes (1, 5, 20, 64)
- Multiple sequence lengths (64, 128, 256)
- Per-position-type analysis (unsorted vs sorted tokens)
- Expert co-selection patterns
- Load balance metrics (Gini, entropy, CV)
- Routing weight distributions
- Expert specialization scores
- Side-by-side comparative visualizations

Usage:
    python kaggle_multi_gpu/analyze_router.py

Produces ~30+ publication-quality plots in kaggle_multi_gpu/plots/
"""
import sys, torch, numpy as np, json, os, math
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# scipy optional — not required for core analysis

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
METRICS_OUT = os.path.join(BASE_DIR, 'metrics', 'router_analysis.json')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'metrics'), exist_ok=True)

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

# ============================================================
# Aesthetic config
# ============================================================
BIBO_COLOR = '#2196F3'       # blue
QWEN_COLOR = '#FF5722'       # deep orange
BIBO_CMAP = 'Blues'
QWEN_CMAP = 'Oranges'
ACCENT_COLORS = ['#4CAF50', '#9C27B0', '#FF9800', '#00BCD4', '#E91E63', '#8BC34A']

# Expert type colors for BiBo
EXPERT_COLORS_BIBO = None  # set dynamically based on n_experts

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

sns.set_theme(style='whitegrid', palette='muted')


def get_expert_colors(n_experts, model_name):
    """Color palette: MLPs=blues, special experts=distinct."""
    if model_name == 'BiBo':
        n_mlp = n_experts - 3
        mlp_colors = sns.color_palette('Blues_d', n_mlp)
        special = ['#4CAF50', '#9E9E9E', '#FF5722']  # Identity=green, Zero=gray, ReLU²=red
        return list(mlp_colors) + special
    else:
        return sns.color_palette('husl', n_experts)


def get_expert_labels(n_experts, model_name):
    """Expert labels."""
    if model_name == 'BiBo':
        n_mlp = n_experts - 3
        return [f'MLP{i}' for i in range(n_mlp)] + ['Identity', 'Zero', 'ReLU²']
    else:
        return [f'E{i}' for i in range(n_experts)]


# ============================================================
# Extraction hooks
# ============================================================

def extract_routing_data(model, input_ids, device, model_type='bibo'):
    """
    Extract full routing data for a batch.
    
    Returns dict per MoE layer:
        indices: [batch, seq_len, top_k]
        weights: [batch, seq_len, top_k]
        logits:  [batch*seq_len, n_experts] (raw router output before topk)
    """
    model.eval()
    layer_data = {}
    hooks = []

    def make_hook(layer_idx, mtype):
        def hook_fn(module, inp, output):
            if mtype == 'bibo':
                indices, weights = output  # [bs, seq, top_k]
                layer_data[layer_idx] = {
                    'indices': indices.detach().cpu(),
                    'weights': weights.detach().cpu().float(),
                }
            else:
                logits, scores, indices = output
                bs_seq = logits.shape[0]
                # Qwen router flattens batch*seq → need to reshape
                layer_data[layer_idx] = {
                    'indices': indices.detach().cpu(),   # [bs*seq, top_k]
                    'weights': scores.detach().cpu().float(),
                    'logits': logits.detach().cpu().float(),
                }
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i, model_type)))

    with torch.no_grad():
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        model(input_ids=input_ids.to(device))

    for h in hooks:
        h.remove()
    return layer_data


# ============================================================
# Metrics computation
# ============================================================

def compute_gini(counts):
    """Gini coefficient: 0=perfect equality, 1=max inequality."""
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    if sorted_c.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_c) - (n + 1) * np.sum(sorted_c)) / (n * np.sum(sorted_c))


def compute_load_balance_metrics(indices, n_experts, top_k):
    """
    Compute comprehensive load balance metrics from routing indices.
    
    Args:
        indices: flat array of expert indices
        n_experts: total experts
        top_k: experts per token
    
    Returns dict with: entropy, gini, cv, max_load, min_load, balance_ratio
    """
    counts = np.bincount(indices.flatten(), minlength=n_experts).astype(float)
    total = counts.sum()
    if total == 0:
        return {'entropy': 0, 'gini': 0, 'cv': 0, 'max_load': 0, 'min_load': 0, 'balance_ratio': 0}
    
    dist = counts / total
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    max_entropy = np.log(n_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    gini = compute_gini(counts)
    cv = np.std(counts) / (np.mean(counts) + 1e-10)  # coefficient of variation
    
    return {
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'gini': float(gini),
        'cv': float(cv),
        'max_load': float(counts.max() / total),
        'min_load': float(counts.min() / total),
        'balance_ratio': float(counts.min() / (counts.max() + 1e-10)),
        'counts': counts.tolist(),
    }


def compute_expert_coselection(indices, n_experts):
    """
    Co-selection matrix: how often pairs of experts are selected together for same token.
    
    Args:
        indices: [n_tokens, top_k]
    Returns:
        cosel: [n_experts, n_experts] symmetric matrix (normalized)
    """
    cosel = np.zeros((n_experts, n_experts))
    for row in indices:
        for i, j in combinations(row, 2):
            cosel[i, j] += 1
            cosel[j, i] += 1
    # Normalize by total pairs
    total_pairs = len(indices) * (indices.shape[1] * (indices.shape[1] - 1) / 2)
    if total_pairs > 0:
        cosel /= total_pairs
    return cosel


def compute_specialization_score(indices, n_experts, seq_len):
    """
    Expert specialization: do experts prefer certain positions?
    Score = KL divergence of position distribution from uniform.
    
    Returns per-expert specialization scores.
    """
    # Position distribution per expert
    scores = np.zeros(n_experts)
    positions = np.arange(seq_len)
    uniform = np.ones(seq_len) / seq_len
    
    for exp_id in range(n_experts):
        # Find positions where this expert is selected (any top-k slot)
        mask = (indices == exp_id).any(axis=-1)  # [n_tokens]
        if mask.sum() == 0:
            scores[exp_id] = 0
            continue
        # Position histogram
        pos_counts = np.zeros(seq_len)
        token_positions = np.arange(len(mask)) % seq_len  # handle batched
        selected_positions = token_positions[mask]
        for p in selected_positions:
            pos_counts[p] += 1
        pos_dist = pos_counts / (pos_counts.sum() + 1e-10)
        # KL(pos_dist || uniform)
        kl = np.sum(pos_dist * np.log((pos_dist + 1e-10) / uniform))
        scores[exp_id] = kl
    
    return scores



# ============================================================
# PLOT 1: Expert Usage Heatmap — Multi-batch sweep
# ============================================================

def plot_expert_usage_sweep(all_data, model_name, n_experts, batch_sizes, seq_lens):
    """
    Grid plot: rows=seq_lens, cols=batch_sizes.
    Each cell = expert usage bar chart.
    Shows how usage stabilizes with more samples.
    """
    labels = get_expert_labels(n_experts, model_name)
    colors = get_expert_colors(n_experts, model_name)
    
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
            bars = ax.bar(range(n_experts), dist, color=colors, edgecolor='white', linewidth=0.5)
            
            # Uniform line
            ax.axhline(y=1/n_experts, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Metrics annotation
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

def plot_comparative_usage(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len, batch_size):
    """Side-by-side bar charts comparing BiBo vs Qwen expert usage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # BiBo
    bibo_labels = get_expert_labels(n_exp_bibo, 'BiBo')
    bibo_colors = get_expert_colors(n_exp_bibo, 'BiBo')
    bibo_counts = np.zeros(n_exp_bibo)
    for ld in bibo_data.values():
        bibo_counts += np.bincount(ld['indices'].numpy().flatten(), minlength=n_exp_bibo)
    bibo_dist = bibo_counts / (bibo_counts.sum() + 1e-10)
    
    ax1.bar(range(n_exp_bibo), bibo_dist, color=bibo_colors, edgecolor='white', linewidth=0.8)
    ax1.axhline(y=1/n_exp_bibo, color='red', linestyle='--', alpha=0.5, label='Uniform')
    ax1.set_xticks(range(n_exp_bibo))
    ax1.set_xticklabels(bibo_labels, rotation=45)
    ax1.set_ylabel('Token Fraction')
    ax1.set_title('BiBo (Skywork-MoE + Conv Router)', color=BIBO_COLOR, fontweight='bold')
    ax1.legend()
    
    # Qwen
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
    
    # Metrics comparison text
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
    
    # Collect all weights
    bibo_top1 = []
    bibo_topk_mean = []
    qwen_top1 = []
    qwen_topk_mean = []
    
    for ld in bibo_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:  # [bs, seq, top_k]
            w = w.reshape(-1, w.shape[-1])
        bibo_top1.extend(w[:, 0].tolist())
        bibo_topk_mean.extend(w.mean(axis=1).tolist())
    
    for ld in qwen_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        qwen_top1.extend(w[:, 0].tolist())
        qwen_topk_mean.extend(w.mean(axis=1).tolist())
    
    # Top-1 weight distribution
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
               label=f'Uniform (1/top_k={1/CFG["bibo"]["num_experts_per_tok"]:.2f})')
    ax.legend()
    
    # Weight spread (std across top-k)
    ax = axes[1]
    bibo_std = []
    qwen_std = []
    for ld in bibo_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        bibo_std.extend(w.std(axis=1).tolist())
    for ld in qwen_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        qwen_std.extend(w.std(axis=1).tolist())
    
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

def plot_coselection_matrix(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len):
    """Heatmap showing which expert pairs are frequently co-selected."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BiBo
    bibo_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in bibo_data.values()
    ])
    bibo_cosel = compute_expert_coselection(bibo_indices, n_exp_bibo)
    bibo_labels = get_expert_labels(n_exp_bibo, 'BiBo')
    
    mask = np.triu(np.ones_like(bibo_cosel, dtype=bool), k=1)
    sns.heatmap(bibo_cosel, mask=~mask, ax=ax1, cmap='YlOrRd', annot=True, fmt='.3f',
                xticklabels=bibo_labels, yticklabels=bibo_labels, square=True,
                cbar_kws={'label': 'Co-selection freq', 'shrink': 0.8})
    ax1.set_title('BiBo — Expert Co-Selection', color=BIBO_COLOR, fontweight='bold')
    
    # Qwen
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

def plot_position_type_routing(layer_data, model_name, n_experts, seq_len):
    """
    Compare routing for unsorted tokens (input) vs sorted tokens (output).
    Sorting task: [unsorted | SEP | sorted] — first half is input, second half is output.
    """
    labels = get_expert_labels(n_experts, model_name)
    colors = get_expert_colors(n_experts, model_name)
    
    # Split point: seq_len tokens unsorted, 1 SEP, seq_len tokens sorted
    # In input_ids (shifted), split is at position seq_len (0-indexed)
    half = seq_len  # first `seq_len` positions = unsorted, rest = sorted
    
    layers = sorted(layer_data.keys())
    fig, axes = plt.subplots(len(layers), 2, figsize=(12, 3*len(layers)), squeeze=False)
    
    for row, l in enumerate(layers):
        idx = layer_data[l]['indices'].numpy()
        if idx.ndim == 3:  # [bs, seq, top_k]
            # Unsorted portion
            unsorted_idx = idx[:, :half, :].reshape(-1)
            sorted_idx = idx[:, half:, :].reshape(-1)
        else:  # [seq, top_k]
            unsorted_idx = idx[:half, :].flatten()
            sorted_idx = idx[half:, :].flatten()
        
        # Unsorted
        ax = axes[row, 0]
        counts_u = np.bincount(unsorted_idx, minlength=n_experts).astype(float)
        dist_u = counts_u / (counts_u.sum() + 1e-10)
        ax.bar(range(n_experts), dist_u, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=1/n_experts, color='red', linestyle='--', alpha=0.4)
        ax.set_title(f'L{l} — Unsorted (input)' if row == 0 else f'L{l}')
        ax.set_ylim(0, max(dist_u.max() * 1.3, 2/n_experts))
        if row == len(layers) - 1:
            ax.set_xticks(range(n_experts))
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
        else:
            ax.set_xticks([])
        
        # Sorted
        ax = axes[row, 1]
        counts_s = np.bincount(sorted_idx, minlength=n_experts).astype(float)
        dist_s = counts_s / (counts_s.sum() + 1e-10)
        ax.bar(range(n_experts), dist_s, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=1/n_experts, color='red', linestyle='--', alpha=0.4)
        ax.set_title(f'L{l} — Sorted (output)' if row == 0 else f'L{l}')
        ax.set_ylim(0, max(dist_s.max() * 1.3, 2/n_experts))
        if row == len(layers) - 1:
            ax.set_xticks(range(n_experts))
            ax.set_xticklabels(labels, rotation=45, fontsize=7)
        else:
            ax.set_xticks([])
    
    fig.suptitle(f'{model_name} — Routing by Token Type (Unsorted Input vs Sorted Output)\nseq_len={seq_len}',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'position_type_routing_{model_name}_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ position_type_routing_{model_name}_seq{seq_len}.png")


# ============================================================
# PLOT 6: Expert specialization radar chart
# ============================================================

def plot_specialization_radar(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len):
    """Radar/polar chart showing per-expert position specialization (KL from uniform)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(polar=True))
    
    # BiBo
    bibo_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in bibo_data.values()
    ])
    # Use first layer's seq_len
    first_layer = list(bibo_data.values())[0]
    if first_layer['indices'].ndim == 3:
        sl = first_layer['indices'].shape[1]
    else:
        sl = first_layer['indices'].shape[0]
    bibo_spec = compute_specialization_score(bibo_indices, n_exp_bibo, sl)
    bibo_labels = get_expert_labels(n_exp_bibo, 'BiBo')
    
    angles = np.linspace(0, 2*np.pi, n_exp_bibo, endpoint=False).tolist()
    angles += angles[:1]
    values = bibo_spec.tolist() + [bibo_spec[0]]
    
    ax1.plot(angles, values, 'o-', color=BIBO_COLOR, linewidth=2)
    ax1.fill(angles, values, alpha=0.25, color=BIBO_COLOR)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(bibo_labels, fontsize=7)
    ax1.set_title('BiBo — Position Specialization\n(KL from uniform)', fontsize=10, fontweight='bold', pad=20)
    
    # Qwen
    qwen_indices = np.concatenate([
        ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in qwen_data.values()
    ])
    first_layer_q = list(qwen_data.values())[0]
    if first_layer_q['indices'].ndim == 3:
        sl_q = first_layer_q['indices'].shape[1]
    else:
        sl_q = first_layer_q['indices'].shape[0]
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
# PLOT 7: Token-level expert heatmap (improved)
# ============================================================

def plot_token_expert_heatmap_v2(layer_data, model_name, n_experts, seq_len_label):
    """
    Improved heatmap: x=token position, y=layer.
    Color = primary expert. Annotated with position type regions.
    """
    layers = sorted(layer_data.keys())
    n_layers = len(layers)
    first_ld = layer_data[layers[0]]
    
    if first_ld['indices'].ndim == 3:
        actual_seq = first_ld['indices'].shape[1]
        heatmap = np.zeros((n_layers, actual_seq))
        for i, l in enumerate(layers):
            heatmap[i] = layer_data[l]['indices'][0, :, 0].numpy()  # first sample, top-1
    else:
        actual_seq = first_ld['indices'].shape[0]
        heatmap = np.zeros((n_layers, actual_seq))
        for i, l in enumerate(layers):
            heatmap[i] = layer_data[l]['indices'][:, 0].numpy()
    
    # Custom discrete colormap
    cmap = plt.cm.get_cmap('tab10', n_experts)
    
    fig, ax = plt.subplots(figsize=(min(18, max(10, actual_seq//8)), max(3, n_layers*0.8)))
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=-0.5, vmax=n_experts-0.5,
                   interpolation='nearest')
    
    # Separator line (between unsorted and sorted)
    half = int(seq_len_label)
    if half < actual_seq:
        ax.axvline(x=half, color='white', linewidth=2, linestyle='-')
        ax.text(half/2, -0.8, 'Unsorted', ha='center', fontsize=9, color='gray')
        ax.text(half + (actual_seq-half)/2, -0.8, 'Sorted', ha='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('MoE Layer')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{l}' for l in layers])
    
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_experts))
    labels = get_expert_labels(n_experts, model_name)
    cbar.set_ticklabels(labels)
    cbar.set_label('Expert ID')
    
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
    
    # BiBo
    bibo_layers = sorted(bibo_data.keys())
    for l in bibo_layers:
        w = bibo_data[l]['weights'].numpy()
        if w.ndim == 3:
            max_w = w[:, :, 0].mean(axis=0)  # avg top-1 across batch
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
    
    # Qwen
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
    
    # Separator
    half = seq_len
    for ax in [ax1, ax2]:
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
    
    top_k_bibo = CFG['bibo']['num_experts_per_tok']
    top_k_qwen = CFG['qwen3moe']['num_experts_per_tok']
    
    # Average across layers for cleaner comparison
    bibo_entropies = []
    for l, ld in sorted(bibo_data.items()):
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.mean(axis=0)  # avg across batch
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
    
    ax.axhline(y=np.log(top_k_bibo), color=BIBO_COLOR, linestyle='--', alpha=0.4,
               label=f'Max entropy (top-{top_k_bibo})')
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
# PLOT 10: Load balance metrics summary (bar chart)
# ============================================================

def plot_load_balance_summary(all_metrics):
    """
    Summary bar chart comparing Gini, normalized entropy, CV across
    all (model, seq_len, batch_size) combinations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = ['gini', 'normalized_entropy', 'cv']
    metric_labels = ['Gini Coefficient\n(lower = more balanced)', 
                     'Normalized Entropy\n(higher = more balanced)',
                     'Coefficient of Variation\n(lower = more balanced)']
    
    for ax, mname, mlabel in zip(axes, metric_names, metric_labels):
        bibo_vals = []
        qwen_vals = []
        x_labels = []
        
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
    """
    How much weight goes to rank-1, rank-2, ..., rank-K expert?
    Shows if top-1 dominates (Qwen) vs distributed (BiBo).
    """
    top_k = CFG['bibo']['num_experts_per_tok']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # BiBo: avg weight per rank
    bibo_rank_weights = np.zeros(top_k)
    bibo_count = 0
    for ld in bibo_data.values():
        w = ld['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        bibo_rank_weights += w.mean(axis=0)
        bibo_count += 1
    bibo_rank_weights /= bibo_count
    
    # Qwen
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
    
    # Annotate values
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
                 'Flatter = better expert utilization (all top-K contribute equally)',
                 fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(bibo_rank_weights.max(), qwen_rank_weights.max()) * 1.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'weight_rank_distribution_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ weight_rank_distribution_seq{seq_len}.png")


# ============================================================
# PLOT 12: Per-layer routing diversity (how many unique experts used)
# ============================================================

def plot_routing_diversity(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len, batch_size):
    """
    Per-layer: what fraction of experts are actually used?
    + effective number of experts (exp(entropy))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # BiBo
    bibo_layers = sorted(bibo_data.keys())
    bibo_effective = []
    bibo_used_frac = []
    for l in bibo_layers:
        idx = bibo_data[l]['indices'].numpy().flatten()
        counts = np.bincount(idx, minlength=n_exp_bibo).astype(float)
        dist = counts / (counts.sum() + 1e-10)
        entropy = -np.sum(dist * np.log(dist + 1e-10))
        effective_n = np.exp(entropy)
        used = (counts > 0).sum()
        bibo_effective.append(effective_n)
        bibo_used_frac.append(used / n_exp_bibo)
    
    # Qwen
    qwen_layers = sorted(qwen_data.keys())
    qwen_effective = []
    qwen_used_frac = []
    for l in qwen_layers:
        idx = qwen_data[l]['indices'].numpy().flatten()
        counts = np.bincount(idx, minlength=n_exp_qwen).astype(float)
        dist = counts / (counts.sum() + 1e-10)
        entropy = -np.sum(dist * np.log(dist + 1e-10))
        effective_n = np.exp(entropy)
        used = (counts > 0).sum()
        qwen_effective.append(effective_n)
        qwen_used_frac.append(used / n_exp_qwen)
    
    # Effective experts
    x_b = np.arange(len(bibo_layers))
    x_q = np.arange(len(qwen_layers))
    ax1.bar(x_b - 0.2, bibo_effective, 0.4, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax1.bar(x_q + 0.2, qwen_effective, 0.4, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    ax1.axhline(y=n_exp_bibo, color='gray', linestyle='--', alpha=0.4, label=f'Max ({n_exp_bibo})')
    ax1.set_xticks(x_b)
    ax1.set_xticklabels([f'L{l}' for l in bibo_layers])
    ax1.set_ylabel('Effective # Experts (exp(H))')
    ax1.set_title('Effective Expert Count per Layer', fontweight='bold')
    ax1.legend()
    
    # Used fraction
    ax2.bar(x_b - 0.2, bibo_used_frac, 0.4, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax2.bar(x_q + 0.2, qwen_used_frac, 0.4, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax2.set_xticks(x_b)
    ax2.set_xticklabels([f'L{l}' for l in bibo_layers])
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
# PLOT 13: Routing stability — how consistent is routing for similar inputs?
# ============================================================

def plot_routing_stability(model, device, model_type, model_name, n_experts, val_data, seq_len):
    """
    Take N similar sequences, measure routing agreement (Jaccard similarity of expert sets).
    Shows if router is stable or noisy.
    """
    # Pick 20 sequences from same bucket
    n_samples = min(20, len(val_data))
    samples = torch.tensor(val_data[:n_samples, :-1], dtype=torch.long)
    
    # Extract routing for each
    all_indices = []
    for i in range(n_samples):
        ld = extract_routing_data(model, samples[i], device, model_type)
        # Flatten all layers into one set per token
        first_layer = list(ld.values())[0]
        idx = first_layer['indices']
        if idx.dim() == 3:
            idx = idx[0]  # [seq, top_k]
        all_indices.append(idx.numpy())
    
    # Pairwise Jaccard similarity of expert sets
    n_tokens = all_indices[0].shape[0]
    jaccard_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i, n_samples):
            # Per-token Jaccard, then average
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
    
    # Overall stability score
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
# PLOT 14: Grand summary dashboard
# ============================================================

def plot_grand_summary(all_metrics, stability_scores):
    """
    Single-page dashboard summarizing key findings.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # --- Panel 1: Key metrics comparison ---
    ax = fig.add_subplot(gs[0, :])
    metrics_to_show = ['gini', 'normalized_entropy', 'cv']
    labels_show = ['Gini ↓', 'Norm. Entropy ↑', 'CV ↓']
    
    # Average across all configs
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
    
    # --- Panel 2: Stability scores ---
    ax2 = fig.add_subplot(gs[1, 0])
    bibo_stab = [v for k, v in stability_scores.items() if 'BiBo' in k]
    qwen_stab = [v for k, v in stability_scores.items() if 'Qwen' in k]
    
    if bibo_stab and qwen_stab:
        ax2.bar(['BiBo', 'Qwen3MoE'], [np.mean(bibo_stab), np.mean(qwen_stab)],
                color=[BIBO_COLOR, QWEN_COLOR], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Avg Jaccard Similarity')
        ax2.set_title('Routing Stability', fontweight='bold')
        ax2.set_ylim(0, 1)
    
    # --- Panel 3: Winner annotation ---
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
    
    findings.append(f"\nBiBo: Skywork-MoE norm (λ={CFG['bibo']['router_lambda']}) + Conv router")
    findings.append(f"Qwen: Raw softmax + linear projection")
    findings.append(f"\nKey insight: Skywork normalization prevents top-1 dominance")
    findings.append(f"→ All top-K experts contribute meaningfully")
    
    text = '\n'.join(findings)
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Key Findings', fontweight='bold', fontsize=12)
    
    # --- Panel 4: Config comparison ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    config_text = (
        f"{'='*80}\n"
        f"{'Config':<20} {'BiBo':<30} {'Qwen3MoE':<30}\n"
        f"{'='*80}\n"
        f"{'Experts':<20} {CFG['bibo']['num_routed_experts']:<30} {CFG['qwen3moe']['num_experts']:<30}\n"
        f"{'Top-K':<20} {CFG['bibo']['num_experts_per_tok']:<30} {CFG['qwen3moe']['num_experts_per_tok']:<30}\n"
        f"{'Router':<20} {'Conv (k=3) + Skywork norm':<30} {'Linear + softmax':<30}\n"
        f"{'Expert types':<20} {'MLP×5 + Id + Zero + ReLU²':<30} {'MLP×8 (homogeneous)':<30}\n"
        f"{'Shared expert':<20} {'CausalConv1D (always on)':<30} {'MLP (always on)':<30}\n"
        f"{'Load balance':<20} {'Heuristic bias update':<30} {'Aux loss (Switch)':<30}\n"
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
# PLOT 15: Per-layer weight distribution (KDE)
# ============================================================

def plot_per_layer_weight_kde(bibo_data, qwen_data, seq_len):
    """KDE of routing weights per layer — shows distribution shape."""
    bibo_layers = sorted(bibo_data.keys())
    qwen_layers = sorted(qwen_data.keys())
    n_layers = max(len(bibo_layers), len(qwen_layers))
    
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 3*n_layers), squeeze=False)
    
    for i, (bl, ql) in enumerate(zip(bibo_layers, qwen_layers)):
        # BiBo
        ax = axes[i, 0]
        w = bibo_data[bl]['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        for k in range(w.shape[1]):
            sns.kdeplot(w[:, k], ax=ax, label=f'Rank-{k+1}', linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_title(f'BiBo L{bl}', color=BIBO_COLOR, fontweight='bold')
        ax.set_xlabel('Weight')
        if i == 0:
            ax.legend(fontsize=7)
        
        # Qwen
        ax = axes[i, 1]
        w = qwen_data[ql]['weights'].numpy()
        if w.ndim == 3:
            w = w.reshape(-1, w.shape[-1])
        for k in range(w.shape[1]):
            sns.kdeplot(w[:, k], ax=ax, label=f'Rank-{k+1}', linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_title(f'Qwen L{ql}', color=QWEN_COLOR, fontweight='bold')
        ax.set_xlabel('Weight')
        if i == 0:
            ax.legend(fontsize=7)
    
    plt.suptitle(f'Per-Layer Weight Distribution (KDE) — seq={seq_len}\n'
                 'Peaked near 1.0 = top-1 dominance | Spread = balanced contribution',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'weight_kde_per_layer_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ weight_kde_per_layer_seq{seq_len}.png")


# ============================================================
# PLOT 16: Expert switching rate
# ============================================================

def plot_expert_switching_rate(bibo_data, qwen_data, n_exp_bibo, n_exp_qwen, seq_len):
    """
    How often does the top-1 expert change between consecutive tokens?
    High switching = router is position-sensitive. Low = sticky routing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # BiBo
    bibo_layers = sorted(bibo_data.keys())
    bibo_switch_rates = []
    for l in bibo_layers:
        idx = bibo_data[l]['indices'].numpy()
        if idx.ndim == 3:
            # Average across batch
            switches = []
            for b in range(idx.shape[0]):
                top1 = idx[b, :, 0]
                switch = (top1[1:] != top1[:-1]).mean()
                switches.append(switch)
            bibo_switch_rates.append(np.mean(switches))
        else:
            top1 = idx[:, 0]
            bibo_switch_rates.append((top1[1:] != top1[:-1]).mean())
    
    # Qwen
    qwen_layers = sorted(qwen_data.keys())
    qwen_switch_rates = []
    for l in qwen_layers:
        idx = qwen_data[l]['indices'].numpy()
        if idx.ndim == 3:
            switches = []
            for b in range(idx.shape[0]):
                top1 = idx[b, :, 0]
                switch = (top1[1:] != top1[:-1]).mean()
                switches.append(switch)
            qwen_switch_rates.append(np.mean(switches))
        else:
            top1 = idx[:, 0]
            qwen_switch_rates.append((top1[1:] != top1[:-1]).mean())
    
    x = np.arange(len(bibo_layers))
    width = 0.35
    ax1.bar(x - width/2, bibo_switch_rates, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax1.bar(x + width/2, qwen_switch_rates, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{l}' for l in bibo_layers])
    ax1.set_ylabel('Switch Rate')
    ax1.set_title('Top-1 Expert Switch Rate per Layer\n(fraction of consecutive tokens with different top-1)',
                  fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Expected switch rate for random routing
    expected_random_bibo = 1 - 1/n_exp_bibo
    expected_random_qwen = 1 - 1/n_exp_qwen
    ax1.axhline(y=expected_random_bibo, color=BIBO_COLOR, linestyle='--', alpha=0.4,
                label=f'Random BiBo ({expected_random_bibo:.2f})')
    ax1.axhline(y=expected_random_qwen, color=QWEN_COLOR, linestyle='--', alpha=0.4,
                label=f'Random Qwen ({expected_random_qwen:.2f})')
    
    # Per-position switch visualization (first sample, first MoE layer)
    ax2_layer = bibo_layers[0]
    bibo_idx = bibo_data[ax2_layer]['indices'].numpy()
    qwen_idx = qwen_data[qwen_layers[0]]['indices'].numpy()
    
    if bibo_idx.ndim == 3:
        bibo_top1 = bibo_idx[0, :, 0]
        qwen_top1 = qwen_idx[0, :, 0]
    else:
        bibo_top1 = bibo_idx[:, 0]
        qwen_top1 = qwen_idx[:, 0]
    
    bibo_switches = np.where(bibo_top1[1:] != bibo_top1[:-1])[0]
    qwen_switches = np.where(qwen_top1[1:] != qwen_top1[:-1])[0]
    
    ax2.eventplot([bibo_switches, qwen_switches], lineoffsets=[1, 0],
                  linelengths=0.8, colors=[BIBO_COLOR, QWEN_COLOR])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Qwen3MoE', 'BiBo'])
    ax2.set_xlabel('Token Position')
    ax2.set_title(f'Switch Events (L{ax2_layer}, first sample)\n'
                  'Each tick = top-1 expert changed', fontweight='bold')
    ax2.axvline(x=seq_len, color='green', linestyle=':', alpha=0.6, label='SEP')
    ax2.legend()
    
    plt.suptitle(f'Expert Switching Analysis — seq={seq_len}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'switching_rate_seq{seq_len}.png'))
    plt.close()
    print(f"  ✓ switching_rate_seq{seq_len}.png")


# ============================================================
# PLOT 17: BiBo special expert analysis
# ============================================================

def plot_special_expert_analysis(bibo_data_dict, n_experts, seq_lens):
    """
    BiBo-specific: how are Identity, Zero, ReLU² experts used?
    Across seq_lens and layers.
    """
    n_mlp = n_experts - 3
    special_names = ['Identity', 'Zero', 'ReLU²']
    special_ids = [n_mlp, n_mlp + 1, n_mlp + 2]
    special_colors = ['#4CAF50', '#9E9E9E', '#FF5722']
    
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5*len(seq_lens), 5), squeeze=False)
    
    for col, sl in enumerate(seq_lens):
        ax = axes[0, col]
        if sl not in bibo_data_dict:
            ax.set_visible(False)
            continue
        
        data = bibo_data_dict[sl]
        layers = sorted(data.keys())
        
        # Per-layer usage of each special expert
        usage_matrix = np.zeros((3, len(layers)))
        for i, l in enumerate(layers):
            idx = data[l]['indices'].numpy().flatten()
            total = len(idx)
            for j, exp_id in enumerate(special_ids):
                usage_matrix[j, i] = (idx == exp_id).sum() / total
        
        x = np.arange(len(layers))
        bottom = np.zeros(len(layers))
        for j in range(3):
            ax.bar(x, usage_matrix[j], bottom=bottom, color=special_colors[j],
                   label=special_names[j], edgecolor='white', linewidth=0.5)
            bottom += usage_matrix[j]
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_ylabel('Fraction of Tokens')
        ax.set_title(f'seq={sl}', fontweight='bold')
        if col == 0:
            ax.legend()
    
    plt.suptitle('BiBo — Special Expert Usage (Identity + Zero + ReLU²)\n'
                 'These provide architectural diversity beyond standard SwiGLU MLPs',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'special_expert_analysis.png'))
    plt.close()
    print(f"  ✓ special_expert_analysis.png")



# ============================================================
# Main orchestrator
# ============================================================

def main():
    print("=" * 70)
    print("  COMPREHENSIVE ROUTER ANALYSIS — BiBo vs Qwen3MoE")
    print("=" * 70)
    
    n_exp_bibo = CFG['bibo']['num_routed_experts']
    n_exp_qwen = CFG['qwen3moe']['num_experts']
    top_k = CFG['bibo']['num_experts_per_tok']
    
    # Load validation data (all buckets)
    print("\n[1/5] Loading data...")
    val_data = {}
    for sl in [64, 128, 256]:
        path = os.path.join(BASE_DIR, 'data', f'val_len_{sl}.npy')
        if os.path.exists(path):
            val_data[sl] = np.load(path)
            print(f"  Loaded val_len_{sl}: {val_data[sl].shape}")
    
    if not val_data:
        print("ERROR: No validation data found. Run `python kaggle_multi_gpu/data.py` first.")
        sys.exit(1)
    
    available_seq_lens = sorted(val_data.keys())
    batch_sizes = [1, 5, 20, 64]
    
    # Load models
    print("\n[2/5] Loading models...")
    device_bibo = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_qwen = 'cuda:1' if torch.cuda.device_count() > 1 else device_bibo
    
    bibo_cfg = {k: v for k, v in CFG['bibo'].items() if k != 'device'}
    bibo_model = BiBoForCausalLM(BiBoConfig(**bibo_cfg))
    
    bibo_ckpt = os.path.join(BASE_DIR, 'checkpoints', 'bibo.pt')
    if os.path.exists(bibo_ckpt):
        bibo_model.load_state_dict(torch.load(bibo_ckpt, map_location=device_bibo))
        print(f"  BiBo: loaded checkpoint → {device_bibo}")
    else:
        print(f"  BiBo: NO CHECKPOINT (using random init) → {device_bibo}")
    bibo_model = bibo_model.to(device_bibo)
    
    qwen_cfg = {k: v for k, v in CFG['qwen3moe'].items() if k != 'device'}
    qwen_model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen_cfg))
    
    qwen_ckpt = os.path.join(BASE_DIR, 'checkpoints', 'qwen3moe.pt')
    if os.path.exists(qwen_ckpt):
        qwen_model.load_state_dict(torch.load(qwen_ckpt, map_location=device_qwen))
        print(f"  Qwen3MoE: loaded checkpoint → {device_qwen}")
    else:
        print(f"  Qwen3MoE: NO CHECKPOINT (using random init) → {device_qwen}")
    qwen_model = qwen_model.to(device_qwen)
    
    # ============================================================
    # Phase 3: Multi-batch, multi-seq extraction
    # ============================================================
    print("\n[3/5] Extracting routing data (multi-batch × multi-seq)...")
    
    bibo_all_data = {}   # (seq_len, batch_size) → layer_data
    qwen_all_data = {}
    all_metrics = {}     # (model, seq_len, batch_size) → metrics dict
    
    for sl in available_seq_lens:
        for bs in batch_sizes:
            if bs > len(val_data[sl]):
                continue
            
            batch = torch.tensor(val_data[sl][:bs, :-1], dtype=torch.long)
            
            print(f"  seq={sl}, batch={bs}...")
            bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
            qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')
            
            bibo_all_data[(sl, bs)] = bibo_ld
            qwen_all_data[(sl, bs)] = qwen_ld
            
            # Compute metrics
            bibo_indices_flat = np.concatenate([
                ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in bibo_ld.values()
            ])
            qwen_indices_flat = np.concatenate([
                ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in qwen_ld.values()
            ])
            
            all_metrics[('BiBo', sl, bs)] = compute_load_balance_metrics(bibo_indices_flat, n_exp_bibo, top_k)
            all_metrics[('Qwen3MoE', sl, bs)] = compute_load_balance_metrics(qwen_indices_flat, n_exp_qwen, top_k)
    
    # ============================================================
    # Phase 4: Generate all plots
    # ============================================================
    print("\n[4/5] Generating plots...")
    
    # --- Plot 1: Usage sweep ---
    print("\n  --- Expert Usage Sweep ---")
    plot_expert_usage_sweep(bibo_all_data, 'BiBo', n_exp_bibo, batch_sizes, available_seq_lens)
    plot_expert_usage_sweep(qwen_all_data, 'Qwen3MoE', n_exp_qwen, batch_sizes, available_seq_lens)
    
    # --- Plot 2: Comparative usage (largest batch, each seq_len) ---
    print("\n  --- Comparative Usage ---")
    for sl in available_seq_lens:
        # Find largest available batch
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_comparative_usage(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)],
            n_exp_bibo, n_exp_qwen, sl, max_bs
        )
    
    # --- Plot 3: Confidence distribution ---
    print("\n  --- Confidence Distribution ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_confidence_distribution(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)], sl
        )
    
    # --- Plot 4: Co-selection ---
    print("\n  --- Expert Co-Selection ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_coselection_matrix(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)],
            n_exp_bibo, n_exp_qwen, sl
        )
    
    # --- Plot 5: Position-type routing ---
    print("\n  --- Position-Type Routing ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_position_type_routing(bibo_all_data[(sl, max_bs)], 'BiBo', n_exp_bibo, sl)
        plot_position_type_routing(qwen_all_data[(sl, max_bs)], 'Qwen3MoE', n_exp_qwen, sl)
    
    # --- Plot 6: Specialization radar ---
    print("\n  --- Specialization Radar ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_specialization_radar(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)],
            n_exp_bibo, n_exp_qwen, sl
        )
    
    # --- Plot 7: Token-expert heatmap v2 ---
    print("\n  --- Token-Expert Heatmaps ---")
    for sl in available_seq_lens:
        # Use single sample for heatmap clarity
        if (sl, 1) in bibo_all_data:
            plot_token_expert_heatmap_v2(bibo_all_data[(sl, 1)], 'BiBo', n_exp_bibo, sl)
            plot_token_expert_heatmap_v2(qwen_all_data[(sl, 1)], 'Qwen3MoE', n_exp_qwen, sl)
    
    # --- Plot 8: Confidence evolution ---
    print("\n  --- Confidence Evolution ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_confidence_evolution_comparative(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)], sl
        )
    
    # --- Plot 9: Entropy evolution ---
    print("\n  --- Entropy Evolution ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_entropy_evolution_comparative(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)], sl
        )
    
    # --- Plot 10: Load balance summary ---
    print("\n  --- Load Balance Summary ---")
    plot_load_balance_summary(all_metrics)
    
    # --- Plot 11: Weight rank distribution ---
    print("\n  --- Weight Rank Distribution ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_weight_rank_distribution(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)], sl
        )
    
    # --- Plot 12: Routing diversity ---
    print("\n  --- Routing Diversity ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_routing_diversity(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)],
            n_exp_bibo, n_exp_qwen, sl, max_bs
        )
    
    # --- Plot 13: Routing stability ---
    print("\n  --- Routing Stability ---")
    stability_scores = {}
    for sl in available_seq_lens:
        if len(val_data[sl]) >= 20:
            s_bibo = plot_routing_stability(
                bibo_model, device_bibo, 'bibo', 'BiBo', n_exp_bibo, val_data[sl], sl
            )
            stability_scores[f'BiBo_seq{sl}'] = s_bibo
            
            s_qwen = plot_routing_stability(
                qwen_model, device_qwen, 'qwen', 'Qwen3MoE', n_exp_qwen, val_data[sl], sl
            )
            stability_scores[f'Qwen3MoE_seq{sl}'] = s_qwen
    
    # --- Plot 15: Per-layer weight KDE ---
    print("\n  --- Per-Layer Weight KDE ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_per_layer_weight_kde(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)], sl
        )
    
    # --- Plot 16: Expert switching rate ---
    print("\n  --- Expert Switching Rate ---")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        plot_expert_switching_rate(
            bibo_all_data[(sl, max_bs)], qwen_all_data[(sl, max_bs)],
            n_exp_bibo, n_exp_qwen, sl
        )
    
    # --- Plot 17: BiBo special expert analysis ---
    print("\n  --- BiBo Special Expert Analysis ---")
    bibo_by_seq = {}
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        bibo_by_seq[sl] = bibo_all_data[(sl, max_bs)]
    plot_special_expert_analysis(bibo_by_seq, n_exp_bibo, available_seq_lens)
    
    # --- Plot 14: Grand summary dashboard ---
    print("\n  --- Grand Summary Dashboard ---")
    plot_grand_summary(all_metrics, stability_scores)
    
    # ============================================================
    # Phase 5: Save metrics JSON
    # ============================================================
    print("\n[5/5] Saving metrics...")
    
    # Convert metrics for JSON serialization
    json_metrics = {}
    for key, metrics in all_metrics.items():
        model, sl, bs = key
        json_key = f"{model}_seq{sl}_bs{bs}"
        json_metrics[json_key] = metrics
    
    json_metrics['stability_scores'] = stability_scores
    json_metrics['config'] = {
        'bibo_experts': n_exp_bibo,
        'qwen_experts': n_exp_qwen,
        'top_k': top_k,
        'seq_lens_analyzed': available_seq_lens,
        'batch_sizes_analyzed': batch_sizes,
    }
    
    with open(METRICS_OUT, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"  Saved: {METRICS_OUT}")
    
    # Cleanup
    del bibo_model, qwen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Plots: {PLOTS_DIR}/ ({len(os.listdir(PLOTS_DIR))} files)")
    print(f"  Metrics: {METRICS_OUT}")
    print()
    
    # Print key findings
    print("  KEY FINDINGS:")
    for sl in available_seq_lens:
        max_bs = max(bs for bs in batch_sizes if (sl, bs) in bibo_all_data)
        bm = all_metrics.get(('BiBo', sl, max_bs), {})
        qm = all_metrics.get(('Qwen3MoE', sl, max_bs), {})
        if bm and qm:
            print(f"    seq={sl}, bs={max_bs}:")
            print(f"      BiBo  — Gini={bm.get('gini',0):.3f} H_norm={bm.get('normalized_entropy',0):.3f} CV={bm.get('cv',0):.3f}")
            print(f"      Qwen  — Gini={qm.get('gini',0):.3f} H_norm={qm.get('normalized_entropy',0):.3f} CV={qm.get('cv',0):.3f}")
    
    if stability_scores:
        print(f"\n    Stability (avg Jaccard):")
        for k, v in stability_scores.items():
            print(f"      {k}: {v:.3f}")


if __name__ == '__main__':
    main()
