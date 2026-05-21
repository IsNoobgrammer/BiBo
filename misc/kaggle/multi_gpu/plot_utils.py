"""
Shared plotting utilities for Kaggle ablation scripts.

Consolidates aesthetic config, color palettes, and helper functions
used across analyze_router.py, analyze_model_sorting.py, etc.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    'BASE_DIR', 'PLOTS_DIR', 'METRICS_DIR',
    'BIBO_COLOR', 'QWEN_COLOR', 'BIBO_CMAP', 'QWEN_CMAP', 'ACCENT_COLORS',
    'POLYGLU_COLORS', 'SPECIAL_COLORS',
    'setup_style', 'save_figure', 'get_expert_layout',
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────
BIBO_COLOR = '#2196F3'       # blue
QWEN_COLOR = '#FF5722'       # deep orange
BIBO_CMAP = 'Blues'
QWEN_CMAP = 'Oranges'
ACCENT_COLORS = ['#4CAF50', '#9C27B0', '#FF9800', '#00BCD4', '#E91E63', '#8BC34A']

# PolyGLU activation colors
POLYGLU_COLORS = {
    'silu': '#1976D2',    # blue
    'relu2': '#D32F2F',   # red
    'tanh': '#7B1FA2',    # purple
}
SPECIAL_COLORS = {
    'identity': '#4CAF50',  # green
    'zero': '#9E9E9E',      # gray
}


def setup_style():
    """Apply consistent matplotlib style for all plots."""
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


def save_figure(fig, name, close=True):
    """Save figure to plots/ directory with consistent settings.

    Args:
        fig: matplotlib figure
        name: filename (without extension)
        close: close figure after saving (default True)
    """
    path = os.path.join(PLOTS_DIR, f'{name}.png')
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"  Saved: {path}")
    return path


def get_expert_layout(cfg):
    """Parse BiBo config to get expert layout info.

    Args:
        cfg: dict from config.yaml (bibo section)

    Returns:
        list of (label, type, color) tuples for each expert
    """
    poly_mult = cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = cfg.get('special_expert_pairs', 1)
    activations = ['silu', 'relu2', 'tanh']

    layout = []
    for g in range(poly_mult):
        for act in activations:
            label = f"{act.upper()}_{g}" if poly_mult > 1 else act.upper()
            layout.append((label, act, POLYGLU_COLORS[act]))

    for p in range(special_pairs):
        layout.append((f"ID_{p}", 'identity', SPECIAL_COLORS['identity']))
        layout.append((f"ZERO_{p}", 'zero', SPECIAL_COLORS['zero']))

    return layout


def get_expert_colors(n_experts, model_name, cfg=None):
    """Get color list for experts.

    Args:
        n_experts: number of experts
        model_name: 'bibo' or 'qwen3moe'
        cfg: BiBo config dict (required for BiBo, ignored for Qwen)

    Returns:
        list of color strings
    """
    if model_name == 'bibo' and cfg is not None:
        layout = get_expert_layout(cfg)
        return [c for _, _, c in layout[:n_experts]]
    # Fallback: use accent color cycle
    return [ACCENT_COLORS[i % len(ACCENT_COLORS)] for i in range(n_experts)]


def get_expert_labels(n_experts, model_name, cfg=None):
    """Get label list for experts.

    Args:
        n_experts: number of experts
        model_name: 'bibo' or 'qwen3moe'
        cfg: BiBo config dict (required for BiBo, ignored for Qwen)

    Returns:
        list of label strings
    """
    if model_name == 'bibo' and cfg is not None:
        layout = get_expert_layout(cfg)
        return [l for l, _, _ in layout[:n_experts]]
    return [f'E{i}' for i in range(n_experts)]


def get_expert_types(cfg):
    """Get list of expert type strings from config.

    Returns list like ['silu', 'relu2', 'tanh', 'silu', 'relu2', 'tanh', 'identity', 'zero']
    """
    layout = get_expert_layout(cfg)
    return [t for _, t, _ in layout]
