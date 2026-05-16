"""
Detailed router analysis — per-token, per-layer expert selection + confidence evolution.

Run after training:
    python kaggle_multi_gpu/analyze_router.py

Produces:
1. Per-token expert selection heatmap (which expert at each position)
2. Confidence (top-k weight) evolution over sequence positions
3. Expert usage histogram across a batch
4. Per-layer entropy evolution over token positions
"""
import sys, torch, numpy as np, json, os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

sns.set_theme(style='whitegrid')


# ============================================================
# Hook-based extraction — captures per-token routing decisions
# ============================================================

def extract_bibo_routing(model, input_ids, device):
    """
    Returns per-layer:
        indices: [seq_len, top_k] — which experts selected per token
        weights: [seq_len, top_k] — routing weights per token
    """
    model.eval()
    layer_data = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            indices, weights = output  # [1, seq_len, top_k]
            layer_data[layer_idx] = {
                'indices': indices[0].detach().cpu().numpy(),  # [seq_len, top_k]
                'weights': weights[0].detach().cpu().numpy(),  # [seq_len, top_k]
            }
        return hook_fn
    
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(input_ids=input_ids.unsqueeze(0).to(device))
    
    for h in hooks:
        h.remove()
    return layer_data


def extract_qwen_routing(model, input_ids, device):
    """Same but for Qwen3MoE router (returns logits, scores, indices)."""
    model.eval()
    layer_data = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            logits, scores, indices = output  # [seq_len, n_exp], [seq_len, top_k], [seq_len, top_k]
            layer_data[layer_idx] = {
                'indices': indices.detach().cpu().numpy(),  # [seq_len, top_k]
                'weights': scores.detach().cpu().numpy(),   # [seq_len, top_k]
            }
        return hook_fn
    
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(input_ids=input_ids.unsqueeze(0).to(device))
    
    for h in hooks:
        h.remove()
    return layer_data


def extract_batch_routing(model, batch_input_ids, device, model_type='bibo'):
    """Extract routing for a full batch — returns aggregated expert counts per layer."""
    model.eval()
    layer_counts = {}
    layer_weights = {}
    hooks = []
    n_experts = CFG['bibo']['num_routed_experts'] if model_type == 'bibo' else CFG['qwen3moe']['num_experts']
    
    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            if model_type == 'bibo':
                indices, weights = output  # [bs, seq, top_k]
                idx_flat = indices.flatten()
                w_flat = weights.flatten()
            else:
                _, weights, indices = output  # logits, scores, indices [bs*seq, top_k]
                idx_flat = indices.flatten()
                w_flat = weights.flatten()
            
            if layer_idx not in layer_counts:
                layer_counts[layer_idx] = torch.zeros(n_experts, device=device)
                layer_weights[layer_idx] = []
            layer_counts[layer_idx] += torch.bincount(idx_flat, minlength=n_experts).float()
            layer_weights[layer_idx].append(w_flat.detach().cpu())
        return hook_fn
    
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(input_ids=batch_input_ids.to(device))
    
    for h in hooks:
        h.remove()
    
    return layer_counts, layer_weights


# ============================================================
# Plotting
# ============================================================

def plot_token_expert_heatmap(layer_data, model_name, seq_len_label):
    """Heatmap: x=token position, y=layer, color=primary expert selected."""
    layers = sorted(layer_data.keys())
    n_layers = len(layers)
    seq_len = layer_data[layers[0]]['indices'].shape[0]
    
    # Primary expert (top-1) per token per layer
    heatmap = np.zeros((n_layers, seq_len))
    for i, l in enumerate(layers):
        heatmap[i] = layer_data[l]['indices'][:, 0]  # top-1 expert
    
    fig, ax = plt.subplots(figsize=(min(20, seq_len//4), max(3, n_layers*0.6)))
    sns.heatmap(heatmap, cmap='tab10', ax=ax, cbar_kws={'label': 'Expert ID'},
                xticklabels=max(1, seq_len//10), yticklabels=[f'L{l}' for l in layers])
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Layer')
    ax.set_title(f'{model_name} — Top-1 Expert Selection per Token (seq={seq_len_label})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'expert_heatmap_{model_name}_{seq_len_label}.png'), dpi=150)
    plt.close()
    print(f"  Saved: expert_heatmap_{model_name}_{seq_len_label}.png")


def plot_confidence_evolution(layer_data, model_name, seq_len_label):
    """Line plot: routing confidence (max weight) over token positions, per layer."""
    layers = sorted(layer_data.keys())
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for l in layers:
        max_weights = layer_data[l]['weights'].max(axis=1)  # [seq_len]
        ax.plot(max_weights, label=f'L{l}', alpha=0.7, linewidth=1.2)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Top-1 Routing Weight (confidence)')
    ax.set_title(f'{model_name} — Router Confidence Evolution (seq={seq_len_label})')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'confidence_evolution_{model_name}_{seq_len_label}.png'), dpi=150)
    plt.close()
    print(f"  Saved: confidence_evolution_{model_name}_{seq_len_label}.png")


def plot_entropy_evolution(layer_data, model_name, seq_len_label, n_experts):
    """Per-token entropy of routing weights over sequence positions."""
    layers = sorted(layer_data.keys())
    
    fig, ax = plt.subplots(figsize=(12, 5))
    max_ent = np.log(n_experts)
    
    for l in layers:
        weights = layer_data[l]['weights']  # [seq_len, top_k]
        # Entropy of top-k weights (normalized)
        w_norm = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
        entropy = -(w_norm * np.log(w_norm + 1e-10)).sum(axis=1)
        ax.plot(entropy, label=f'L{l}', alpha=0.7, linewidth=1.2)
    
    ax.axhline(y=np.log(weights.shape[1]), color='red', linestyle='--', alpha=0.5, label='Max (uniform top-k)')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Routing Entropy')
    ax.set_title(f'{model_name} — Router Entropy per Token (seq={seq_len_label})')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'entropy_evolution_{model_name}_{seq_len_label}.png'), dpi=150)
    plt.close()
    print(f"  Saved: entropy_evolution_{model_name}_{seq_len_label}.png")


def plot_batch_expert_usage(layer_counts, model_name, n_experts):
    """Bar chart: expert usage across a batch, per layer."""
    layers = sorted(layer_counts.keys())
    
    fig, axes = plt.subplots(1, len(layers), figsize=(4*len(layers), 4), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    
    for ax, l in zip(axes, layers):
        counts = layer_counts[l].cpu().numpy()
        dist = counts / counts.sum()
        colors = sns.color_palette('husl', n_experts)
        ax.bar(range(n_experts), dist, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Expert')
        ax.set_title(f'L{l}')
        ax.set_xticks(range(n_experts))
        entropy = -(dist * np.log(dist + 1e-10)).sum()
        ax.text(0.5, 0.95, f'H={entropy:.2f}', transform=ax.transAxes, ha='center', fontsize=9)
    
    axes[0].set_ylabel('Fraction of tokens')
    plt.suptitle(f'{model_name} — Expert Usage (batch)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'expert_usage_batch_{model_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: expert_usage_batch_{model_name}.png")


# ============================================================
# Main
# ============================================================

def main():
    # Load val data
    val_64 = np.load(os.path.join(BASE_DIR, 'data', 'val_len_64.npy'))
    val_128 = np.load(os.path.join(BASE_DIR, 'data', 'val_len_128.npy'))
    
    # Single sequence for per-token analysis
    single_64 = torch.tensor(val_64[0, :-1], dtype=torch.long)   # 128 tokens (64 unsorted + sep + 64 sorted)
    single_128 = torch.tensor(val_128[0, :-1], dtype=torch.long) # 256 tokens
    
    # Batch for aggregate stats
    batch_128 = torch.tensor(val_128[:64, :-1], dtype=torch.long)  # [64, 256]
    
    # === BiBo ===
    print("\n=== Loading BiBo ===")
    bibo_cfg = {k:v for k,v in CFG['bibo'].items() if k != 'device'}
    bibo = BiBoForCausalLM(BiBoConfig(**bibo_cfg)).cuda(0)
    bibo.load_state_dict(torch.load(os.path.join(BASE_DIR, 'checkpoints', 'bibo.pt'), map_location='cuda:0'))
    
    print("  Analyzing single seq_len=64...")
    bibo_64 = extract_bibo_routing(bibo, single_64, 'cuda:0')
    plot_token_expert_heatmap(bibo_64, 'BiBo', '64')
    plot_confidence_evolution(bibo_64, 'BiBo', '64')
    plot_entropy_evolution(bibo_64, 'BiBo', '64', CFG['bibo']['num_routed_experts'])
    
    print("  Analyzing single seq_len=128...")
    bibo_128 = extract_bibo_routing(bibo, single_128, 'cuda:0')
    plot_token_expert_heatmap(bibo_128, 'BiBo', '128')
    plot_confidence_evolution(bibo_128, 'BiBo', '128')
    plot_entropy_evolution(bibo_128, 'BiBo', '128', CFG['bibo']['num_routed_experts'])
    
    print("  Analyzing batch (64 samples, seq=128)...")
    bibo_counts, bibo_weights = extract_batch_routing(bibo, batch_128, 'cuda:0', 'bibo')
    plot_batch_expert_usage(bibo_counts, 'BiBo', CFG['bibo']['num_routed_experts'])
    
    del bibo; torch.cuda.empty_cache()
    
    # === Qwen3MoE ===
    print("\n=== Loading Qwen3MoE ===")
    qwen_cfg = {k:v for k,v in CFG['qwen3moe'].items() if k != 'device'}
    qwen = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen_cfg)).cuda(1)
    qwen.load_state_dict(torch.load(os.path.join(BASE_DIR, 'checkpoints', 'qwen3moe.pt'), map_location='cuda:1'))
    
    print("  Analyzing single seq_len=64...")
    qwen_64 = extract_qwen_routing(qwen, single_64.clone(), 'cuda:1')
    plot_token_expert_heatmap(qwen_64, 'Qwen3MoE', '64')
    plot_confidence_evolution(qwen_64, 'Qwen3MoE', '64')
    plot_entropy_evolution(qwen_64, 'Qwen3MoE', '64', CFG['qwen3moe']['num_experts'])
    
    print("  Analyzing single seq_len=128...")
    qwen_128 = extract_qwen_routing(qwen, single_128.clone(), 'cuda:1')
    plot_token_expert_heatmap(qwen_128, 'Qwen3MoE', '128')
    plot_confidence_evolution(qwen_128, 'Qwen3MoE', '128')
    plot_entropy_evolution(qwen_128, 'Qwen3MoE', '128', CFG['qwen3moe']['num_experts'])
    
    print("  Analyzing batch (64 samples, seq=128)...")
    qwen_counts, qwen_weights = extract_batch_routing(qwen, batch_128.clone(), 'cuda:1', 'qwen')
    plot_batch_expert_usage(qwen_counts, 'Qwen3MoE', CFG['qwen3moe']['num_experts'])
    
    del qwen; torch.cuda.empty_cache()
    
    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
