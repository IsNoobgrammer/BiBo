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
- PolyGLU expert-type analysis (SiLU vs ReLU² vs Tanh vs Identity vs Zero)
- Side-by-side comparative visualizations

Usage:
    python misc/kaggle/multi_gpu/analyze_router.py

Produces publication-quality plots in misc/kaggle/multi_gpu/plots/
"""
import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_utils import setup_style, PLOTS_DIR, METRICS_DIR
from model_utils import CFG, load_bibo, load_qwen, extract_routing_data
from router_metrics import compute_load_balance_metrics
from router_plots import (
    plot_expert_usage_sweep,
    plot_comparative_usage,
    plot_confidence_distribution,
    plot_coselection_matrix,
    plot_position_type_routing,
    plot_specialization_radar,
    plot_token_expert_heatmap_v2,
    plot_confidence_evolution_comparative,
    plot_entropy_evolution_comparative,
    plot_load_balance_summary,
    plot_weight_rank_distribution,
    plot_routing_diversity,
    plot_routing_stability,
    plot_expert_type_analysis,
    plot_special_expert_analysis,
    plot_expert_switching_rate,
    plot_grand_summary,
    plot_per_layer_weight_kde,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_OUT = os.path.join(METRICS_DIR, 'router_analysis.json')

setup_style()


def main():
    print("=" * 70)
    print("  COMPREHENSIVE ROUTER ANALYSIS — BiBo (PolyGLU) vs Qwen3MoE")
    print("=" * 70)

    task = CFG['training'].get('task', 'sort')

    bibo_cfg = CFG['bibo']
    poly_mult = bibo_cfg.get('polyglu_expert_multiplier', 2)
    special_pairs = bibo_cfg.get('special_expert_pairs', 1)
    n_exp_bibo = poly_mult * 3 + special_pairs * 2
    n_exp_qwen = CFG['qwen3moe']['num_experts']
    top_k = bibo_cfg['num_experts_per_tok']

    print(f"\n  Task: {task}")
    print(f"  BiBo experts: {poly_mult}×[SiLU,ReLU²,Tanh] + {special_pairs}×[Identity,Zero] = {n_exp_bibo}")
    print(f"  Qwen experts: {n_exp_qwen} homogeneous MLPs")
    print(f"  Top-K: {top_k}")

    # Load validation data (task-aware)
    print("\n[1/6] Loading data...")
    val_data = {}

    if task == 'arithmetic':
        arith_cfg = CFG['training'].get('arithmetic', {})
        buckets = arith_cfg.get('buckets', [[3, 7], [9, 16], [19, 30], [35, 50]])
        for min_t, max_t in buckets:
            bucket_name = f'arith_{min_t}_{max_t}'
            path = os.path.join(BASE_DIR, 'data', f'val_{bucket_name}.npy')
            if os.path.exists(path):
                data = np.load(path)
                sl_key = (min_t + max_t) // 2
                val_data[sl_key] = data
                print(f"  Loaded val_{bucket_name}: {data.shape} (key={sl_key})")
    else:
        for sl in [64, 128, 256]:
            path = os.path.join(BASE_DIR, 'data', f'val_len_{sl}.npy')
            if os.path.exists(path):
                val_data[sl] = np.load(path)
                print(f"  Loaded val_len_{sl}: {val_data[sl].shape}")

    if not val_data:
        print("ERROR: No validation data found. Run data generation script first.")
        sys.exit(1)

    available_seq_lens = sorted(val_data.keys())
    batch_sizes = [1, 5, 20, 64]

    # Load models
    print("\n[2/6] Loading models...")
    device_bibo = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_qwen = 'cuda:1' if torch.cuda.device_count() > 1 else device_bibo

    bibo_model = load_bibo(device_bibo, bibo_cfg)
    qwen_model = load_qwen(device_qwen)

    # Extract routing data
    print("\n[3/6] Extracting routing data...")
    bibo_all_data = {}
    qwen_all_data = {}
    bibo_by_seq = {}

    for sl in available_seq_lens:
        for bs in batch_sizes:
            if bs > len(val_data[sl]):
                continue
            batch = torch.tensor(val_data[sl][:bs, :-1], dtype=torch.long)

            bibo_ld = extract_routing_data(bibo_model, batch, device_bibo, 'bibo')
            qwen_ld = extract_routing_data(qwen_model, batch, device_qwen, 'qwen')

            bibo_all_data[(sl, bs)] = bibo_ld
            qwen_all_data[(sl, bs)] = qwen_ld

            if sl not in bibo_by_seq or bs > list(bibo_by_seq[sl].values())[0]['indices'].shape[0]:
                bibo_by_seq[sl] = bibo_ld

            print(f"  seq={sl}, batch={bs} ✓")

    # Compute metrics
    print("\n[4/6] Computing metrics...")
    all_metrics = {}
    for (sl, bs), data in bibo_all_data.items():
        indices = np.concatenate([
            ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in data.values()
        ])
        all_metrics[('BiBo', sl, bs)] = compute_load_balance_metrics(indices, n_exp_bibo, top_k)

    for (sl, bs), data in qwen_all_data.items():
        indices = np.concatenate([
            ld['indices'].numpy().reshape(-1, ld['indices'].shape[-1]) for ld in data.values()
        ])
        all_metrics[('Qwen3MoE', sl, bs)] = compute_load_balance_metrics(indices, n_exp_qwen, top_k)

    # Generate plots
    print("\n[5/6] Generating plots...")

    plot_expert_usage_sweep(bibo_all_data, 'BiBo', n_exp_bibo, batch_sizes, available_seq_lens, bibo_cfg)
    plot_expert_usage_sweep(qwen_all_data, 'Qwen3MoE', n_exp_qwen, batch_sizes, available_seq_lens)

    primary_bs = 64
    for sl in available_seq_lens:
        key = (sl, primary_bs)
        if key not in bibo_all_data:
            key = (sl, batch_sizes[-1])
        if key not in bibo_all_data:
            continue

        bibo_ld = bibo_all_data[key]
        qwen_ld = qwen_all_data[key]

        plot_comparative_usage(bibo_ld, qwen_ld, n_exp_bibo, n_exp_qwen, sl, key[1], bibo_cfg)
        plot_confidence_distribution(bibo_ld, qwen_ld, sl)
        plot_coselection_matrix(bibo_ld, qwen_ld, n_exp_bibo, n_exp_qwen, sl, bibo_cfg)
        plot_position_type_routing(bibo_ld, 'BiBo', n_exp_bibo, sl, bibo_cfg)
        plot_position_type_routing(qwen_ld, 'Qwen3MoE', n_exp_qwen, sl)
        plot_specialization_radar(bibo_ld, qwen_ld, n_exp_bibo, n_exp_qwen, sl, bibo_cfg)
        plot_token_expert_heatmap_v2(bibo_ld, 'BiBo', n_exp_bibo, sl, bibo_cfg)
        plot_token_expert_heatmap_v2(qwen_ld, 'Qwen3MoE', n_exp_qwen, sl)
        plot_confidence_evolution_comparative(bibo_ld, qwen_ld, sl)
        plot_entropy_evolution_comparative(bibo_ld, qwen_ld, sl)
        plot_weight_rank_distribution(bibo_ld, qwen_ld, sl)
        plot_routing_diversity(bibo_ld, qwen_ld, n_exp_bibo, n_exp_qwen, sl, key[1])
        plot_expert_switching_rate(bibo_ld, qwen_ld, n_exp_bibo, n_exp_qwen, sl)
        plot_per_layer_weight_kde(bibo_ld, qwen_ld, sl)

    # PolyGLU expert-type deep dive (BiBo-specific)
    print("\n  --- PolyGLU Expert-Type Analysis ---")
    plot_expert_type_analysis(bibo_by_seq, bibo_cfg, available_seq_lens)
    plot_special_expert_analysis(bibo_by_seq, bibo_cfg, available_seq_lens)

    plot_load_balance_summary(all_metrics)

    # Stability analysis
    print("\n  --- Routing Stability ---")
    stability_scores = {}
    for sl in available_seq_lens[:2]:
        if sl in val_data:
            s = plot_routing_stability(bibo_model, device_bibo, 'bibo', 'BiBo', n_exp_bibo, val_data[sl], sl)
            stability_scores[f'BiBo_seq{sl}'] = s
            s = plot_routing_stability(qwen_model, device_qwen, 'qwen', 'Qwen3MoE', n_exp_qwen, val_data[sl], sl)
            stability_scores[f'Qwen3MoE_seq{sl}'] = s

    plot_grand_summary(all_metrics, stability_scores, bibo_cfg)

    # Save metrics
    print("\n[6/6] Saving metrics...")
    metrics_out = {
        'config': {
            'bibo': {
                'polyglu_expert_multiplier': poly_mult,
                'special_expert_pairs': special_pairs,
                'num_routed_experts': n_exp_bibo,
                'num_experts_per_tok': top_k,
                'router_type': bibo_cfg.get('router_type', 'conv'),
                'router_lambda': bibo_cfg.get('router_lambda', 1.0),
            },
            'qwen3moe': {
                'num_experts': n_exp_qwen,
                'num_experts_per_tok': CFG['qwen3moe']['num_experts_per_tok'],
            }
        },
        'load_balance': {f'{m}_seq{sl}_bs{bs}': v
                         for (m, sl, bs), v in all_metrics.items()},
        'stability': stability_scores,
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"  Saved → {METRICS_OUT}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Metrics: {METRICS_OUT}")
    print("=" * 70)


if __name__ == '__main__':
    main()
