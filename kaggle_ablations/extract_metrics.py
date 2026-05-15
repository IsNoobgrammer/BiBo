"""
Extract post-training metrics from saved models.
Run AFTER train.py completes.

Extracts:
- Router expert selection distribution per layer
- Router confidence (inverse entropy)
- Forward pass timing
- Expert specialization patterns

Usage: python kaggle_ablations/extract_metrics.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import yaml
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from torch.utils.data import DataLoader
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
from train import HardSequenceDataset

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']
DEVICE = torch.device(T['device'])
VOCAB_SIZE = T['vocab_size']
SEQ_LEN = T['seq_len']
BATCH_SIZE = T['batch_size']
VAL_SAMPLES = T['val_samples']

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')


def make_bibo_model():
    return BiBoForCausalLM(BiBoConfig(**CFG['bibo'])).to(DEVICE)

def make_qwen_model():
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(**CFG['qwen3moe'])).to(DEVICE)


def load_model(model_class, config_cls, config_dict, name):
    model = model_class(config_cls(**config_dict)).to(DEVICE)
    state_path = os.path.join(SAVE_DIR, f'{name}.pt')
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded {name} from {state_path}")
    return model


# ============================================================
# Router Analysis (BiBo)
# ============================================================

def extract_bibo_router(model, val_loader, num_batches=30):
    """Extract per-layer router stats."""
    expert_counts = {}
    expert_weights_all = {}
    router_logits_all = {}
    hooks = []
    
    def make_gate_hook(layer_idx):
        def hook_fn(module, input, output):
            indices, weights = output  # [b, s, k], [b, s, k]
            key = f'layer_{layer_idx}'
            if key not in expert_counts:
                n_experts = model.config.num_routed_experts
                expert_counts[key] = torch.zeros(n_experts, device=DEVICE)
                expert_weights_all[key] = []
            for k in range(indices.shape[-1]):
                flat = indices[:, :, k].flatten()
                expert_counts[key] += torch.bincount(flat, minlength=model.config.num_routed_experts).float()
            expert_weights_all[key].append(weights.detach().cpu())
        return hook_fn
    
    # Hook router outputs
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_gate_hook(layer_idx)))
    
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            model(input_ids=input_ids.to(DEVICE))
    
    for h in hooks:
        h.remove()
    
    # Compute stats
    router_stats = {}
    for key in sorted(expert_counts.keys()):
        counts = expert_counts[key].cpu().numpy()
        dist = counts / counts.sum()
        weights = torch.cat(expert_weights_all[key]).flatten().numpy()
        
        entropy = -(dist * np.log(dist + 1e-10)).sum()
        max_entropy = np.log(len(dist))
        
        router_stats[key] = {
            'expert_distribution': dist.tolist(),
            'expert_counts': counts.tolist(),
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'normalized_entropy': float(entropy / max_entropy),
            'inverse_entropy': float(1.0 - entropy / max_entropy),  # confidence
            'avg_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'max_weight': float(np.max(weights)),
            'min_weight': float(np.min(weights)),
            'load_balance_std': float(dist.std()),
            'most_used_expert': int(counts.argmax()),
            'least_used_expert': int(counts.argmin()),
            'usage_ratio': float(counts.max() / (counts.min() + 1e-10)),
        }
    
    return router_stats


# ============================================================
# Qwen3MoE Router Analysis
# ============================================================

def extract_qwen_router(model, val_loader, num_batches=30):
    """Extract Qwen3MoE router stats."""
    expert_counts = {}
    expert_weights_all = {}
    hooks = []
    
    def make_gate_hook(layer_idx):
        def hook_fn(module, input, output):
            _, weights, indices = output  # logits, scores, indices
            key = f'layer_{layer_idx}'
            n_experts = model.config.num_experts
            if key not in expert_counts:
                expert_counts[key] = torch.zeros(n_experts, device=DEVICE)
                expert_weights_all[key] = []
            for k in range(indices.shape[-1]):
                flat = indices[:, k].flatten()
                expert_counts[key] += torch.bincount(flat, minlength=n_experts).float()
            expert_weights_all[key].append(weights.detach().cpu())
        return hook_fn
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(make_gate_hook(layer_idx)))
    
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            model(input_ids=input_ids.to(DEVICE))
    
    for h in hooks:
        h.remove()
    
    router_stats = {}
    for key in sorted(expert_counts.keys()):
        counts = expert_counts[key].cpu().numpy()
        dist = counts / counts.sum()
        weights = torch.cat(expert_weights_all[key]).flatten().numpy()
        
        entropy = -(dist * np.log(dist + 1e-10)).sum()
        max_entropy = np.log(len(dist))
        
        router_stats[key] = {
            'expert_distribution': dist.tolist(),
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'normalized_entropy': float(entropy / max_entropy),
            'inverse_entropy': float(1.0 - entropy / max_entropy),
            'avg_weight': float(np.mean(weights)),
            'std_weight': float(np.std(weights)),
            'load_balance_std': float(dist.std()),
            'most_used_expert': int(counts.argmax()),
            'least_used_expert': int(counts.argmin()),
            'usage_ratio': float(counts.max() / (counts.min() + 1e-10)),
        }
    
    return router_stats


# ============================================================
# Forward Pass Benchmark
# ============================================================

def benchmark_forward(model, name, warmup=15, runs=100):
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN - 1), device=DEVICE)
    
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=x)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    
    avg = np.mean(times)
    std = np.std(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    tps = (BATCH_SIZE * (SEQ_LEN - 1)) / (avg / 1000)
    
    print(f"  {name}: avg={avg:.2f}ms ± {std:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  | {tps:,.0f} tok/s")
    return {'avg_ms': avg, 'std_ms': std, 'p50_ms': p50, 'p95_ms': p95, 'tok_per_sec': tps}


# ============================================================
# Main
# ============================================================

def main():
    val_ds = HardSequenceDataset(VAL_SAMPLES, SEQ_LEN, VOCAB_SIZE, seed=123)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    results = {}
    
    # BiBo
    print("\n=== BiBo Router Analysis ===")
    bibo = load_model(BiBoForCausalLM, BiBoConfig, CFG['bibo'], 'bibo')
    bibo_router = extract_bibo_router(bibo, val_loader)
    for key, stats in sorted(bibo_router.items()):
        print(f"  {key}: entropy={stats['normalized_entropy']:.3f} confidence={stats['inverse_entropy']:.3f} balance_std={stats['load_balance_std']:.4f}")
        print(f"    dist={[f'{d:.3f}' for d in stats['expert_distribution']]}")
    
    print("\n=== BiBo Forward Benchmark ===")
    bibo_timing = benchmark_forward(bibo, 'BiBo')
    results['bibo'] = {'router': bibo_router, 'timing': bibo_timing}
    del bibo; torch.cuda.empty_cache()
    
    # Qwen3MoE
    print("\n=== Qwen3MoE Router Analysis ===")
    qwen = load_model(Qwen3MoeForCausalLM, Qwen3MoeConfig, CFG['qwen3moe'], 'qwen3moe')
    qwen_router = extract_qwen_router(qwen, val_loader)
    for key, stats in sorted(qwen_router.items()):
        print(f"  {key}: entropy={stats['normalized_entropy']:.3f} confidence={stats['inverse_entropy']:.3f} balance_std={stats['load_balance_std']:.4f}")
        print(f"    dist={[f'{d:.3f}' for d in stats['expert_distribution']]}")
    
    print("\n=== Qwen3MoE Forward Benchmark ===")
    qwen_timing = benchmark_forward(qwen, 'Qwen3MoE')
    results['qwen3moe'] = {'router': qwen_router, 'timing': qwen_timing}
    del qwen; torch.cuda.empty_cache()
    
    # Save
    out_path = os.path.join(METRICS_DIR, 'post_training_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Next: python kaggle_ablations/plots.py")


if __name__ == '__main__':
    main()
