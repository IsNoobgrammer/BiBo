"""
T4x2 Benchmark — BiMo vs Qwen3MoE parallel training.

Target: val_loss ~2.8 (±5% z-score → 2.66 - 2.94)
Hardware: 2× T4 16GB (Kaggle)
Precision: fp16 (T4 does NOT support bf16)

Usage:
    python misc/kaggle/t4x2_bench/data.py      # generate data first
    python misc/kaggle/t4x2_bench/train.py     # train both models
    python misc/kaggle/t4x2_bench/train.py --model bibo   # train only BiMo
    python misc/kaggle/t4x2_bench/train.py --model qwen   # train only Qwen
"""
import os
import sys
import time
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SortDataset(Dataset):
    """Sorting task dataset — input+target pairs."""
    def __init__(self, data_path, seq_len):
        self.data = np.load(data_path)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data[idx]
        x = torch.from_numpy(row[:-1].astype(np.int64))
        y = torch.from_numpy(row[1:].astype(np.int64))
        return x, y


def get_lr(step, total_steps, warmup_steps, lr, min_lr):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, device, num_steps=100):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= num_steps:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, labels=y)
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_model(model_name, cfg, train_cfg):
    """Train a single model."""
    device = torch.device(cfg.get('device', 'cuda:0'))
    seq_len = train_cfg['seq_len']
    
    # Model
    model_cfg = {k: v for k, v in cfg.items() if k != 'device'}
    if model_name == 'bibo':
        model = BiBoForCausalLM(BiBoConfig(**model_cfg))
    else:
        from baseline.qwen3moe.config import Qwen3MoeConfig
        from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
        model = BiBoForCausalLM(BiBoConfig(**model_cfg)) if model_name == 'bibo' else \
                Qwen3MoeForCausalLM(Qwen3MoeConfig(**model_cfg))
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{model_name}] {n_params/1e6:.1f}M params on {device}")
    
    # Compile if supported
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, dynamic=False)
            print(f"  [{model_name}] Compiled with torch.compile")
        except Exception as e:
            print(f"  [{model_name}] Compile failed: {e}")
    
    # Data
    train_data = SortDataset(os.path.join(DATA_DIR, 'train_sort.npy'), seq_len)
    val_data = SortDataset(os.path.join(DATA_DIR, 'val_sort.npy'), seq_len)
    
    train_loader = DataLoader(
        train_data, batch_size=train_cfg['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_data, batch_size=train_cfg['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Training
    total_steps = train_cfg['epochs'] * len(train_loader)
    warmup_steps = train_cfg['warmup_epochs'] * len(train_loader)
    grad_clip = train_cfg['grad_clip']
    eval_every = train_cfg['eval_every']
    eval_steps = train_cfg['eval_steps']
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'step': [],
        'time': [],
    }
    
    best_val_loss = float('inf')
    step = 0
    t0 = time.time()
    
    print(f"  [{model_name}] Training for {total_steps} steps...")
    
    for epoch in range(train_cfg['epochs']):
        for x, y in train_loader:
            # LR schedule
            lr = get_lr(step, total_steps, warmup_steps, train_cfg['lr'], train_cfg['min_lr'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward
            x, y = x.to(device), y.to(device)
            out = model(x, labels=y)
            loss = out.loss
            
            # Backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Log
            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [{model_name}] step={step}/{total_steps} loss={loss.item():.4f} lr={lr:.6f} time={elapsed:.0f}s")
                metrics['train_loss'].append(loss.item())
                metrics['lr'].append(lr)
                metrics['step'].append(step)
                metrics['time'].append(elapsed)
            
            # Eval
            if step > 0 and step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device, eval_steps)
                metrics['val_loss'].append({'step': step, 'loss': val_loss})
                print(f"  [{model_name}] EVAL step={step} val_loss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save checkpoint
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    ckpt = model.state_dict()
                    # Strip _orig_mod prefix from compiled model
                    clean_ckpt = {}
                    for k, v in ckpt.items():
                        clean_ckpt[k.replace('_orig_mod.', '')] = v
                    torch.save(clean_ckpt, os.path.join(SAVE_DIR, f'{model_name}.pt'))
                    print(f"  [{model_name}] Saved checkpoint (best val_loss={best_val_loss:.4f})")
            
            step += 1
    
    # Final eval
    val_loss = evaluate(model, val_loader, device, eval_steps)
    metrics['val_loss'].append({'step': step, 'loss': val_loss})
    
    elapsed = time.time() - t0
    print(f"\n  [{model_name}] DONE in {elapsed:.0f}s")
    print(f"  [{model_name}] Best val_loss: {best_val_loss:.4f}")
    print(f"  [{model_name}] Final val_loss: {val_loss:.4f}")
    
    # Check target
    target = 2.8
    z_score = abs(val_loss - target) / target * 100
    in_range = z_score <= 5.0
    print(f"  [{model_name}] Z-score: {z_score:.1f}% ({'PASS' if in_range else 'FAIL'} — target 2.8 ±5%)")
    
    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics['best_val_loss'] = best_val_loss
    metrics['final_val_loss'] = val_loss
    metrics['z_score'] = z_score
    metrics['pass'] = in_range
    metrics['n_params'] = n_params
    metrics['elapsed_s'] = elapsed
    
    with open(os.path.join(METRICS_DIR, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='T4x2 Benchmark')
    parser.add_argument('--model', type=str, default='both', choices=['bibo', 'qwen', 'both'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(os.path.join(BASE_DIR, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    
    seed = args.seed if args.seed is not None else cfg['training']['seed']
    seed_everything(seed)
    
    train_cfg = cfg['training']
    print("=" * 60)
    print("T4x2 Benchmark — BiMo vs Qwen3MoE")
    print("=" * 60)
    print(f"  Seed: {seed}")
    print(f"  Task: {train_cfg['task']}")
    print(f"  Target: val_loss ~2.8 (±5%)")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Batch: {train_cfg['batch_size']}")
    print(f"  LR: {train_cfg['lr']}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    results = {}
    
    if args.model in ('bibo', 'both'):
        print("\n" + "=" * 60)
        print("Training BiMo (BiBo MoE)")
        print("=" * 60)
        results['bibo'] = train_model('bibo', cfg['bibo'], train_cfg)
    
    if args.model in ('qwen', 'both'):
        print("\n" + "=" * 60)
        print("Training Qwen3MoE")
        print("=" * 60)
        results['qwen3moe'] = train_model('qwen3moe', cfg['qwen3moe'], train_cfg)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, m in results.items():
        status = "PASS" if m['pass'] else "FAIL"
        print(f"  {name}: val_loss={m['final_val_loss']:.4f} z={m['z_score']:.1f}% {status} ({m['n_params']/1e6:.1f}M params, {m['elapsed_s']:.0f}s)")


if __name__ == '__main__':
    main()
