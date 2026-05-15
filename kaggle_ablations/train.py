"""
Train BiBo vs Qwen3MoE — Kaggle T4
===================================
Only trains, logs metrics, saves models.

Usage:
    !git clone https://github.com/IsNoobgrammer/BiBo.git
    %cd BiBo
    !pip install -q transformers einops wandb bitsandbytes pyyaml
    !python kaggle_ablations/train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
import yaml
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM

# Load config
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']
DEVICE = torch.device(T['device'])
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

USE_WANDB = T['use_wandb']
try:
    import wandb
except ImportError:
    USE_WANDB = False
    print("wandb not installed, terminal-only logging")


# ============================================================
# Data
# ============================================================

class HardSequenceDataset(Dataset):
    """Complex synthetic sequences — polynomial + periodic + skip + XOR patterns."""
    def __init__(self, num_samples, seq_len, vocab_size, seed=42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.data = np.zeros((num_samples, seq_len), dtype=np.int64)
        
        for s in range(num_samples):
            seq = self.data[s]
            seq[:5] = rng.integers(0, vocab_size, size=5)
            pattern_weights = rng.dirichlet(np.ones(5))
            period = rng.integers(5, 18)
            delta = rng.integers(1, vocab_size // 10)
            
            for i in range(5, seq_len):
                r = rng.random()
                cumw = np.cumsum(pattern_weights)
                if r < 0.10:
                    seq[i] = rng.integers(0, vocab_size)
                elif r < cumw[0]:
                    seq[i] = (seq[i-1] * seq[i-2] + seq[i-1] * 3 + 17) % vocab_size
                elif r < cumw[1]:
                    seq[i] = (seq[i % period] + i * 7) % vocab_size
                elif r < cumw[2]:
                    skip = 5 if i >= 7 else 2
                    seq[i] = (seq[i-1] * 11 + seq[i-skip] * 23 + 5) % vocab_size
                elif r < cumw[3]:
                    seq[i] = (seq[i-1] + delta) % vocab_size
                else:
                    seq[i] = (seq[i-1] ^ seq[i-3]) % vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]


# ============================================================
# Model builders
# ============================================================

def make_bibo_model():
    c = CFG['bibo']
    config = BiBoConfig(**c)
    return BiBoForCausalLM(config).to(DEVICE)

def make_qwen_model():
    c = CFG['qwen3moe']
    config = Qwen3MoeConfig(**c)
    return Qwen3MoeForCausalLM(config).to(DEVICE)

def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Training
# ============================================================

def make_optimizer(model):
    if T['optimizer'] == 'adamw_8bit':
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])
        except ImportError:
            print("  bitsandbytes not available, falling back to AdamW")
    return optim.AdamW(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])


def train_model(model, model_name, train_loader, val_loader):
    optimizer = make_optimizer(model)
    
    total_steps = T['epochs'] * len(train_loader)
    warmup_steps = int(total_steps * T['warmup_ratio'])
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    metrics = {
        'model_name': model_name,
        'params': count_params(model),
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'steps': [],
        'epochs': [],
    }
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Params: {count_params(model):,}")
    print(f"  Steps/epoch: {len(train_loader)} | Total: {total_steps} | Warmup: {warmup_steps}")
    print(f"{'='*60}")
    
    global_step = 0
    for epoch in range(T['epochs']):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), T['grad_clip'])
            optimizer.step()
            scheduler.step()
            
            step_loss = loss.item()
            cur_lr = scheduler.get_last_lr()[0]
            total_loss += step_loss
            n_batches += 1
            global_step += 1
            
            metrics['steps'].append({'step': global_step, 'loss': step_loss, 'lr': cur_lr})
            
            if batch_idx % T['log_every_n_steps'] == 0:
                print(f"  [{model_name}] E{epoch:02d} S{batch_idx:03d}/{len(train_loader)} | loss={step_loss:.4f} lr={cur_lr:.2e}")
            
            if USE_WANDB:
                wandb.log({f'{model_name}/loss': step_loss, f'{model_name}/lr': cur_lr})
        
        epoch_time = time.time() - t0
        train_loss = total_loss / n_batches
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
                outputs = model(input_ids=input_ids, labels=labels)
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        metrics['epochs'].append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'time': epoch_time
        })
        
        print(f"  [{model_name}] === E{epoch:02d} | train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} time={epoch_time:.1f}s ===")
        if USE_WANDB:
            wandb.log({
                f'{model_name}/train_loss_epoch': train_loss,
                f'{model_name}/val_loss': val_loss,
                f'{model_name}/val_acc': val_acc,
            })
    
    # Save
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}.pt'))
    with open(os.path.join(METRICS_DIR, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Saved: checkpoints/{model_name}.pt + metrics/{model_name}_metrics.json")
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(T['seed'])
    np.random.seed(T['seed'])
    
    if USE_WANDB:
        wandb.init(project=T['wandb_project'], name=T['wandb_run_name'], config=CFG)
    
    # Data
    print("Generating data...")
    train_ds = HardSequenceDataset(T['train_samples'], T['seq_len'], T['vocab_size'], seed=42)
    val_ds = HardSequenceDataset(T['val_samples'], T['seq_len'], T['vocab_size'], seed=123)
    train_loader = DataLoader(train_ds, batch_size=T['batch_size'], shuffle=True,
                              num_workers=T['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=T['batch_size'], shuffle=False,
                            num_workers=T['num_workers'], pin_memory=True)
    print(f"  Train: {len(train_ds)} ({len(train_loader)} steps/epoch)")
    print(f"  Val: {len(val_ds)} ({len(val_loader)} steps)")
    
    # Save data config
    with open(os.path.join(METRICS_DIR, 'data_config.json'), 'w') as f:
        json.dump(T, f, indent=2)
    
    # BiBo
    bibo = make_bibo_model()
    train_model(bibo, 'bibo', train_loader, val_loader)
    del bibo; torch.cuda.empty_cache()
    
    # Qwen3MoE
    qwen = make_qwen_model()
    train_model(qwen, 'qwen3moe', train_loader, val_loader)
    del qwen; torch.cuda.empty_cache()
    
    if USE_WANDB:
        wandb.finish()
    
    print("\n" + "="*60)
    print("DONE. Next: python kaggle_ablations/extract_metrics.py")
    print("="*60)


if __name__ == '__main__':
    main()
