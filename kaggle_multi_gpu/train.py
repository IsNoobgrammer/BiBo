"""
Parallel training — BiBo on cuda:0, Qwen3MoE on cuda:1.
Both log to same wandb run for side-by-side comparison.

Usage:
    python kaggle_multi_gpu/data.py      # generate data first
    python kaggle_multi_gpu/train.py     # train both in parallel
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
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']


# ============================================================
# Dataset (loads from disk)
# ============================================================

class SequenceDataset(Dataset):
    """
    Sorting task — single bucket (fixed length, no padding).
    Format: [unsorted] [SEP=2047] [sorted]
    Labels: [-100 for unsorted+SEP] [sorted tokens]
    """
    def __init__(self, npy_path):
        self.data = np.load(npy_path)  # [N, full_len]
        self.full_len = self.data.shape[1]
        self.seq_len = (self.full_len - 1) // 2  # original seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        full_seq = self.data[idx]
        input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)
        
        # Labels: only compute loss on sorted portion (after SEP)
        labels = torch.tensor(full_seq[1:], dtype=torch.long)
        # Mask everything up to and including SEP position
        # SEP is at index seq_len, so mask [0..seq_len] in labels (= first seq_len+1 positions shifted)
        labels[:self.seq_len] = -100
        
        return input_ids, labels


class BucketedDataLoader:
    """
    Iterates over multiple buckets (different seq lengths) in round-robin.
    Each batch contains same-length sequences (no padding needed).
    """
    def __init__(self, data_dir, split, batch_size, shuffle=True):
        self.datasets = []
        self.loaders = []
        
        for seq_len in [64, 128, 256]:
            path = os.path.join(data_dir, f'{split}_len_{seq_len}.npy')
            if os.path.exists(path):
                ds = SequenceDataset(path)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=T['num_workers'], pin_memory=True, drop_last=True)
                self.datasets.append(ds)
                self.loaders.append(loader)
        
        self.total_batches = sum(len(l) for l in self.loaders)
    
    def __len__(self):
        return self.total_batches
    
    def __iter__(self):
        """Round-robin across buckets."""
        iterators = [iter(l) for l in self.loaders]
        active = list(range(len(iterators)))
        
        while active:
            for i in list(active):
                try:
                    batch = next(iterators[i])
                    yield batch
                except StopIteration:
                    active.remove(i)


# ============================================================
# Training function (runs in subprocess)
# ============================================================

def train_worker(model_name):
    """Train one model. Called as separate process."""
    import wandb
    
    # Config
    if model_name == 'bibo':
        model_cfg = CFG['bibo']
        device = torch.device(model_cfg.pop('device', 'cuda:0'))
        model = BiBoForCausalLM(BiBoConfig(**model_cfg)).to(device)
    else:
        model_cfg = CFG['qwen3moe']
        device = torch.device(model_cfg.pop('device', 'cuda:1'))
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**model_cfg)).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # wandb — separate run per model, grouped for comparison
    wandb.init(
        project=T['wandb_project'],
        name=f"{T['wandb_run_name']}-{model_name}",
        group="parallel-2xT4",
        config={**T, 'model': model_name, 'params': n_params},
    )
    
    # Data — bucketed (no padding, variable length batches)
    train_loader = BucketedDataLoader(DATA_DIR, 'train', T['batch_size'], shuffle=True)
    val_loader = BucketedDataLoader(DATA_DIR, 'val', T['batch_size'], shuffle=False)
    
    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])
    
    total_steps = T['epochs'] * len(train_loader)
    warmup_steps = int(total_steps * T['warmup_ratio'])
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Metrics storage
    metrics = {
        'model_name': model_name,
        'params': n_params,
        'device': str(device),
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'steps': [],
        'epochs': [],
    }
    
    print(f"\n[{model_name}] Params: {n_params:,} | Device: {device}")
    print(f"[{model_name}] Steps/epoch: {len(train_loader)} | Total: {total_steps} | Warmup: {warmup_steps}")
    
    # Train
    global_step = 0
    for epoch in range(T['epochs']):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids, labels = input_ids.to(device), labels.to(device)
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
            
            wandb.log({
                f'{model_name}/loss': step_loss,
                f'{model_name}/lr': cur_lr,
            })
        
        epoch_time = time.time() - t0
        train_loss = total_loss / n_batches
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, labels=labels)
                val_loss += outputs.loss.item()
                # Accuracy only on non-masked positions (target portion)
                mask = labels != -100
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        metrics['epochs'].append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'time': epoch_time
        })
        
        print(f"  [{model_name}] === E{epoch:02d} | train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} time={epoch_time:.1f}s ===")
        wandb.log({
            f'{model_name}/train_loss_epoch': train_loss,
            f'{model_name}/val_loss': val_loss,
            f'{model_name}/val_acc': val_acc,
            f'{model_name}/epoch_time': epoch_time,
        })
    
    # Save
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}.pt'))
    with open(os.path.join(METRICS_DIR, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  [{model_name}] DONE. Saved checkpoint + metrics.")
    wandb.finish()


# ============================================================
# Main — spawn both processes
# ============================================================

def main():
    torch.manual_seed(T['seed'])
    np.random.seed(T['seed'])
    
    # Verify data exists
    train_path = os.path.join(DATA_DIR, 'train.npy')
    val_path = os.path.join(DATA_DIR, 'val.npy')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: Data not found. Run `python kaggle_multi_gpu/data.py` first.")
        sys.exit(1)
    
    # Spawn parallel processes
    print("\nLaunching parallel training...")
    print(f"  BiBo → cuda:0")
    print(f"  Qwen3MoE → cuda:1")
    print()
    
    mp.set_start_method('spawn', force=True)
    
    p_bibo = mp.Process(target=train_worker, args=('bibo',))
    p_qwen = mp.Process(target=train_worker, args=('qwen3moe',))
    
    p_bibo.start()
    p_qwen.start()
    
    p_bibo.join()
    p_qwen.join()
    
    print("\n" + "="*60)
    print("BOTH MODELS DONE")
    print("="*60)
    print(f"  Checkpoints: {SAVE_DIR}/")
    print(f"  Metrics: {METRICS_DIR}/")
    print(f"  wandb: https://wandb.ai/<entity>/{T['wandb_project']}/runs/{run_id}")
    print("  Next: python kaggle_multi_gpu/extract_metrics.py")


if __name__ == '__main__':
    main()
