"""
Parallel training — BiBo on cuda:0, Qwen3MoE on cuda:1.
Supports curriculum learning: train on progressively longer sequences.

Usage:
    python misc/kaggle/multi_gpu/data.py      # generate data first
    python misc/kaggle/multi_gpu/train.py     # train both
    python misc/kaggle/multi_gpu/train.py --seed 69
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
import argparse
import random
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

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
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    """
    Sorting task — single bucket (fixed length, no padding).
    Format: [unsorted] [SEP] [sorted]
    Labels: [-100 for unsorted+SEP] [sorted tokens]
    """
    def __init__(self, npy_path):
        self.data = np.load(npy_path)
        self.full_len = self.data.shape[1]
        self.seq_len = (self.full_len - 1) // 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data[idx]
        input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)

        labels = torch.tensor(full_seq[1:], dtype=torch.long)
        labels[:self.seq_len] = -100

        return input_ids, labels


class CurriculumDataLoader:
    """
    Curriculum learning: iterates through stages of increasing seq_len.
    Each stage trains on its own bucket for a portion of the epoch.
    
    Strategy: within each epoch, cycle through all stages in order.
    Earlier stages (shorter) are easier → model learns algorithm first.
    """
    def __init__(self, data_dir, split, batch_size, stages, shuffle=True):
        self.stages = stages
        self.loaders = {}
        self.datasets = {}
        
        for seq_len in stages:
            path = os.path.join(data_dir, f'{split}_len_{seq_len}.npy')
            if os.path.exists(path):
                ds = SequenceDataset(path)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=T['num_workers'], pin_memory=True, drop_last=True)
                self.datasets[seq_len] = ds
                self.loaders[seq_len] = loader
            else:
                print(f"  WARNING: {path} not found, skipping stage seq_len={seq_len}")
        
        self.available_stages = [s for s in stages if s in self.loaders]
        self.total_batches = sum(len(self.loaders[s]) for s in self.available_stages)
    
    def __len__(self):
        return self.total_batches
    
    def __iter__(self):
        """Iterate through stages in curriculum order (short → long)."""
        for stage_idx, seq_len in enumerate(self.available_stages):
            loader = self.loaders[seq_len]
            for batch in loader:
                yield batch, seq_len, stage_idx


class BucketedDataLoader:
    """
    Non-curriculum: round-robin across all buckets (original behavior).
    """
    def __init__(self, data_dir, split, batch_size, stages=None, shuffle=True):
        self.datasets = []
        self.loaders = []
        
        seq_lens = stages if stages else [64, 128, 256]
        for seq_len in seq_lens:
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
        iterators = [iter(l) for l in self.loaders]
        active = list(range(len(iterators)))
        while active:
            for i in list(active):
                try:
                    batch = next(iterators[i])
                    yield batch, None, None
                except StopIteration:
                    active.remove(i)


# ============================================================
# Training function (runs in subprocess)
# ============================================================

def train_worker(model_name):
    """Train one model. Called as separate process."""
    import wandb

    seed_everything(T['seed'])

    # Config
    if model_name == 'bibo':
        model_cfg = {k: v for k, v in CFG['bibo'].items() if k != 'device'}
        device = torch.device(CFG['bibo'].get('device', 'cuda:0'))
        model = BiBoForCausalLM(BiBoConfig(**model_cfg)).to(device)
    else:
        model_cfg = {k: v for k, v in CFG['qwen3moe'].items() if k != 'device'}
        device = torch.device(CFG['qwen3moe'].get('device', 'cuda:1'))
        model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**model_cfg)).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    # wandb
    wandb.init(
        project=T['wandb_project'],
        name=f"{T['wandb_run_name']}-{model_name}",
        group="parallel-2xT4-curriculum",
        config={**T, 'model': model_name, 'params': n_params},
    )

    # Data loader — curriculum or bucketed
    use_curriculum = T.get('curriculum', False)
    stages = T.get('curriculum_stages', [64, 128, 256])
    
    if use_curriculum:
        train_loader = CurriculumDataLoader(DATA_DIR, 'train', T['batch_size'], stages, shuffle=True)
        val_loader = CurriculumDataLoader(DATA_DIR, 'val', T['batch_size'], stages, shuffle=False)
        print(f"  [{model_name}] Curriculum mode: stages={stages}")
    else:
        train_loader = BucketedDataLoader(DATA_DIR, 'train', T['batch_size'], stages, shuffle=True)
        val_loader = BucketedDataLoader(DATA_DIR, 'val', T['batch_size'], stages, shuffle=False)

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=T['lr'], weight_decay=T['weight_decay'])

    total_steps = T['epochs'] * len(train_loader)
    warmup_steps = int(total_steps * T['warmup_ratio'])
    val_every = T.get('val_every_n_steps', len(train_loader))

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
        'curriculum': use_curriculum,
        'stages': stages,
        'steps': [],
        'epochs': [],
        'val_checkpoints': [],
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
        current_stage = None

        for batch_idx, (batch_data, seq_len, stage_idx) in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)):
                input_ids, labels = batch_data
            else:
                input_ids, labels = batch_data

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

            metrics['steps'].append({
                'step': global_step, 'loss': step_loss, 'lr': cur_lr,
                'seq_len': seq_len, 'stage': stage_idx
            })

            # Log stage transitions
            if use_curriculum and stage_idx != current_stage:
                current_stage = stage_idx
                print(f"  [{model_name}] E{epoch:02d} → Stage {stage_idx}: seq_len={seq_len}")

            if batch_idx % T['log_every_n_steps'] == 0:
                stage_str = f" seq={seq_len}" if seq_len else ""
                print(f"  [{model_name}] E{epoch:02d} S{batch_idx:04d}/{len(train_loader)}"
                      f" | loss={step_loss:.4f} lr={cur_lr:.2e}{stage_str}")

            wandb.log({
                f'{model_name}/loss': step_loss,
                f'{model_name}/lr': cur_lr,
                f'{model_name}/seq_len': seq_len if seq_len else 0,
                'global_step': global_step,
            })

            # Validation
            if global_step % val_every == 0:
                model.eval()
                v_loss = 0
                v_correct = 0
                v_total = 0
                v_batches = 0
                with torch.no_grad():
                    for v_batch, v_sl, v_si in val_loader:
                        if isinstance(v_batch, (list, tuple)):
                            v_input, v_labels = v_batch
                        else:
                            v_input, v_labels = v_batch
                        v_input, v_labels = v_input.to(device), v_labels.to(device)
                        v_out = model(input_ids=v_input, labels=v_labels)
                        v_loss += v_out.loss.item()
                        mask = v_labels != -100
                        preds = v_out.logits.argmax(dim=-1)
                        v_correct += (preds[mask] == v_labels[mask]).sum().item()
                        v_total += mask.sum().item()
                        v_batches += 1
                v_loss /= max(v_batches, 1)
                v_acc = v_correct / max(v_total, 1)
                print(f"  [{model_name}] VAL @ step {global_step} | val_loss={v_loss:.4f} val_acc={v_acc:.4f}")
                wandb.log({f'{model_name}/val_loss': v_loss, f'{model_name}/val_acc': v_acc})
                metrics['val_checkpoints'].append({'step': global_step, 'val_loss': v_loss, 'val_acc': v_acc})
                model.train()

        epoch_time = time.time() - t0
        train_loss = total_loss / max(n_batches, 1)

        metrics['epochs'].append({
            'epoch': epoch, 'train_loss': train_loss, 'time': epoch_time
        })

        print(f"  [{model_name}] === E{epoch:02d} | train={train_loss:.4f} time={epoch_time:.1f}s ===")

    # Save
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{model_name}.pt'))
    with open(os.path.join(METRICS_DIR, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  [{model_name}] DONE. Saved checkpoint + metrics.")
    wandb.finish()


def seed_everything(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Main — spawn both processes
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BiBo vs Qwen3MoE parallel training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config.yaml, default: config value)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else T['seed']
    T['seed'] = seed

    seed_everything(seed)
    print(f"\n  Seed: {seed}")
    print(f"  Curriculum: {T.get('curriculum', False)}")
    if T.get('curriculum'):
        print(f"  Stages: {T.get('curriculum_stages')}")
    print(f"  Epochs: {T['epochs']}")
    print(f"  LR: {T['lr']}")
    print(f"  Vocab: {T['vocab_size']}")

    # Verify data exists
    stages = T.get('curriculum_stages', [64, 128, 256])
    check_file = os.path.join(DATA_DIR, f'train_len_{stages[0]}.npy')
    if not os.path.exists(check_file):
        print(f"ERROR: Data not found ({check_file}). Run `python misc/kaggle/multi_gpu/data.py` first.")
        sys.exit(1)

    # Spawn parallel processes
    print("\nLaunching parallel training...")
    print(f"  BiBo → {CFG['bibo'].get('device', 'cuda:0')}")
    print(f"  Qwen3MoE → {CFG['qwen3moe'].get('device', 'cuda:1')}")
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
    print("  Next: python misc/kaggle/multi_gpu/analyze_model.py")


if __name__ == '__main__':
    main()
