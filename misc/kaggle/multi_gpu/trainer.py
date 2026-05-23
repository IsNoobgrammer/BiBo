"""Training loop and helpers for parallel BiBo/Qwen training."""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os
import random

from model_utils import CFG, BASE_DIR
from losses import NTILLoss
from datasets import (
    SequenceDataset, ArithmeticDataset,
    CurriculumDataLoader, BucketedDataLoader,
)

__all__ = ['train_worker', 'seed_everything']

T = CFG['training']
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def seed_everything(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_worker(model_name):
    """Train one model. Called as separate process."""
    import wandb
    from src.configuration_bibo import BiBoConfig
    from src.modeling.models import BiBoForCausalLM
    from baseline.qwen3moe.config import Qwen3MoeConfig
    from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM

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

    # Loss function — NTIL (external, not inside model)
    # For arithmetic: disable EMD (not ordinal), keep CCE + seq-level
    task = T.get('task', 'sort')
    alpha_emd = T.get('alpha_emd', 0.3) if task == 'sort' else 0.0
    alpha_seq = T.get('alpha_seq', 0.1)

    ntil_loss_fn = NTILLoss(
        vocab_size=T['vocab_size'],
        sep_token=T['vocab_size'] - 1,
        alpha_emd=alpha_emd,
        alpha_seq=alpha_seq,
    ).to(device)

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
    task = T.get('task', 'sort')

    if task == 'arithmetic':
        # Arithmetic task: always bucketed (no curriculum for now)
        train_loader = BucketedDataLoader(DATA_DIR, 'train', T['batch_size'], stages, shuffle=True, task='arithmetic')
        val_loader = BucketedDataLoader(DATA_DIR, 'val', T['batch_size'], stages, shuffle=False, task='arithmetic')
        # OOD loader for generalization testing
        ood_buckets = T.get('arithmetic', {}).get('ood_buckets', [])
        ood_loaders = {}
        for min_t, max_t in ood_buckets:
            bucket_name = f'arith_{min_t}_{max_t}'
            ood_path = os.path.join(DATA_DIR, f'ood_{bucket_name}.npy')
            ood_len_path = os.path.join(DATA_DIR, f'ood_{bucket_name}_lengths.npy')
            if os.path.exists(ood_path):
                ds = ArithmeticDataset(ood_path, ood_len_path)
                ood_loaders[bucket_name] = DataLoader(
                    ds, batch_size=T['batch_size'], shuffle=False,
                    num_workers=T['num_workers'], pin_memory=True, drop_last=False)
        print(f"  [{model_name}] Task: arithmetic | OOD buckets: {list(ood_loaders.keys())}")
    elif use_curriculum:
        train_loader = CurriculumDataLoader(DATA_DIR, 'train', T['batch_size'], stages, shuffle=True)
        val_loader = CurriculumDataLoader(DATA_DIR, 'val', T['batch_size'], stages, shuffle=False)
        ood_loaders = {}
        print(f"  [{model_name}] Curriculum mode: stages={stages}")
    else:
        train_loader = BucketedDataLoader(DATA_DIR, 'train', T['batch_size'], stages, shuffle=True, task='sort')
        val_loader = BucketedDataLoader(DATA_DIR, 'val', T['batch_size'], stages, shuffle=False, task='sort')
        ood_loaders = {}
        print(f"  [{model_name}] Task: sort (bucketed)")

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

            # Forward WITHOUT labels — loss computed externally via NTIL
            if model_name == 'qwen3moe':
                outputs = model(input_ids=input_ids, output_router_logits=True)
            else:
                outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # Dataset already provides shifted labels (input=full[:-1], labels=full[1:])
            # No additional shift needed — logits[i] predicts labels[i]
            # NTIL loss (CCE + EMD + sequence-level)
            loss, loss_components = ntil_loss_fn(logits, labels)

            # Aux loss: only for Qwen3MoE (uses Switch Transformer style load balancing)
            # BiBo uses bias heuristic instead — no aux loss in the training objective
            aux_loss_val = 0.0
            if model_name == 'qwen3moe':
                if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                    from baseline.qwen3moe.modeling import load_balancing_loss_func
                    qwen_cfg = CFG['qwen3moe']
                    aux_loss = load_balancing_loss_func(
                        outputs.router_logits,
                        num_experts=qwen_cfg['num_experts'],
                        top_k=qwen_cfg['num_experts_per_tok'],
                    )
                    aux_coef = qwen_cfg.get('router_aux_loss_coef', 0.001)
                    loss = loss + aux_coef * aux_loss
                    aux_loss_val = aux_loss.item()

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
                'seq_len': seq_len, 'stage': stage_idx,
                'cce': loss_components['cce'], 'emd': loss_components['emd'],
                'seq_loss': loss_components['seq'], 'aux_loss': aux_loss_val,
            })

            # Log stage transitions
            if use_curriculum and stage_idx != current_stage:
                current_stage = stage_idx
                if stage_idx == len(train_loader.available_stages):
                    print(f"  [{model_name}] E{epoch:02d} → Mixed stage (10% each length)")
                else:
                    print(f"  [{model_name}] E{epoch:02d} → Stage {stage_idx}: seq_len={seq_len}")

            if batch_idx % T['log_every_n_steps'] == 0:
                stage_str = f" seq={seq_len}" if seq_len else ""
                print(f"  [{model_name}] E{epoch:02d} S{batch_idx:04d}/{len(train_loader)}"
                      f" | loss={step_loss:.4f} cce={loss_components['cce']:.4f}"
                      f" emd={loss_components['emd']:.4f} seq={loss_components['seq']:.4f}"
                      f" lr={cur_lr:.2e}{stage_str}")

            wandb.log({
                f'{model_name}/loss': step_loss,
                f'{model_name}/cce': loss_components['cce'],
                f'{model_name}/emd_raw': loss_components['emd'],
                f'{model_name}/seq_raw': loss_components['seq'],
                f'{model_name}/emd_scaled': loss_components['emd_scaled'],
                f'{model_name}/seq_scaled': loss_components['seq_scaled'],
                f'{model_name}/aux_loss': aux_loss_val,
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

                        # External loss (no labels to model)
                        v_out = model(input_ids=v_input)
                        v_logits = v_out.logits
                        v_l, _ = ntil_loss_fn(v_logits, v_labels)
                        v_loss += v_l.item()

                        mask = v_labels != -100
                        preds = v_logits.argmax(dim=-1)
                        v_correct += (preds[mask] == v_labels[mask]).sum().item()
                        v_total += mask.sum().item()
                        v_batches += 1
                v_loss /= max(v_batches, 1)
                v_acc = v_correct / max(v_total, 1)
                print(f"  [{model_name}] VAL @ step {global_step} | val_loss={v_loss:.4f} val_acc={v_acc:.4f}")
                wandb.log({f'{model_name}/val_loss': v_loss, f'{model_name}/val_acc': v_acc})
                metrics['val_checkpoints'].append({'step': global_step, 'val_loss': v_loss, 'val_acc': v_acc})

                # OOD evaluation (arithmetic only)
                if task == 'arithmetic' and ood_loaders:
                    ood_results = {}
                    with torch.no_grad():
                        for ood_name, ood_loader in ood_loaders.items():
                            ood_loss = 0
                            ood_correct = 0
                            ood_total = 0
                            ood_batches = 0
                            for ood_batch in ood_loader:
                                if isinstance(ood_batch, (list, tuple)):
                                    ood_input, ood_labels = ood_batch
                                else:
                                    ood_input, ood_labels = ood_batch
                                ood_input = ood_input.to(device)
                                ood_labels = ood_labels.to(device)
                                ood_out = model(input_ids=ood_input)
                                ood_l, _ = ntil_loss_fn(ood_out.logits, ood_labels)
                                ood_loss += ood_l.item()
                                ood_mask = ood_labels != -100
                                ood_preds = ood_out.logits.argmax(dim=-1)
                                ood_correct += (ood_preds[ood_mask] == ood_labels[ood_mask]).sum().item()
                                ood_total += ood_mask.sum().item()
                                ood_batches += 1
                            ood_loss /= max(ood_batches, 1)
                            ood_acc = ood_correct / max(ood_total, 1)
                            ood_results[ood_name] = {'loss': ood_loss, 'acc': ood_acc}
                            wandb.log({
                                f'{model_name}/ood_{ood_name}_loss': ood_loss,
                                f'{model_name}/ood_{ood_name}_acc': ood_acc,
                            })
                    ood_summary = " | ".join(f"{k}={v['acc']:.3f}" for k, v in ood_results.items())
                    print(f"  [{model_name}] OOD @ step {global_step} | {ood_summary}")

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
