"""
Model Output Analysis — Arithmetic Task
=========================================

Evaluates BiBo vs Qwen3MoE on arithmetic chain-of-thought.

Arithmetic format: [phase1] [SEP] [phase2] [SEP] [phase3]
  Phase 1: raw expression (e.g. 3 * 4 + 2 - 6 // 3)
  Phase 2: precedence resolved (mul/div computed, e.g. 12 + 2 - 2)
  Phase 3: final answer (e.g. 8)
  Labels: predict from first SEP onward

Metrics:
  1. Loss per bucket (cross-entropy)
  2. Token accuracy (exact match per token in target region)
  3. Phase 2 accuracy (all intermediate tokens correct)
  4. Phase 3 accuracy (final answer correct)
  5. Full sequence accuracy (everything correct)
  6. OOD generalization (held-out difficulty buckets)

Usage:
    python misc/kaggle/multi_gpu/analyze_model.py
"""
import sys
import os
import torch
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from model_utils import CFG, load_models, BASE_DIR
from plot_utils import PLOTS_DIR, METRICS_DIR
from datasets import ArithmeticDataset, arithmetic_collate_fn

T = CFG['training']
ARITH = T.get('arithmetic', {})
VOCAB_SIZE = T['vocab_size']
SEP_TOKEN = VOCAB_SIZE - 1
BATCH_SIZE = 8

OP_TOKENS = {
    VOCAB_SIZE - 2: '+',
    VOCAB_SIZE - 3: '-',
    VOCAB_SIZE - 4: '*',
    VOCAB_SIZE - 5: '//',
}


def decode_token(t):
    """Decode a single token to string."""
    if t == 0:
        return '<PAD>'
    if t == SEP_TOKEN:
        return 'SEP'
    if t in OP_TOKENS:
        return OP_TOKENS[t]
    return str(t)


def decode_sequence(tokens):
    """Decode token sequence to human-readable string."""
    return ' '.join(decode_token(t) for t in tokens if t != 0)


def find_sep_positions(seq):
    """Find SEP token positions in a sequence."""
    return [i for i, t in enumerate(seq) if t == SEP_TOKEN]


def split_phases(seq):
    """Split sequence into phases at SEP boundaries."""
    seps = find_sep_positions(seq)
    if len(seps) < 2:
        return seq, [], []
    phase1 = seq[:seps[0]]
    phase2 = seq[seps[0]+1:seps[1]]
    phase3 = seq[seps[1]+1:]
    # Strip padding from phase3
    phase3 = [t for t in phase3 if t != 0]
    return phase1, phase2, phase3


def evaluate_model(model, input_ids, labels, device):
    """Run model forward and collect predictions."""
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        labels_dev = labels.to(device)
        out = model(input_ids=input_ids, labels=labels_dev)
        logits = out.logits.cpu()
        loss = out.loss.item()

    # Predictions (greedy)
    preds = logits.argmax(dim=-1)  # [B, S]

    return {
        'loss': loss,
        'logits': logits,
        'preds': preds,
        'labels': labels,
        'input_ids': input_ids.cpu(),
    }


def compute_metrics(eval_result, full_sequences):
    """Compute arithmetic-specific metrics."""
    preds = eval_result['preds']      # [B, S] shifted predictions
    labels = eval_result['labels']    # [B, S] shifted labels (-100 = ignore)
    input_ids = eval_result['input_ids']  # [B, S] original input

    mask = (labels != -100)
    if not mask.any():
        return {
            'loss': eval_result['loss'],
            'token_accuracy': 0.0,
            'phase2_accuracy': 0.0,
            'phase3_accuracy': 0.0,
            'full_sequence_accuracy': 0.0,
        }

    # Token accuracy (only on target positions)
    correct = (preds == labels) & mask
    token_acc = correct.sum().float() / mask.sum().float()

    # Per-sample analysis
    batch_size = preds.shape[0]
    phase2_correct = 0
    phase3_correct = 0
    full_correct = 0

    for i in range(batch_size):
        # Get the full original sequence for this sample
        full_seq = full_sequences[i]
        seps = find_sep_positions(full_seq)

        if len(seps) < 2:
            continue

        # Phase 2 starts after first SEP in the original sequence
        # In shifted labels, labels[first_sep-1] = SEP token
        # Then labels[first_sep:seps[1]-1] = phase2 tokens
        # labels[seps[1]-1] = second SEP
        # labels[seps[1]:] = phase3 tokens

        first_sep = seps[0]
        second_sep = seps[1]

        # Phase 2 target region in shifted labels: positions first_sep-1 to second_sep-2
        # (the model predicts SEP at first_sep-1, then phase2 tokens, then second SEP)
        p2_start = first_sep - 1  # shifted: predict SEP here
        p2_end = second_sep - 1   # shifted: predict second SEP here

        # Phase 3 target region: positions second_sep onwards (predict final answer)
        p3_start = second_sep  # shifted: predict first token of phase3 here

        # Check phase 2 accuracy (all phase2 tokens + SEPs correct)
        if p2_start >= 0 and p2_end < labels.shape[1]:
            p2_mask = mask[i, p2_start:p2_end]
            p2_correct = (preds[i, p2_start:p2_end] == labels[i, p2_start:p2_end]) & p2_mask
            if p2_mask.any() and p2_correct.sum() == p2_mask.sum():
                phase2_correct += 1

        # Check phase 3 accuracy (final answer token correct)
        if p3_start < labels.shape[1]:
            p3_mask = mask[i, p3_start:]
            p3_preds = preds[i, p3_start:]
            p3_labels = labels[i, p3_start:]
            if p3_mask.any():
                # Phase3 is just the final answer — check if it matches
                p3_correct = (p3_preds == p3_labels) & p3_mask
                if p3_correct.sum() == p3_mask.sum():
                    phase3_correct += 1

        # Full sequence accuracy
        sample_mask = mask[i]
        sample_correct = (preds[i] == labels[i]) & sample_mask
        if sample_mask.any() and sample_correct.sum() == sample_mask.sum():
            full_correct += 1

    return {
        'loss': eval_result['loss'],
        'token_accuracy': token_acc.item(),
        'phase2_accuracy': phase2_correct / batch_size,
        'phase3_accuracy': phase3_correct / batch_size,
        'full_sequence_accuracy': full_correct / batch_size,
    }


def evaluate_bucket(model, data_path, lengths_path, device):
    """Evaluate model on one data bucket."""
    ds = ArithmeticDataset(data_path, lengths_path)

    all_metrics = []
    for i in range(0, len(ds), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(ds))
        batch = [ds[j] for j in range(i, batch_end)]
        input_ids, labels = arithmetic_collate_fn(batch)

        # Get full sequences for phase analysis
        full_seqs = [ds.data[j] for j in range(i, batch_end)]

        eval_result = evaluate_model(model, input_ids, labels, device)
        metrics = compute_metrics(eval_result, full_seqs)
        all_metrics.append(metrics)

    # Average metrics
    avg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        avg[key] = sum(vals) / len(vals)
    avg['num_samples'] = len(ds)
    return avg


def display_samples(model, data_path, lengths_path, device, model_name, num_samples=3):
    """Display sample predictions."""
    ds = ArithmeticDataset(data_path, lengths_path)
    if len(ds) == 0:
        return

    indices = list(range(min(num_samples, len(ds))))
    batch = [ds[j] for j in indices]
    input_ids, labels = arithmetic_collate_fn(batch)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids.to(device), labels=labels.to(device).float() if labels.is_floating_point() else labels.to(device))
        preds = out.logits.argmax(dim=-1).cpu()

    print(f"\n  [{model_name}] Sample predictions:")
    for idx in range(len(indices)):
        full_seq = ds.data[indices[idx]]
        seps = find_sep_positions(full_seq)
        if len(seps) < 2:
            continue

        phase1, phase2, phase3 = split_phases(full_seq)
        print(f"    Sample {idx}:")
        print(f"      Input:  {decode_sequence(phase1)}")
        print(f"      Phase2: {decode_sequence(phase2)}")
        print(f"      Phase3: {decode_sequence(phase3)}")

        # Show predictions for target region
        first_sep = seps[0]
        pred_target = preds[idx, first_sep-1:]  # predictions from SEP onward
        label_target = labels[idx, first_sep-1:]
        mask = label_target != -100

        if mask.any():
            pred_tokens = pred_target[mask].tolist()
            label_tokens = label_target[mask].tolist()
            print(f"      Pred:   {decode_sequence(pred_tokens)}")
            print(f"      Match:  {'✓' if pred_tokens == label_tokens else '✗'}")


def print_summary_table(all_results, bucket_names):
    """Print comparison table."""
    print(f"\n{'='*100}")
    print(f"  SUMMARY — Arithmetic: BiBo vs Qwen3MoE")
    print(f"{'='*100}")
    print(f"  {'Bucket':<16} {'| BiBo Loss':<12} {'TokAcc':<8} {'Ph2Acc':<8} {'Ph3Acc':<8} {'Full':<6}"
          f" {'| Qwen Loss':<12} {'TokAcc':<8} {'Ph2Acc':<8} {'Ph3Acc':<8} {'Full':<6}")
    print(f"  {'-'*15} {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*5}"
          f" {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*5}")

    bibo_wins = 0
    qwen_wins = 0

    for name in bucket_names:
        b = all_results['bibo'].get(name, {})
        q = all_results['qwen3moe'].get(name, {})
        if not b or not q:
            continue

        b_acc = b.get('token_accuracy', 0)
        q_acc = q.get('token_accuracy', 0)
        if b_acc > q_acc:
            bibo_wins += 1
        elif q_acc > b_acc:
            qwen_wins += 1

        print(f"  {name:<15} | {b.get('loss', 0):<10.4f} {b_acc:<8.4f} "
              f"{b.get('phase2_accuracy', 0):<8.4f} {b.get('phase3_accuracy', 0):<8.4f} {b.get('full_sequence_accuracy', 0):<6.4f}"
              f" | {q.get('loss', 0):<10.4f} {q_acc:<8.4f} "
              f"{q.get('phase2_accuracy', 0):<8.4f} {q.get('phase3_accuracy', 0):<8.4f} {q.get('full_sequence_accuracy', 0):<6.4f}")

    print(f"  {'-'*95}")
    print(f"  Score: BiBo {bibo_wins} — Qwen {qwen_wins}")
    print(f"{'='*100}")


def main():
    print("\n" + "="*70)
    print("  MODEL OUTPUT ANALYSIS — Arithmetic (BiBo vs Qwen3MoE)")
    print("="*70)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")

    print("\nLoading models...")
    bibo_model, qwen_model = load_models(device)

    data_dir = os.path.join(BASE_DIR, 'data')
    buckets = ARITH.get('buckets', [[3, 7], [9, 16], [19, 30], [35, 50]])
    ood_buckets = ARITH.get('ood_buckets', [])

    all_results = {'bibo': {}, 'qwen3moe': {}}
    bucket_names = []

    # In-distribution buckets
    print(f"\n{'='*70}")
    print(f"  In-Distribution Buckets")
    print(f"{'='*70}")

    for min_t, max_t in buckets:
        name = f'arith_{min_t}_{max_t}'
        bucket_names.append(name)
        val_path = os.path.join(data_dir, f'val_{name}.npy')
        val_lens = os.path.join(data_dir, f'val_{name}_lengths.npy')

        if not os.path.exists(val_path):
            print(f"  SKIP: {name} — data not found")
            continue

        print(f"\n  Bucket: {name} (terms={min_t}-{max_t})")

        bibo_metrics = evaluate_bucket(bibo_model, val_path, val_lens, device)
        all_results['bibo'][name] = bibo_metrics
        print(f"    [BiBo]  loss={bibo_metrics['loss']:.4f} | "
              f"tok_acc={bibo_metrics['token_accuracy']:.4f} | "
              f"ph2={bibo_metrics['phase2_accuracy']:.4f} | "
              f"ph3={bibo_metrics['phase3_accuracy']:.4f} | "
              f"full={bibo_metrics['full_sequence_accuracy']:.4f}")

        qwen_metrics = evaluate_bucket(qwen_model, val_path, val_lens, device)
        all_results['qwen3moe'][name] = qwen_metrics
        print(f"    [Qwen]  loss={qwen_metrics['loss']:.4f} | "
              f"tok_acc={qwen_metrics['token_accuracy']:.4f} | "
              f"ph2={qwen_metrics['phase2_accuracy']:.4f} | "
              f"ph3={qwen_metrics['phase3_accuracy']:.4f} | "
              f"full={qwen_metrics['full_sequence_accuracy']:.4f}")

    # OOD buckets
    if ood_buckets:
        print(f"\n{'='*70}")
        print(f"  Out-of-Distribution Buckets (Generalization)")
        print(f"{'='*70}")

        for min_t, max_t in ood_buckets:
            name = f'ood_{min_t}_{max_t}'
            bucket_names.append(name)
            ood_path = os.path.join(data_dir, f'ood_{name}.npy')
            ood_lens = os.path.join(data_dir, f'ood_{name}_lengths.npy')

            if not os.path.exists(ood_path):
                print(f"  SKIP: {name} — data not found")
                continue

            print(f"\n  OOD Bucket: {name} (terms={min_t}-{max_t})")

            bibo_metrics = evaluate_bucket(bibo_model, ood_path, ood_lens, device)
            all_results['bibo'][name] = bibo_metrics
            print(f"    [BiBo]  loss={bibo_metrics['loss']:.4f} | "
                  f"tok_acc={bibo_metrics['token_accuracy']:.4f} | "
                  f"ph3={bibo_metrics['phase3_accuracy']:.4f}")

            qwen_metrics = evaluate_bucket(qwen_model, ood_path, ood_lens, device)
            all_results['qwen3moe'][name] = qwen_metrics
            print(f"    [Qwen]  loss={qwen_metrics['loss']:.4f} | "
                  f"tok_acc={qwen_metrics['token_accuracy']:.4f} | "
                  f"ph3={qwen_metrics['phase3_accuracy']:.4f}")

    # Display samples from first bucket
    first_bucket = buckets[0]
    first_name = f'arith_{first_bucket[0]}_{first_bucket[1]}'
    first_path = os.path.join(data_dir, f'val_{first_name}.npy')
    first_lens = os.path.join(data_dir, f'val_{first_name}_lengths.npy')
    if os.path.exists(first_path):
        print(f"\n{'='*70}")
        print(f"  Sample Predictions ({first_name})")
        print(f"{'='*70}")
        display_samples(bibo_model, first_path, first_lens, device, "BiBo", 3)
        display_samples(qwen_model, first_path, first_lens, device, "Qwen3MoE", 3)

    # Summary table
    print_summary_table(all_results, bucket_names)

    # Save metrics
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_out = os.path.join(METRICS_DIR, 'model_analysis_arithmetic.json')
    with open(metrics_out, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Metrics saved to: {metrics_out}")
    print("\nDone.")
