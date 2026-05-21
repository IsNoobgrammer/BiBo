"""
Sorting task metrics — data generation, evaluation, and error analysis.
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

__all__ = [
    'generate_sorting_batch', 'evaluate_model', 'compute_metrics',
    'display_predictions', 'analyze_errors',
]


def _get_config():
    """Lazy-load config to avoid import-time side effects."""
    from model_utils import CFG
    return CFG


def generate_sorting_batch(seq_len, batch_size, rng):
    """
    Generate a batch of sorting task samples.
    Format: [unsorted (seq_len)] [SEP] [sorted (seq_len)]

    Returns:
        input_ids: [batch_size, 2*seq_len]  (full_seq minus last token)
        labels: [batch_size, 2*seq_len]  (shifted, masked unsorted portion)
        raw_unsorted: [batch_size, seq_len]  (original unsorted tokens)
        raw_sorted: [batch_size, seq_len]  (ground truth sorted tokens)
    """
    cfg = _get_config()
    vocab_size = cfg['training']['vocab_size']
    sep_token = vocab_size - 1

    all_input_ids = []
    all_labels = []
    all_unsorted = []
    all_sorted = []

    for _ in range(batch_size):
        tokens = rng.integers(0, sep_token, size=seq_len)
        sorted_tokens = np.sort(tokens)
        full_seq = np.concatenate([tokens, [sep_token], sorted_tokens]).astype(np.int64)

        input_ids = full_seq[:-1]
        labels = full_seq[1:].copy()
        labels[:seq_len] = -100

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_unsorted.append(tokens)
        all_sorted.append(sorted_tokens)

    return (
        torch.tensor(np.array(all_input_ids), dtype=torch.long),
        torch.tensor(np.array(all_labels), dtype=torch.long),
        np.array(all_unsorted),
        np.array(all_sorted),
    )


@torch.no_grad()
def evaluate_model(model, input_ids, labels, device):
    """
    Run model and compute detailed metrics.

    Returns dict with:
        loss: scalar cross-entropy on sorted portion
        logits: [batch, seq, vocab] raw logits
        probs: [batch, seq, vocab] softmax probabilities
        preds: [batch, seq] argmax predictions
        top1_probs: [batch, seq] probability of top-1 prediction
        correct_mask: [batch, seq] bool
        label_mask: [batch, seq] bool
    """
    input_ids = input_ids.to(device)
    labels_dev = labels.to(device)

    outputs = model(input_ids=input_ids, labels=labels_dev)
    logits = outputs.logits

    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    top1_probs = probs.gather(2, preds.unsqueeze(-1)).squeeze(-1)

    label_mask = (labels != -100).to(device)
    correct_mask = (preds == labels_dev) & label_mask

    return {
        'loss': outputs.loss.item(),
        'logits': logits.cpu(),
        'probs': probs.cpu(),
        'preds': preds.cpu(),
        'top1_probs': top1_probs.cpu(),
        'correct_mask': correct_mask.cpu(),
        'label_mask': label_mask.cpu(),
    }


def compute_metrics(eval_result, labels):
    """Compute aggregate metrics from evaluation result."""
    mask = eval_result['label_mask']
    correct = eval_result['correct_mask']

    total_tokens = mask.sum().item()
    correct_tokens = correct.sum().item()
    token_acc = correct_tokens / max(total_tokens, 1)

    batch_size = mask.shape[0]
    seq_correct = []
    for i in range(batch_size):
        sample_mask = mask[i]
        sample_correct = correct[i]
        if sample_mask.sum() > 0:
            seq_correct.append(sample_correct[sample_mask].all().item())
        else:
            seq_correct.append(False)
    full_seq_acc = sum(seq_correct) / max(len(seq_correct), 1)

    pos_correct = []
    pos_total = []
    for pos in range(mask.shape[1]):
        if mask[:, pos].sum() > 0:
            pos_correct.append(correct[:, pos].sum().item())
            pos_total.append(mask[:, pos].sum().item())
    pos_acc = [c / max(t, 1) for c, t in zip(pos_correct, pos_total)]

    top1_probs = eval_result['top1_probs']
    valid_probs = top1_probs[mask].numpy()
    correct_probs = top1_probs[correct].numpy()
    wrong_mask = mask & ~correct
    wrong_probs = top1_probs[wrong_mask].numpy() if wrong_mask.sum() > 0 else np.array([])

    mean_conf_correct = float(correct_probs.mean()) if len(correct_probs) > 0 else 0.0
    mean_conf_wrong = float(wrong_probs.mean()) if len(wrong_probs) > 0 else 0.0

    return {
        'loss': eval_result['loss'],
        'token_accuracy': token_acc,
        'full_sequence_accuracy': full_seq_acc,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'position_accuracy': pos_acc,
        'mean_confidence_correct': mean_conf_correct,
        'mean_confidence_wrong': mean_conf_wrong,
        'mean_confidence_all': float(valid_probs.mean()) if len(valid_probs) > 0 else 0.0,
        'per_sample_correct': seq_correct,
    }


def display_predictions(eval_result, labels, raw_unsorted, raw_sorted, seq_len, model_name, num_samples=3):
    """Print detailed token-by-token predictions for a few samples."""
    print(f"\n{'='*70}")
    print(f"  {model_name} — Detailed Predictions (seq_len={seq_len})")
    print(f"{'='*70}")

    preds = eval_result['preds']
    probs = eval_result['probs']
    mask = eval_result['label_mask']
    batch_size = preds.shape[0]

    for sample_idx in range(min(num_samples, batch_size)):
        sample_mask = mask[sample_idx]
        sample_preds = preds[sample_idx]
        sample_probs = probs[sample_idx]

        gt_sorted = raw_sorted[sample_idx]
        unsorted = raw_unsorted[sample_idx]

        valid_positions = sample_mask.nonzero(as_tuple=True)[0]
        all_correct = all(
            sample_preds[pos].item() == gt_sorted[i]
            for i, pos in enumerate(valid_positions)
            if i < len(gt_sorted)
        )

        status = "PERFECT" if all_correct else "HAS ERRORS"
        print(f"\n  Sample {sample_idx + 1} [{status}]")
        print(f"  Input (unsorted): {unsorted[:20].tolist()}{'...' if len(unsorted) > 20 else ''}")
        print(f"  Target (sorted):  {gt_sorted[:20].tolist()}{'...' if len(gt_sorted) > 20 else ''}")
        print(f"  {'Pos':<5} {'GT':<6} {'Pred':<6} {'Prob':<8} {'Status':<8} {'Top-3 Predictions'}")
        print(f"  {'-'*65}")

        errors_shown = 0
        for i, pos in enumerate(valid_positions):
            if i >= len(gt_sorted):
                break
            pos_idx = pos.item()
            gt_token = gt_sorted[i]
            pred_token = sample_preds[pos_idx].item()
            pred_prob = sample_probs[pos_idx, pred_token].item()

            is_correct = (pred_token == gt_token)
            status_str = "  OK" if is_correct else "  WRONG"

            top3_probs, top3_indices = sample_probs[pos_idx].topk(3)
            top3_str = " | ".join(
                f"{top3_indices[k].item()}({top3_probs[k].item():.3f})"
                for k in range(3)
            )

            should_print = (i < 5) or (i >= len(valid_positions) - 3) or (not is_correct and errors_shown < 10)
            if not is_correct:
                errors_shown += 1

            if should_print:
                print(f"  {i:<5} {gt_token:<6} {pred_token:<6} {pred_prob:<8.4f} {status_str:<8} {top3_str}")
            elif i == 5:
                print(f"  {'...':<5} (showing first 5, last 3, and up to 10 errors)")

        n_correct = sum(
            1 for i, pos in enumerate(valid_positions)
            if i < len(gt_sorted) and sample_preds[pos.item()].item() == gt_sorted[i]
        )
        n_total = min(len(valid_positions), len(gt_sorted))
        print(f"  -> {n_correct}/{n_total} tokens correct ({100*n_correct/max(n_total,1):.1f}%)")


def analyze_errors(eval_result, labels, raw_sorted, seq_len, model_name):
    """Analyze prediction errors: confusion pairs, position errors, error magnitude."""
    preds = eval_result['preds']
    mask = eval_result['label_mask']
    batch_size = preds.shape[0]

    confusion_pairs = defaultdict(int)
    position_errors = defaultdict(int)
    position_total = defaultdict(int)
    error_magnitudes = []

    for sample_idx in range(batch_size):
        sample_mask = mask[sample_idx]
        valid_positions = sample_mask.nonzero(as_tuple=True)[0]
        gt_sorted = raw_sorted[sample_idx]

        for i, pos in enumerate(valid_positions):
            if i >= len(gt_sorted):
                break
            pos_idx = pos.item()
            gt_token = int(gt_sorted[i])
            pred_token = preds[sample_idx, pos_idx].item()
            position_total[i] += 1

            if pred_token != gt_token:
                confusion_pairs[(gt_token, pred_token)] += 1
                position_errors[i] += 1
                error_magnitudes.append(abs(pred_token - gt_token))

    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]

    print(f"\n  {model_name} — Error Analysis (seq_len={seq_len})")
    print(f"  {'_'*50}")

    if len(sorted_confusions) == 0:
        print(f"  No errors! Perfect predictions.")
        return {}

    print(f"  Top confusion pairs (GT -> Pred : count):")
    for (gt, pred), count in sorted_confusions[:10]:
        print(f"    {gt:>5} -> {pred:<5} : {count}x  (off by {abs(pred-gt)})")

    if error_magnitudes:
        mag = np.array(error_magnitudes)
        print(f"\n  Error magnitude (|pred - gt|):")
        print(f"    Mean: {mag.mean():.1f} | Median: {np.median(mag):.1f} | "
              f"Max: {mag.max()} | Std: {mag.std():.1f}")
        print(f"    Off by <=1: {(mag <= 1).sum()}/{len(mag)} ({100*(mag<=1).mean():.1f}%)")
        print(f"    Off by <=5: {(mag <= 5).sum()}/{len(mag)} ({100*(mag<=5).mean():.1f}%)")
        print(f"    Off by <=10: {(mag <= 10).sum()}/{len(mag)} ({100*(mag<=10).mean():.1f}%)")

    if position_errors:
        early_errors = sum(position_errors.get(i, 0) for i in range(min(seq_len//4, len(position_total))))
        early_total = sum(position_total.get(i, 0) for i in range(min(seq_len//4, len(position_total))))
        late_errors = sum(position_errors.get(i, 0) for i in range(max(0, seq_len*3//4), seq_len))
        late_total = sum(position_total.get(i, 0) for i in range(max(0, seq_len*3//4), seq_len))

        print(f"\n  Error rate by position:")
        print(f"    First quarter: {early_errors}/{max(early_total,1)} "
              f"({100*early_errors/max(early_total,1):.1f}%)")
        print(f"    Last quarter:  {late_errors}/{max(late_total,1)} "
              f"({100*late_errors/max(late_total,1):.1f}%)")

    return {
        'top_confusions': sorted_confusions[:15],
        'error_magnitudes': error_magnitudes,
        'position_errors': dict(position_errors),
        'position_total': dict(position_total),
    }
