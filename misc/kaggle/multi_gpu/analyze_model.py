"""
Comprehensive Model Output Analysis — BiBo vs Qwen3MoE
=======================================================

Evaluates MODEL QUALITY (not routing behavior — see analyze_router.py for that).

Tests across sequence lengths: 8, 32, 64, 128, 256, 512
Batch size: 8 (fixed)

Analyses:
1. Loss per sequence length (cross-entropy on sorted portion)
2. Token-level accuracy (exact match on sorted output)
3. Full-sequence accuracy (entire sorted output correct)
4. Top-1 predicted token + probability for each position (sample display)
5. Position-wise accuracy curve (which positions are hardest?)
6. Confidence calibration (is high confidence = correct?)
7. Error analysis (what tokens get confused with what?)
8. Length generalization (trained on 64/128/256, tested on 8/32/512)

Usage:
    python misc/kaggle/multi_gpu/analyze_model.py

Requires trained checkpoints in misc/kaggle/multi_gpu/checkpoints/
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import yaml
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

# ============================================================
# Config
# ============================================================
TEST_SEQ_LENS = [8, 32, 64, 96, 128, 192]
BATCH_SIZE = 8
VOCAB_SIZE = CFG['training']['vocab_size']  # 512
SEP_TOKEN = VOCAB_SIZE - 1  # 511
SEED = CFG['training']['seed']
NUM_DISPLAY_SAMPLES = 3  # how many samples to show detailed predictions for


# ============================================================
# Data generation (on-the-fly, no disk dependency)
# ============================================================

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
    full_len = 2 * seq_len + 1  # unsorted + SEP + sorted
    
    all_input_ids = []
    all_labels = []
    all_unsorted = []
    all_sorted = []
    
    for _ in range(batch_size):
        tokens = rng.integers(0, SEP_TOKEN, size=seq_len)
        sorted_tokens = np.sort(tokens)
        full_seq = np.concatenate([tokens, [SEP_TOKEN], sorted_tokens]).astype(np.int64)
        
        # input_ids = full_seq[:-1], labels = full_seq[1:]
        input_ids = full_seq[:-1]
        labels = full_seq[1:].copy()
        # Mask unsorted portion (first seq_len positions in labels)
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


# ============================================================
# Model loading
# ============================================================

def load_models(device):
    """Load both models from checkpoints."""
    # BiBo
    bibo_cfg = {k: v for k, v in CFG['bibo'].items() if k != 'device'}
    bibo_model = BiBoForCausalLM(BiBoConfig(**bibo_cfg)).to(device)
    bibo_ckpt = os.path.join(CKPT_DIR, 'bibo.pt')
    if os.path.exists(bibo_ckpt):
        bibo_model.load_state_dict(torch.load(bibo_ckpt, map_location=device))
        print(f"  [BiBo] Loaded checkpoint from {bibo_ckpt}")
    else:
        print(f"  [BiBo] WARNING: No checkpoint found at {bibo_ckpt}, using random weights")
    bibo_model.eval()
    
    # Qwen3MoE
    qwen_cfg = {k: v for k, v in CFG['qwen3moe'].items() if k != 'device'}
    qwen_model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen_cfg)).to(device)
    qwen_ckpt = os.path.join(CKPT_DIR, 'qwen3moe.pt')
    if os.path.exists(qwen_ckpt):
        qwen_model.load_state_dict(torch.load(qwen_ckpt, map_location=device))
        print(f"  [Qwen3MoE] Loaded checkpoint from {qwen_ckpt}")
    else:
        print(f"  [Qwen3MoE] WARNING: No checkpoint found at {qwen_ckpt}, using random weights")
    qwen_model.eval()
    
    bibo_params = sum(p.numel() for p in bibo_model.parameters())
    qwen_params = sum(p.numel() for p in qwen_model.parameters())
    print(f"  [BiBo] Params: {bibo_params:,}")
    print(f"  [Qwen3MoE] Params: {qwen_params:,}")
    
    return bibo_model, qwen_model


# ============================================================
# Core evaluation
# ============================================================

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
        correct_mask: [batch, seq] bool — correct predictions (only where labels != -100)
        label_mask: [batch, seq] bool — positions with valid labels
    """
    input_ids = input_ids.to(device)
    labels_dev = labels.to(device)
    
    outputs = model(input_ids=input_ids, labels=labels_dev)
    logits = outputs.logits  # [batch, seq, vocab]
    
    probs = F.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)  # [batch, seq]
    top1_probs = probs.gather(2, preds.unsqueeze(-1)).squeeze(-1)  # [batch, seq]
    
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
    
    # Token-level accuracy
    total_tokens = mask.sum().item()
    correct_tokens = correct.sum().item()
    token_acc = correct_tokens / max(total_tokens, 1)
    
    # Full-sequence accuracy (ALL sorted tokens correct)
    batch_size = mask.shape[0]
    seq_correct = []
    for i in range(batch_size):
        sample_mask = mask[i]
        sample_correct = correct[i]
        # All valid positions must be correct
        if sample_mask.sum() > 0:
            seq_correct.append(sample_correct[sample_mask].all().item())
        else:
            seq_correct.append(False)
    full_seq_acc = sum(seq_correct) / max(len(seq_correct), 1)
    
    # Position-wise accuracy (per position in sorted output)
    # labels mask starts after seq_len positions
    pos_correct = []
    pos_total = []
    sorted_len = mask.sum(dim=0)  # per-position count across batch
    for pos in range(mask.shape[1]):
        if mask[:, pos].sum() > 0:
            pos_correct.append(correct[:, pos].sum().item())
            pos_total.append(mask[:, pos].sum().item())
    pos_acc = [c / max(t, 1) for c, t in zip(pos_correct, pos_total)]
    
    # Confidence stats
    top1_probs = eval_result['top1_probs']
    valid_probs = top1_probs[mask].numpy()
    correct_probs = top1_probs[correct].numpy()
    wrong_mask = mask & ~correct
    wrong_probs = top1_probs[wrong_mask].numpy() if wrong_mask.sum() > 0 else np.array([])
    
    # Mean confidence when correct vs wrong
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


# ============================================================
# Detailed prediction display
# ============================================================

def display_predictions(eval_result, labels, raw_unsorted, raw_sorted, seq_len, model_name, num_samples=3):
    """
    Print detailed token-by-token predictions for a few samples.
    Shows: position, ground truth, predicted token, probability, correct/wrong.
    Only shows the sorted output portion (where model is actually predicting).
    """
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
        
        # Ground truth sorted tokens
        gt_sorted = raw_sorted[sample_idx]
        unsorted = raw_unsorted[sample_idx]
        
        # Check if entire sequence is correct
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
        
        # Show predictions for sorted portion
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
            
            # Top-3 predictions with probabilities
            top3_probs, top3_indices = sample_probs[pos_idx].topk(3)
            top3_str = " | ".join(
                f"{top3_indices[k].item()}({top3_probs[k].item():.3f})"
                for k in range(3)
            )
            
            # Only print first/last few + all errors (keep output manageable)
            should_print = (i < 5) or (i >= len(valid_positions) - 3) or (not is_correct and errors_shown < 10)
            if not is_correct:
                errors_shown += 1
            
            if should_print:
                print(f"  {i:<5} {gt_token:<6} {pred_token:<6} {pred_prob:<8.4f} {status_str:<8} {top3_str}")
            elif i == 5:
                print(f"  {'...':<5} (showing first 5, last 3, and up to 10 errors)")
        
        # Summary for this sample
        n_correct = sum(
            1 for i, pos in enumerate(valid_positions)
            if i < len(gt_sorted) and sample_preds[pos.item()].item() == gt_sorted[i]
        )
        n_total = min(len(valid_positions), len(gt_sorted))
        print(f"  → {n_correct}/{n_total} tokens correct ({100*n_correct/max(n_total,1):.1f}%)")


# ============================================================
# Error analysis
# ============================================================

def analyze_errors(eval_result, labels, raw_sorted, seq_len, model_name):
    """
    Analyze prediction errors:
    - Most common confusion pairs (predicted X when answer was Y)
    - Error distribution by position (early vs late in sorted output)
    - Error magnitude (how far off is the predicted token from ground truth?)
    """
    preds = eval_result['preds']
    mask = eval_result['label_mask']
    batch_size = preds.shape[0]
    
    confusion_pairs = defaultdict(int)  # (gt, pred) -> count
    position_errors = defaultdict(int)  # position -> error count
    position_total = defaultdict(int)
    error_magnitudes = []  # |pred - gt| for wrong predictions
    
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
    
    # Top confusion pairs
    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]
    
    print(f"\n  {model_name} — Error Analysis (seq_len={seq_len})")
    print(f"  {'─'*50}")
    
    if len(sorted_confusions) == 0:
        print(f"  No errors! Perfect predictions.")
        return {}
    
    print(f"  Top confusion pairs (GT → Pred : count):")
    for (gt, pred), count in sorted_confusions[:10]:
        print(f"    {gt:>5} → {pred:<5} : {count}x  (off by {abs(pred-gt)})")
    
    # Error magnitude stats
    if error_magnitudes:
        mag = np.array(error_magnitudes)
        print(f"\n  Error magnitude (|pred - gt|):")
        print(f"    Mean: {mag.mean():.1f} | Median: {np.median(mag):.1f} | "
              f"Max: {mag.max()} | Std: {mag.std():.1f}")
        print(f"    Off by ≤1: {(mag <= 1).sum()}/{len(mag)} ({100*(mag<=1).mean():.1f}%)")
        print(f"    Off by ≤5: {(mag <= 5).sum()}/{len(mag)} ({100*(mag<=5).mean():.1f}%)")
        print(f"    Off by ≤10: {(mag <= 10).sum()}/{len(mag)} ({100*(mag<=10).mean():.1f}%)")
    
    # Position error rate
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


# ============================================================
# Plotting
# ============================================================

def plot_results(all_results):
    """Generate comparison plots from collected results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style='whitegrid', palette='muted')
        plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})
    except ImportError:
        print("  [WARN] matplotlib/seaborn not available, skipping plots")
        return
    
    BIBO_COLOR = '#2196F3'
    QWEN_COLOR = '#FF5722'
    
    seq_lens = sorted(all_results['bibo'].keys())
    
    # --- Plot 1: Loss vs Sequence Length ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    bibo_losses = [all_results['bibo'][sl]['loss'] for sl in seq_lens]
    qwen_losses = [all_results['qwen3moe'][sl]['loss'] for sl in seq_lens]
    
    ax = axes[0]
    ax.plot(seq_lens, bibo_losses, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(seq_lens, qwen_losses, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Loss vs Sequence Length', fontweight='bold')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)
    
    # --- Plot 2: Token Accuracy vs Sequence Length ---
    bibo_acc = [all_results['bibo'][sl]['token_accuracy'] for sl in seq_lens]
    qwen_acc = [all_results['qwen3moe'][sl]['token_accuracy'] for sl in seq_lens]
    
    ax = axes[1]
    ax.plot(seq_lens, bibo_acc, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(seq_lens, qwen_acc, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Token Accuracy vs Sequence Length', fontweight='bold')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)
    ax.set_ylim(0, 1.05)
    
    # --- Plot 3: Full Sequence Accuracy ---
    bibo_full = [all_results['bibo'][sl]['full_sequence_accuracy'] for sl in seq_lens]
    qwen_full = [all_results['qwen3moe'][sl]['full_sequence_accuracy'] for sl in seq_lens]
    
    ax = axes[2]
    ax.plot(seq_lens, bibo_full, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(seq_lens, qwen_full, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Full Sequence Accuracy')
    ax.set_title('Full Sequence Accuracy\n(entire sorted output correct)', fontweight='bold')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)
    ax.set_ylim(0, 1.05)
    
    plt.suptitle('Model Quality — BiBo vs Qwen3MoE (Sorting Task)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_quality_comparison.png'))
    plt.close()
    print(f"  ✓ model_quality_comparison.png")
    
    # --- Plot 4: Confidence Calibration ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    bibo_conf_correct = [all_results['bibo'][sl]['mean_confidence_correct'] for sl in seq_lens]
    bibo_conf_wrong = [all_results['bibo'][sl]['mean_confidence_wrong'] for sl in seq_lens]
    qwen_conf_correct = [all_results['qwen3moe'][sl]['mean_confidence_correct'] for sl in seq_lens]
    qwen_conf_wrong = [all_results['qwen3moe'][sl]['mean_confidence_wrong'] for sl in seq_lens]
    
    ax = axes[0]
    ax.plot(seq_lens, bibo_conf_correct, 'o-', color=BIBO_COLOR, linewidth=2, label='BiBo (correct)')
    ax.plot(seq_lens, bibo_conf_wrong, 'o--', color=BIBO_COLOR, linewidth=1.5, alpha=0.6, label='BiBo (wrong)')
    ax.plot(seq_lens, qwen_conf_correct, 's-', color=QWEN_COLOR, linewidth=2, label='Qwen (correct)')
    ax.plot(seq_lens, qwen_conf_wrong, 's--', color=QWEN_COLOR, linewidth=1.5, alpha=0.6, label='Qwen (wrong)')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Mean Top-1 Probability')
    ax.set_title('Confidence: Correct vs Wrong Predictions', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_lens)
    
    # --- Plot 5: Position-wise accuracy for select seq_lens ---
    ax = axes[1]
    for sl in [64, 96, 128, 192]:
        if sl in all_results['bibo'] and all_results['bibo'][sl]['position_accuracy']:
            pos_acc = all_results['bibo'][sl]['position_accuracy']
            positions = np.linspace(0, 1, len(pos_acc))  # normalize to [0,1]
            ax.plot(positions, pos_acc, '-', linewidth=1.5, label=f'BiBo seq={sl}')
    for sl in [64, 96, 128, 192]:
        if sl in all_results['qwen3moe'] and all_results['qwen3moe'][sl]['position_accuracy']:
            pos_acc = all_results['qwen3moe'][sl]['position_accuracy']
            positions = np.linspace(0, 1, len(pos_acc))
            ax.plot(positions, pos_acc, '--', linewidth=1.5, label=f'Qwen seq={sl}')
    ax.set_xlabel('Relative Position in Sorted Output')
    ax.set_ylabel('Accuracy')
    ax.set_title('Position-wise Accuracy\n(0=first sorted token, 1=last)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_confidence_position.png'))
    plt.close()
    print(f"  ✓ model_confidence_position.png")
    
    # --- Plot 6: Length generalization (trained vs untrained lengths) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    trained_lens = [2, 8, 32, 64, 128, 256]
    
    x = np.arange(len(seq_lens))
    width = 0.35
    
    bibo_bars = [all_results['bibo'][sl]['token_accuracy'] for sl in seq_lens]
    qwen_bars = [all_results['qwen3moe'][sl]['token_accuracy'] for sl in seq_lens]
    
    bars1 = ax.bar(x - width/2, bibo_bars, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    bars2 = ax.bar(x + width/2, qwen_bars, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    
    # Highlight trained vs untrained
    for i, sl in enumerate(seq_lens):
        if sl not in trained_lens:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Length Generalization\n(red background = NOT in training data)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_length_generalization.png'))
    plt.close()
    print(f"  ✓ model_length_generalization.png")


# ============================================================
# Summary table
# ============================================================

def print_summary_table(all_results):
    """Print a clean comparison table."""
    seq_lens = sorted(all_results['bibo'].keys())
    
    print(f"\n{'='*90}")
    print(f"  SUMMARY — Model Quality Comparison (BiBo vs Qwen3MoE)")
    print(f"{'='*90}")
    print(f"  {'SeqLen':<8} {'│ BiBo Loss':<12} {'Acc':<8} {'FullSeq':<9} "
          f"{'│ Qwen Loss':<12} {'Acc':<8} {'FullSeq':<9} {'│ Winner'}")
    print(f"  {'─'*8} {'─'*11} {'─'*7} {'─'*8} "
          f"{'─'*11} {'─'*7} {'─'*8} {'─'*8}")
    
    bibo_wins = 0
    qwen_wins = 0
    
    for sl in seq_lens:
        b = all_results['bibo'][sl]
        q = all_results['qwen3moe'][sl]
        
        # Winner by token accuracy
        if b['token_accuracy'] > q['token_accuracy']:
            winner = "BiBo"
            bibo_wins += 1
        elif q['token_accuracy'] > b['token_accuracy']:
            winner = "Qwen"
            qwen_wins += 1
        else:
            winner = "Tie"
        
        trained = " *" if sl not in [64, 128, 256] else ""
        print(f"  {sl:<8}{trained}│ {b['loss']:<10.4f} {b['token_accuracy']:<8.4f} {b['full_sequence_accuracy']:<9.4f}"
              f"│ {q['loss']:<10.4f} {q['token_accuracy']:<8.4f} {q['full_sequence_accuracy']:<9.4f}"
              f"│ {winner}")
    
    print(f"  {'─'*85}")
    print(f"  * = sequence length NOT in training data (generalization test)")
    print(f"  Score: BiBo {bibo_wins} — Qwen {qwen_wins}")
    print(f"{'='*90}")
    
    # Confidence analysis
    print(f"\n  Confidence Analysis:")
    print(f"  {'SeqLen':<8} {'│ BiBo Conf(✓)':<15} {'Conf(✗)':<10} {'Gap':<8}"
          f"{'│ Qwen Conf(✓)':<15} {'Conf(✗)':<10} {'Gap':<8}")
    print(f"  {'─'*75}")
    for sl in seq_lens:
        b = all_results['bibo'][sl]
        q = all_results['qwen3moe'][sl]
        b_gap = b['mean_confidence_correct'] - b['mean_confidence_wrong']
        q_gap = q['mean_confidence_correct'] - q['mean_confidence_wrong']
        print(f"  {sl:<8}│ {b['mean_confidence_correct']:<13.4f} {b['mean_confidence_wrong']:<10.4f} {b_gap:<8.4f}"
              f"│ {q['mean_confidence_correct']:<13.4f} {q['mean_confidence_wrong']:<10.4f} {q_gap:<8.4f}")
    print(f"  (Higher gap = better calibrated — model is confident when right, uncertain when wrong)")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "="*70)
    print("  MODEL OUTPUT ANALYSIS — BiBo vs Qwen3MoE")
    print("  Task: Sorting | Batch: 8 | Seq Lens: 8, 32, 64, 128, 256, 512")
    print("="*70)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    
    # Load models
    print("\nLoading models...")
    bibo_model, qwen_model = load_models(device)
    
    # RNG for reproducible test data
    rng = np.random.default_rng(SEED + 9999)  # different from train/val seeds
    
    # Collect results
    all_results = {'bibo': {}, 'qwen3moe': {}}
    all_errors = {'bibo': {}, 'qwen3moe': {}}
    
    for seq_len in TEST_SEQ_LENS:
        print(f"\n{'─'*70}")
        print(f"  Testing seq_len = {seq_len}")
        print(f"{'─'*70}")
        
        # Generate test batch
        input_ids, labels, raw_unsorted, raw_sorted = generate_sorting_batch(
            seq_len, BATCH_SIZE, rng
        )
        
        # Evaluate BiBo
        bibo_eval = evaluate_model(bibo_model, input_ids, labels, device)
        bibo_metrics = compute_metrics(bibo_eval, labels)
        all_results['bibo'][seq_len] = bibo_metrics
        
        print(f"  [BiBo]    loss={bibo_metrics['loss']:.4f} | "
              f"token_acc={bibo_metrics['token_accuracy']:.4f} | "
              f"full_seq_acc={bibo_metrics['full_sequence_accuracy']:.4f}")
        
        # Evaluate Qwen3MoE
        qwen_eval = evaluate_model(qwen_model, input_ids, labels, device)
        qwen_metrics = compute_metrics(qwen_eval, labels)
        all_results['qwen3moe'][seq_len] = qwen_metrics
        
        print(f"  [Qwen]    loss={qwen_metrics['loss']:.4f} | "
              f"token_acc={qwen_metrics['token_accuracy']:.4f} | "
              f"full_seq_acc={qwen_metrics['full_sequence_accuracy']:.4f}")
        
        # Detailed predictions (top-1 token + probability)
        display_predictions(bibo_eval, labels, raw_unsorted, raw_sorted, seq_len, "BiBo", NUM_DISPLAY_SAMPLES)
        display_predictions(qwen_eval, labels, raw_unsorted, raw_sorted, seq_len, "Qwen3MoE", NUM_DISPLAY_SAMPLES)
        
        # Error analysis
        bibo_errors = analyze_errors(bibo_eval, labels, raw_sorted, seq_len, "BiBo")
        qwen_errors = analyze_errors(qwen_eval, labels, raw_sorted, seq_len, "Qwen3MoE")
        all_errors['bibo'][seq_len] = bibo_errors
        all_errors['qwen3moe'][seq_len] = qwen_errors
    
    # Summary
    print_summary_table(all_results)
    
    # Plots
    print("\nGenerating plots...")
    plot_results(all_results)
    
    # Save metrics to JSON
    # Convert numpy types for JSON serialization
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
    
    metrics_out = os.path.join(METRICS_DIR, 'model_analysis.json')
    with open(metrics_out, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Metrics saved to: {metrics_out}")
    print(f"  Plots saved to: {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == '__main__':
    main()
