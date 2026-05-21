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
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from model_utils import CFG, load_models, BASE_DIR
from plot_utils import PLOTS_DIR, METRICS_DIR
from sorting_metrics import (
    generate_sorting_batch, evaluate_model, compute_metrics,
    display_predictions, analyze_errors,
)
from sorting_plots import plot_results

TEST_SEQ_LENS = [8, 32, 64, 96, 128, 192]
BATCH_SIZE = 8
SEED = CFG['training']['seed']
NUM_DISPLAY_SAMPLES = 3


# ============================================================
# Summary table
# ============================================================

def print_summary_table(all_results):
    """Print a clean comparison table."""
    seq_lens = sorted(all_results['bibo'].keys())

    print(f"\n{'='*90}")
    print(f"  SUMMARY — Model Quality Comparison (BiBo vs Qwen3MoE)")
    print(f"{'='*90}")
    print(f"  {'SeqLen':<8} {'| BiBo Loss':<12} {'Acc':<8} {'FullSeq':<9} "
          f"{'| Qwen Loss':<12} {'Acc':<8} {'FullSeq':<9} {'| Winner'}")
    print(f"  {'-'*8} {'-'*11} {'-'*7} {'-'*8} "
          f"{'-'*11} {'-'*7} {'-'*8} {'-'*8}")

    bibo_wins = 0
    qwen_wins = 0

    for sl in seq_lens:
        b = all_results['bibo'][sl]
        q = all_results['qwen3moe'][sl]

        if b['token_accuracy'] > q['token_accuracy']:
            winner = "BiBo"
            bibo_wins += 1
        elif q['token_accuracy'] > b['token_accuracy']:
            winner = "Qwen"
            qwen_wins += 1
        else:
            winner = "Tie"

        trained = " *" if sl not in [64, 128, 256] else ""
        print(f"  {sl:<8}{trained}| {b['loss']:<10.4f} {b['token_accuracy']:<8.4f} {b['full_sequence_accuracy']:<9.4f}"
              f"| {q['loss']:<10.4f} {q['token_accuracy']:<8.4f} {q['full_sequence_accuracy']:<9.4f}"
              f"| {winner}")

    print(f"  {'-'*85}")
    print(f"  * = sequence length NOT in training data (generalization test)")
    print(f"  Score: BiBo {bibo_wins} — Qwen {qwen_wins}")
    print(f"{'='*90}")

    # Confidence analysis
    print(f"\n  Confidence Analysis:")
    print(f"  {'SeqLen':<8} {'| BiBo Conf(V)':<15} {'Conf(X)':<10} {'Gap':<8}"
          f"{'| Qwen Conf(V)':<15} {'Conf(X)':<10} {'Gap':<8}")
    print(f"  {'-'*75}")
    for sl in seq_lens:
        b = all_results['bibo'][sl]
        q = all_results['qwen3moe'][sl]
        b_gap = b['mean_confidence_correct'] - b['mean_confidence_wrong']
        q_gap = q['mean_confidence_correct'] - q['mean_confidence_wrong']
        print(f"  {sl:<8}| {b['mean_confidence_correct']:<13.4f} {b['mean_confidence_wrong']:<10.4f} {b_gap:<8.4f}"
              f"| {q['mean_confidence_correct']:<13.4f} {q['mean_confidence_wrong']:<10.4f} {q_gap:<8.4f}")
    print(f"  (Higher gap = better calibrated — model is confident when right, uncertain when wrong)")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "="*70)
    print("  MODEL OUTPUT ANALYSIS — BiBo vs Qwen3MoE")
    print("  Task: Sorting | Batch: 8 | Seq Lens: 8, 32, 64, 128, 256, 512")
    print("="*70)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")

    print("\nLoading models...")
    bibo_model, qwen_model = load_models(device)

    rng = np.random.default_rng(SEED + 9999)

    all_results = {'bibo': {}, 'qwen3moe': {}}
    all_errors = {'bibo': {}, 'qwen3moe': {}}

    for seq_len in TEST_SEQ_LENS:
        print(f"\n{'-'*70}")
        print(f"  Testing seq_len = {seq_len}")
        print(f"{'-'*70}")

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

        display_predictions(bibo_eval, labels, raw_unsorted, raw_sorted, seq_len, "BiBo", NUM_DISPLAY_SAMPLES)
        display_predictions(qwen_eval, labels, raw_unsorted, raw_sorted, seq_len, "Qwen3MoE", NUM_DISPLAY_SAMPLES)

        bibo_errors = analyze_errors(bibo_eval, labels, raw_sorted, seq_len, "BiBo")
        qwen_errors = analyze_errors(qwen_eval, labels, raw_sorted, seq_len, "Qwen3MoE")
        all_errors['bibo'][seq_len] = bibo_errors
        all_errors['qwen3moe'][seq_len] = qwen_errors

    print_summary_table(all_results)

    print("\nGenerating plots...")
    plot_results(all_results)

    # Save metrics to JSON
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
