"""
Model Output Analysis — Dispatcher
====================================

Reads `task` from config.yaml and runs the appropriate analyzer:
  - "sort"       → analyze_model_sorting.py
  - "arithmetic" → analyze_model_arithmetic.py

Usage:
    python misc/kaggle/multi_gpu/analyze_model.py
"""
import os
import sys
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

task = CFG['training'].get('task', 'sort')

if __name__ == '__main__':
    if task == 'arithmetic':
        print(f"  Task: arithmetic → running analyze_model_arithmetic.py")
        from analyze_model_arithmetic import main
        main()
    elif task == 'sort':
        print(f"  Task: sort → running analyze_model_sorting.py")
        from analyze_model_sorting import main
        main()
    else:
        print(f"  ERROR: Unknown task '{task}' in config.yaml. Expected 'sort' or 'arithmetic'.")
        sys.exit(1)


# ============================================================
# Model loading
# ============================================================

def load_models(device):
    """Load both models from checkpoints."""
    bibo_cfg = {k: v for k, v in CFG['bibo'].items() if k != 'device'}
    bibo_model = BiBoForCausalLM(BiBoConfig(**bibo_cfg)).to(device)
    bibo_ckpt = os.path.join(CKPT_DIR, 'bibo.pt')
    if os.path.exists(bibo_ckpt):
        bibo_model.load_state_dict(torch.load(bibo_ckpt, map_location=device))
        print(f"  [BiBo] Loaded checkpoint from {bibo_ckpt}")
    else:
        print(f"  [BiBo] WARNING: No checkpoint, using random weights")
    bibo_model.eval()

    qwen_cfg = {k: v for k, v in CFG['qwen3moe'].items() if k != 'device'}
    qwen_model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**qwen_cfg)).to(device)
    qwen_ckpt = os.path.join(CKPT_DIR, 'qwen3moe.pt')
    if os.path.exists(qwen_ckpt):
        qwen_model.load_state_dict(torch.load(qwen_ckpt, map_location=device))
        print(f"  [Qwen3MoE] Loaded checkpoint from {qwen_ckpt}")
    else:
        print(f"  [Qwen3MoE] WARNING: No checkpoint, using random weights")
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
def evaluate_batch(model, input_ids, labels, device):
    """Run model on a batch, return logits and predictions."""
    input_ids = input_ids.to(device)
    labels_dev = labels.to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    preds = logits.argmax(dim=-1)
    probs = F.softmax(logits, dim=-1)
    top1_probs = probs.gather(2, preds.unsqueeze(-1)).squeeze(-1)

    label_mask = (labels != -100).to(device)
    correct_mask = (preds == labels_dev) & label_mask

    # Loss (CCE only)
    flat_logits = logits[label_mask]
    flat_labels = labels_dev[label_mask]
    loss = F.cross_entropy(flat_logits, flat_labels).item() if flat_labels.numel() > 0 else 0.0

    return {
        'loss': loss,
        'preds': preds.cpu(),
        'probs': probs.cpu(),
        'top1_probs': top1_probs.cpu(),
        'correct_mask': correct_mask.cpu(),
        'label_mask': label_mask.cpu(),
    }


def compute_phase_metrics(eval_result, labels, raw_samples):
    """
    Compute per-phase accuracy metrics.

    Phase 2: tokens between first SEP and second SEP
    Phase 3: tokens after second SEP (final answer)
    """
    preds = eval_result['preds']
    label_mask = eval_result['label_mask']
    correct_mask = eval_result['correct_mask']
    batch_size = preds.shape[0]

    # Overall token accuracy
    total_tokens = label_mask.sum().item()
    correct_tokens = correct_mask.sum().item()
    token_acc = correct_tokens / max(total_tokens, 1)

    # Per-phase accuracy
    phase2_correct = 0
    phase2_total = 0
    phase3_correct = 0
    phase3_total = 0
    full_seq_correct = 0
    phase2_seq_correct = 0
    phase3_seq_correct = 0

    for i in range(batch_size):
        sample = raw_samples[i]
        sep_positions = [j for j, t in enumerate(sample) if t == SEP_TOKEN]
        if len(sep_positions) < 2:
            continue

        length = len(sample)
        # In shifted labels: labels[j] = sample[j+1]
        # Phase2 tokens in sample: positions sep_positions[0]+1 to sep_positions[1]-1
        # In labels: these correspond to indices sep_positions[0] to sep_positions[1]-1
        # But we also include the SEP tokens as targets
        p2_start = sep_positions[0] - 1  # label index for first SEP prediction
        p2_end = sep_positions[1] - 1    # label index for second SEP prediction
        p3_start = sep_positions[1]      # label index for phase3 start
        p3_end = min(length - 1, preds.shape[1])  # label index for phase3 end

        # Phase 2 accuracy (includes predicting SEPs)
        p2_mask = label_mask[i, p2_start:p2_end+1]
        p2_correct = correct_mask[i, p2_start:p2_end+1]
        phase2_total += p2_mask.sum().item()
        phase2_correct += p2_correct.sum().item()
        if p2_mask.sum() > 0 and p2_correct.sum() == p2_mask.sum():
            phase2_seq_correct += 1

        # Phase 3 accuracy (final answer)
        if p3_start < preds.shape[1]:
            p3_mask = label_mask[i, p3_start:p3_end]
            p3_correct = correct_mask[i, p3_start:p3_end]
            phase3_total += p3_mask.sum().item()
            phase3_correct += p3_correct.sum().item()
            if p3_mask.sum() > 0 and p3_correct.sum() == p3_mask.sum():
                phase3_seq_correct += 1

        # Full sequence correct
        sample_mask = label_mask[i]
        sample_correct = correct_mask[i]
        if sample_mask.sum() > 0 and sample_correct[sample_mask].all():
            full_seq_correct += 1

    return {
        'loss': eval_result['loss'],
        'token_accuracy': token_acc,
        'phase2_token_acc': phase2_correct / max(phase2_total, 1),
        'phase3_token_acc': phase3_correct / max(phase3_total, 1),
        'phase2_seq_acc': phase2_seq_correct / max(batch_size, 1),
        'phase3_seq_acc': phase3_seq_correct / max(batch_size, 1),
        'full_seq_acc': full_seq_correct / max(batch_size, 1),
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'batch_size': batch_size,
    }


# ============================================================
# Display predictions
# ============================================================

def display_predictions(eval_result, raw_samples, model_name, bucket_name, num_samples=3):
    """Print detailed token-by-token predictions for a few samples."""
    preds = eval_result['preds']
    probs = eval_result['probs']
    label_mask = eval_result['label_mask']

    print(f"\n  {model_name} — Predictions [{bucket_name}]")
    print(f"  {'─'*60}")

    for idx in range(min(num_samples, len(raw_samples))):
        sample = raw_samples[idx]
        phase1, phase2, phase3 = parse_phases(sample)
        if phase1 is None:
            continue

        # Get predicted tokens for target region
        mask_positions = label_mask[idx].nonzero(as_tuple=True)[0]
        pred_tokens = [preds[idx, p].item() for p in mask_positions]
        gt_tokens = []
        sep_positions = [j for j, t in enumerate(sample) if t == SEP_TOKEN]
        # Ground truth target: sample[first_sep:] (shifted by 1 in labels)
        target_start = sep_positions[0]
        for j in range(target_start, len(sample) - 1):
            if sample[j+1] != 0:  # skip padding
                gt_tokens.append(int(sample[j+1]))

        # Display
        input_str = ' '.join(token_to_str(t) for t in phase1)
        gt_p2_str = ' '.join(token_to_str(t) for t in phase2)
        gt_p3_str = ' '.join(token_to_str(t) for t in phase3)

        # Predicted phase2 and phase3
        pred_all = pred_tokens[:len(gt_tokens)] if pred_tokens else []
        # Find second SEP in predictions to split phases
        pred_sep2_idx = None
        for k, t in enumerate(pred_all):
            if t == SEP_TOKEN and k > 0:
                pred_sep2_idx = k
                break

        if pred_sep2_idx is not None:
            pred_p2 = pred_all[1:pred_sep2_idx]  # skip first SEP
            pred_p3 = pred_all[pred_sep2_idx+1:]
        else:
            pred_p2 = pred_all[1:] if pred_all else []
            pred_p3 = []

        pred_p2_str = ' '.join(token_to_str(t) for t in pred_p2)
        pred_p3_str = ' '.join(token_to_str(t) for t in pred_p3)

        # Check correctness
        p2_ok = pred_p2 == [int(t) for t in phase2]
        p3_ok = pred_p3 == [int(t) for t in phase3]
        status = "✓" if (p2_ok and p3_ok) else "✗"

        print(f"  {status} Sample {idx+1}:")
        print(f"    Input:   {input_str}")
        print(f"    GT  P2:  {gt_p2_str}")
        print(f"    Pred P2: {pred_p2_str} {'✓' if p2_ok else '✗'}")
        print(f"    GT  P3:  {gt_p3_str}")
        print(f"    Pred P3: {pred_p3_str} {'✓' if p3_ok else '✗'}")
        print()


# ============================================================
# Plotting
# ============================================================

def plot_results(all_results):
    """Generate comparison plots."""
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

    bucket_names = list(all_results['bibo'].keys())
    train_names = [f"{b[0]}-{b[1]}" for b in TRAIN_BUCKETS]
    ood_names = [f"{b[0]}-{b[1]}" for b in OOD_BUCKETS]

    # --- Plot 1: Phase accuracies comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(bucket_names))
    width = 0.35

    # Token accuracy
    ax = axes[0]
    bibo_vals = [all_results['bibo'][b]['token_accuracy'] for b in bucket_names]
    qwen_vals = [all_results['qwen3moe'][b]['token_accuracy'] for b in bucket_names]
    ax.bar(x - width/2, bibo_vals, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax.bar(x + width/2, qwen_vals, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    # Highlight OOD
    for i, name in enumerate(bucket_names):
        if name in ood_names:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Token Accuracy')
    ax.set_title('Overall Token Accuracy', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Phase 2 seq accuracy
    ax = axes[1]
    bibo_vals = [all_results['bibo'][b]['phase2_seq_acc'] for b in bucket_names]
    qwen_vals = [all_results['qwen3moe'][b]['phase2_seq_acc'] for b in bucket_names]
    ax.bar(x - width/2, bibo_vals, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax.bar(x + width/2, qwen_vals, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    for i, name in enumerate(bucket_names):
        if name in ood_names:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sequence Accuracy')
    ax.set_title('Phase 2 Accuracy\n(precedence resolution)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Phase 3 seq accuracy
    ax = axes[2]
    bibo_vals = [all_results['bibo'][b]['phase3_seq_acc'] for b in bucket_names]
    qwen_vals = [all_results['qwen3moe'][b]['phase3_seq_acc'] for b in bucket_names]
    ax.bar(x - width/2, bibo_vals, width, color=BIBO_COLOR, alpha=0.8, label='BiBo')
    ax.bar(x + width/2, qwen_vals, width, color=QWEN_COLOR, alpha=0.8, label='Qwen3MoE')
    for i, name in enumerate(bucket_names):
        if name in ood_names:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sequence Accuracy')
    ax.set_title('Phase 3 Accuracy\n(final answer correct)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.suptitle('Arithmetic Task — BiBo vs Qwen3MoE\n(red = OOD, not in training)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'arithmetic_quality_comparison.png'))
    plt.close()
    print(f"  ✓ arithmetic_quality_comparison.png")

    # --- Plot 2: Loss comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bibo_losses = [all_results['bibo'][b]['loss'] for b in bucket_names]
    qwen_losses = [all_results['qwen3moe'][b]['loss'] for b in bucket_names]
    ax.plot(range(len(bucket_names)), bibo_losses, 'o-', color=BIBO_COLOR, linewidth=2, markersize=8, label='BiBo')
    ax.plot(range(len(bucket_names)), qwen_losses, 's-', color=QWEN_COLOR, linewidth=2, markersize=8, label='Qwen3MoE')
    for i, name in enumerate(bucket_names):
        if name in ood_names:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color='red')
    ax.set_xticks(range(len(bucket_names)))
    ax.set_xticklabels(bucket_names, rotation=45, ha='right')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Loss by Difficulty Bucket (red = OOD)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'arithmetic_loss_comparison.png'))
    plt.close()
    print(f"  ✓ arithmetic_loss_comparison.png")


# ============================================================
# Summary table
# ============================================================

def print_summary_table(all_results):
    """Print a clean comparison table."""
    bucket_names = list(all_results['bibo'].keys())
    train_names = [f"{b[0]}-{b[1]}" for b in TRAIN_BUCKETS]

    print(f"\n{'='*100}")
    print(f"  SUMMARY — Arithmetic Task: BiBo vs Qwen3MoE")
    print(f"{'='*100}")
    print(f"  {'Bucket':<10} {'Type':<5} │ {'BiBo Loss':<10} {'TokAcc':<8} {'P2Seq':<7} {'P3Seq':<7} {'Full':<7}"
          f"│ {'Qwen Loss':<10} {'TokAcc':<8} {'P2Seq':<7} {'P3Seq':<7} {'Full':<7} │ {'Winner'}")
    print(f"  {'─'*95}")

    bibo_wins = 0
    qwen_wins = 0

    for name in bucket_names:
        b = all_results['bibo'][name]
        q = all_results['qwen3moe'][name]
        bucket_type = "ID" if name in train_names else "OOD"

        if b['full_seq_acc'] > q['full_seq_acc']:
            winner = "BiBo"
            bibo_wins += 1
        elif q['full_seq_acc'] > b['full_seq_acc']:
            winner = "Qwen"
            qwen_wins += 1
        else:
            winner = "Tie"

        print(f"  {name:<10} {bucket_type:<5} │ "
              f"{b['loss']:<10.4f} {b['token_accuracy']:<8.4f} {b['phase2_seq_acc']:<7.3f} "
              f"{b['phase3_seq_acc']:<7.3f} {b['full_seq_acc']:<7.3f} │ "
              f"{q['loss']:<10.4f} {q['token_accuracy']:<8.4f} {q['phase2_seq_acc']:<7.3f} "
              f"{q['phase3_seq_acc']:<7.3f} {q['full_seq_acc']:<7.3f} │ {winner}")

    print(f"  {'─'*95}")
    print(f"  ID = in-distribution (trained on) | OOD = out-of-distribution (generalization)")
    print(f"  Score: BiBo {bibo_wins} — Qwen {qwen_wins}")
    print(f"{'='*100}")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "="*70)
    print("  MODEL OUTPUT ANALYSIS — BiBo vs Qwen3MoE")
    print("  Task: Arithmetic (multi-phase chain-of-thought)")
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

    for bucket in ALL_BUCKETS:
        min_t, max_t = bucket
        bucket_name = f"{min_t}-{max_t}"
        is_ood = bucket in OOD_BUCKETS

        print(f"\n{'─'*70}")
        print(f"  Testing bucket [{min_t}, {max_t}] {'(OOD)' if is_ood else '(ID)'}")
        print(f"{'─'*70}")

        # Generate test data
        input_ids, labels, raw_samples = generate_arithmetic_batch(
            min_t, max_t, NUM_TEST_SAMPLES, rng
        )
        if input_ids is None:
            print(f"  SKIPPED — could not generate samples")
            continue

        actual_batch = len(raw_samples)
        print(f"  Generated {actual_batch} samples (seq_len range: "
              f"{min(len(s) for s in raw_samples)}-{max(len(s) for s in raw_samples)})")

        # Evaluate in sub-batches
        bibo_metrics_list = []
        qwen_metrics_list = []

        for start in range(0, actual_batch, BATCH_SIZE):
            end = min(start + BATCH_SIZE, actual_batch)
            batch_input = input_ids[start:end]
            batch_labels = labels[start:end]
            batch_samples = raw_samples[start:end]

            bibo_eval = evaluate_batch(bibo_model, batch_input, batch_labels, device)
            bibo_m = compute_phase_metrics(bibo_eval, batch_labels, batch_samples)
            bibo_metrics_list.append(bibo_m)

            qwen_eval = evaluate_batch(qwen_model, batch_input, batch_labels, device)
            qwen_m = compute_phase_metrics(qwen_eval, batch_labels, batch_samples)
            qwen_metrics_list.append(qwen_m)

        # Aggregate metrics across sub-batches
        def aggregate_metrics(metrics_list):
            agg = {}
            total_samples = sum(m['batch_size'] for m in metrics_list)
            total_tokens = sum(m['total_tokens'] for m in metrics_list)
            correct_tokens = sum(m['correct_tokens'] for m in metrics_list)
            agg['loss'] = np.mean([m['loss'] for m in metrics_list])
            agg['token_accuracy'] = correct_tokens / max(total_tokens, 1)
            agg['phase2_token_acc'] = np.mean([m['phase2_token_acc'] for m in metrics_list])
            agg['phase2_seq_acc'] = np.mean([m['phase2_seq_acc'] for m in metrics_list])
            agg['phase3_token_acc'] = np.mean([m['phase3_token_acc'] for m in metrics_list])
            agg['phase3_seq_acc'] = np.mean([m['phase3_seq_acc'] for m in metrics_list])
            agg['full_seq_acc'] = np.mean([m['full_seq_acc'] for m in metrics_list])
            agg['total_tokens'] = total_tokens
            agg['correct_tokens'] = correct_tokens
            agg['batch_size'] = total_samples
            return agg

        bibo_agg = aggregate_metrics(bibo_metrics_list)
        qwen_agg = aggregate_metrics(qwen_metrics_list)

        all_results['bibo'][bucket_name] = bibo_agg
        all_results['qwen3moe'][bucket_name] = qwen_agg

        print(f"  [BiBo]    loss={bibo_agg['loss']:.4f} | tok={bibo_agg['token_accuracy']:.4f} | "
              f"P2={bibo_agg['phase2_seq_acc']:.3f} | P3={bibo_agg['phase3_seq_acc']:.3f} | "
              f"full={bibo_agg['full_seq_acc']:.3f}")
        print(f"  [Qwen]    loss={qwen_agg['loss']:.4f} | tok={qwen_agg['token_accuracy']:.4f} | "
              f"P2={qwen_agg['phase2_seq_acc']:.3f} | P3={qwen_agg['phase3_seq_acc']:.3f} | "
              f"full={qwen_agg['full_seq_acc']:.3f}")

        # Show sample predictions for first sub-batch
        first_batch_input = input_ids[:BATCH_SIZE]
        first_batch_labels = labels[:BATCH_SIZE]
        first_batch_samples = raw_samples[:BATCH_SIZE]
        bibo_eval_display = evaluate_batch(bibo_model, first_batch_input, first_batch_labels, device)
        qwen_eval_display = evaluate_batch(qwen_model, first_batch_input, first_batch_labels, device)
        display_predictions(bibo_eval_display, first_batch_samples, "BiBo", bucket_name, NUM_DISPLAY_SAMPLES)
        display_predictions(qwen_eval_display, first_batch_samples, "Qwen3MoE", bucket_name, NUM_DISPLAY_SAMPLES)

    # Summary
    print_summary_table(all_results)

    # Plots
    print("\nGenerating plots...")
    plot_results(all_results)

    # Save metrics
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_out = os.path.join(METRICS_DIR, 'arithmetic_analysis.json')
    with open(metrics_out, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\n  Metrics saved to: {metrics_out}")
    print(f"  Plots saved to: {PLOTS_DIR}/")
    print("\nDone.")


if __name__ == '__main__':
    main()
