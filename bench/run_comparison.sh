#!/bin/bash
# ================================================================
# BiBo vs Qwen3MoE — Parallel Training on 2×T4
#
# GPU 0: BiBo baseline
# GPU 1: Qwen3MoE baseline
#
# Both log to same WandB project (bibo-bench) with name "qwen-vs-bibo"
# so you can compare loss curves side-by-side on one dashboard.
#
# Usage:
#   bash bench/run_comparison.sh
# ================================================================

set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  BiBo vs Qwen3MoE — 2×T4 Parallel Training"
echo "============================================================"
echo "  Steps: 2000 | Batch: 4 | Grad Accum: 4 | Eff Batch: 16"
echo "  WandB Project: bibo-bench"
echo "============================================================"

# ── Launch BiBo on GPU 0 ──────────────────────────────────────
echo "[launcher] Starting BiBo on cuda:0..."
CUDA_VISIBLE_DEVICES=0 python bench/train.py \
    --batch_size 4 \
    --grad_accum 4 \
    --total_steps 2000 \
    --warmup_steps 200 \
    --lr 3e-4 \
    --eval_every 200 \
    --log_every 10 \
    --seq_len 1024 \
    --wandb_project bibo-bench \
    --wandb_name qwen-vs-bibo \
    --wandb_notes "BiBo vs Qwen3MoE comparison — 2000 steps, 2×T4" \
    &
PID_BIBO=$!

# ── Launch Qwen3MoE on GPU 1 ──────────────────────────────────
echo "[launcher] Starting Qwen3MoE on cuda:1..."
CUDA_VISIBLE_DEVICES=1 python bench/train_qwen.py \
    --batch_size 4 \
    --grad_accum 4 \
    --total_steps 2000 \
    --warmup_steps 200 \
    --lr 1e-4 \
    --eval_every 200 \
    --log_every 10 \
    --seq_len 1024 \
    --wandb_project bibo-bench \
    --wandb_name qwen-vs-bibo \
    --wandb_notes "BiBo vs Qwen3MoE comparison — 2000 steps, 2×T4" \
    &
PID_QWEN=$!

echo "[launcher] BiBo PID: $PID_BIBO"
echo "[launcher] Qwen3MoE PID: $PID_QWEN"
echo "[launcher] Waiting for both to finish..."

# ── Wait for both ──────────────────────────────────────────────
wait $PID_BIBO
EXIT_BIBO=$?
echo "[launcher] BiBo finished (exit=$EXIT_BIBO)"

wait $PID_QWEN
EXIT_QWEN=$?
echo "[launcher] Qwen3MoE finished (exit=$EXIT_QWEN)"

echo ""
echo "============================================================"
echo "  Both runs complete!"
echo "  BiBo exit: $EXIT_BIBO | Qwen3MoE exit: $EXIT_QWEN"
echo "  Check WandB: https://wandb.ai/ablations-tinycompany-ai/bibo-bench"
echo "============================================================"
