#!/bin/bash
# ================================================================
# BiBo vs Qwen3MoE — Parallel Training on 2×T4
#
# GPU 0: BiBo    → WandB run "bibo"
# GPU 1: Qwen3MoE → WandB run "qwen"
#
# Both in project "bibo-bench" — compare loss curves side-by-side.
#
# Usage:
#   bash bench/run_comparison.sh
# ================================================================

set -e
cd "$(dirname "$0")/.."

STEPS=2000
BS=4
GRAD_ACCUM=4
WARMUP=200
LR=3e-4

echo "============================================================"
echo "  BiBo vs Qwen3MoE — 2×T4 Parallel Training"
echo "============================================================"
echo "  Steps: $STEPS | Batch: $BS | Grad Accum: $GRAD_ACCUM"
echo "  LR: $LR | Warmup: $WARMUP"
echo "  WandB: bibo-bench/bibo vs bibo-bench/qwen"
echo "============================================================"

# ── Launch BiBo on GPU 0 ──────────────────────────────────────
echo "[launcher] Starting BiBo on cuda:0..."
CUDA_VISIBLE_DEVICES=0 python bench/train.py \
    --batch_size $BS \
    --grad_accum $GRAD_ACCUM \
    --total_steps $STEPS \
    --warmup_steps $WARMUP \
    --lr $LR \
    --eval_every 200 \
    --log_every 10 \
    --seq_len 1024 \
    --wandb_project bibo-bench \
    --wandb_name bibo \
    &
PID_BIBO=$!

# ── Launch Qwen3MoE on GPU 1 ──────────────────────────────────
echo "[launcher] Starting Qwen3MoE on cuda:1..."
CUDA_VISIBLE_DEVICES=1 python bench/train_qwen.py \
    --batch_size $BS \
    --grad_accum $GRAD_ACCUM \
    --total_steps $STEPS \
    --warmup_steps $WARMUP \
    --lr $LR \
    --eval_every 200 \
    --log_every 10 \
    --seq_len 1024 \
    --wandb_project bibo-bench \
    --wandb_name qwen \
    &
PID_QWEN=$!

echo "[launcher] BiBo PID: $PID_BIBO | Qwen3MoE PID: $PID_QWEN"
echo "[launcher] Waiting for both to finish..."

wait $PID_BIBO
echo "[launcher] BiBo done (exit=$?)"

wait $PID_QWEN
echo "[launcher] Qwen3MoE done (exit=$?)"

echo ""
echo "============================================================"
echo "  Done! Compare at: https://wandb.ai/ablations-tinycompany-ai/bibo-bench"
echo "============================================================"
