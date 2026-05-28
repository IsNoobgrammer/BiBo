#!/bin/bash
# ================================================================
# BiBo vs Qwen3MoE — Parallel Training on 2×T4
#
# GPU 0: BiBo    → WandB run "bibo"
# GPU 1: Qwen3MoE → WandB run "qwen"
#
# Both in project "bibo-bench" — compare loss curves side-by-side.
#
# Triton fused RMSNorm is ON by default (8-9x faster norm ops).
# To disable: bash bench/run_comparison.sh --no_triton
#
# Usage:
#   bash bench/run_comparison.sh
#   bash bench/run_comparison.sh --no_triton
# ================================================================

set -e
cd "$(dirname "$0")/.."

STEPS=3000
BS=12
GRAD_ACCUM=4
WARMUP=300
LR=7e-4

# Pass through extra args (e.g. --no_triton)
EXTRA_ARGS="$@"

echo "============================================================"
echo "  BiBo vs Qwen3MoE — 2×T4 Parallel Training"
echo "============================================================"
echo "  Steps: $STEPS | Batch: $BS | Grad Accum: $GRAD_ACCUM"
echo "  LR: $LR | Warmup: $WARMUP"
echo "  Extra args: $EXTRA_ARGS"
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
    $EXTRA_ARGS \
    2>&1 | sed 's/^/[GPU0] /' &
PID_BIBO=$!

# Wait for dataset to be downloaded/cached before starting Qwen
# Prevents HuggingFace cache race condition
echo "[launcher] Waiting 30s for dataset cache to settle..."
sleep 30

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
    $EXTRA_ARGS \
    2>&1 | sed 's/^/[GPU1] /' &
PID_QWEN=$!

echo "[launcher] BiBo PID: $PID_BIBO | Qwen3MoE PID: $PID_QWEN"
echo "[launcher] Waiting for both to finish..."

wait $PID_BIBO
BIBO_EXIT=$?
echo "[launcher] BiBo done (exit=$BIBO_EXIT)"

wait $PID_QWEN
QWEN_EXIT=$?
echo "[launcher] Qwen3MoE done (exit=$QWEN_EXIT)"

if [ $BIBO_EXIT -ne 0 ] || [ $QWEN_EXIT -ne 0 ]; then
    echo "[launcher] WARNING: One or both processes failed!"
    echo "[launcher]   BiBo exit: $BIBO_EXIT"
    echo "[launcher]   Qwen exit: $QWEN_EXIT"
fi

echo ""
echo "============================================================"
echo "  Done! Compare at: https://wandb.ai/ablations-tinycompany-ai/bibo-bench"
echo "============================================================"
