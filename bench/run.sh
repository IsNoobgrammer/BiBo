#!/bin/bash
# BiBo Benchmark — Kaggle Entry Point
# Usage: bash bench/run.sh [args...]
set -e

# ── Environment ────────────────────────────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── Install dependencies ──────────────────────────────────────
echo "[run.sh] Installing dependencies..."
pip install -q hf_transfer wandb datasets transformers

# Install BiBo from repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
pip install -e .

# ── Login to WandB (if WANDB_API_KEY is set) ──────────────────
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "[run.sh] WandB logged in"
else
    echo "[run.sh] WARNING: WANDB_API_KEY not set, WandB logging disabled"
    export WANDB_MODE=disabled
fi

# ── Detect GPU count ──────────────────────────────────────────
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[run.sh] Detected $GPU_COUNT GPU(s)"

# ── Launch training ───────────────────────────────────────────
cd "$REPO_ROOT"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "[run.sh] Multi-GPU mode: torchrun --nproc_per_node=$GPU_COUNT"
    torchrun --nproc_per_node="$GPU_COUNT" bench/train.py \
        --batch_size 16 \
        --total_steps 50000 \
        --warmup_steps 1000 \
        --eval_every 500 \
        --sample_every 1000 \
        --lr 3e-4 \
        "$@"
else
    echo "[run.sh] Single-GPU mode"
    python bench/train.py \
        --batch_size 32 \
        --total_steps 50000 \
        --warmup_steps 1000 \
        --eval_every 500 \
        --sample_every 1000 \
        --lr 3e-4 \
        "$@"
fi
