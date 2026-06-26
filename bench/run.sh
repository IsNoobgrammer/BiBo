#!/bin/bash
# BiBo Benchmark — Unified Entry Point
#
# Usage:
#   bash bench/run.sh bench/configs/bibo.yaml [extra args...]
#   bash bench/run.sh bench/configs/qwen3moe.yaml --total_steps 3000
#
# Dual-GPU comparison (1 GPU per model, simultaneous):
#   CUDA_VISIBLE_DEVICES=0 bash bench/run.sh bench/configs/bibo.yaml &
#   CUDA_VISIBLE_DEVICES=1 bash bench/run.sh bench/configs/qwen3moe.yaml &
#   wait

set -e

CONFIG="${1:-bench/configs/bibo.yaml}"
shift 2>/dev/null || true

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[run.sh] Config: $CONFIG"
echo "[run.sh] Extra args: $@"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Install deps
pip install -q hf_transfer wandb datasets transformers pyyaml bitsandbytes einops liger-kernel

# WandB login. Respect either WANDB_API_KEY or a prior `wandb login` (~/.netrc).
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null
elif grep -qs "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "[run.sh] Using existing wandb login (~/.netrc)"
else
    echo "[run.sh] WANDB not configured (no WANDB_API_KEY, no prior 'wandb login') — WandB disabled"
    export WANDB_MODE=disabled
fi

# Detect GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "[run.sh] Detected $GPU_COUNT GPU(s)"

# Launch
if [ "$GPU_COUNT" -gt 1 ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[run.sh] Multi-GPU: torchrun --nproc_per_node=$GPU_COUNT"
    torchrun --nproc_per_node="$GPU_COUNT" bench/train.py \
        --config "$CONFIG" "$@"
else
    echo "[run.sh] Single-GPU"
    python bench/train.py --config "$CONFIG" "$@"
fi
