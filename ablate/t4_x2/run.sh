#!/usr/bin/env bash
# 2x T4 (16GB each): ONE ARM PER GPU in parallel (one-arm-per-GPU pattern). Global batch 48 = micro 12 x
# grad_accum 4 -> IDENTICAL optimization to rtx_6000's batch-48 run, so the two hardware arms are
# directly comparable. bf16 (T4 has no bf16 tensor cores -> slower, but numerically identical; NEVER fp16).
# attn = sdpa: FlashAttention-4 is a Blackwell kernel and CANNOT run on Turing (sm75), so T4 uses SDPA.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # -> BiBo repo root
PY=${PY:-python}
SEED=${SEED:-0}
TOKENS=${TOKENS:-1000000000}
OUT=ablate/t4_x2/runs
# train.py evals PERIODICALLY (every 500 steps) + full final eval -> W&B curves. No separate eval step.
COMMON="--seed $SEED --tokens $TOKENS --batch 12 --grad_accum 4 --seq_len 1024 \
        --precision bf16 --attn sdpa --patches liger_norm,liger_rope,ce,moe \
        --eval_every 500 --out $OUT --wandb"

echo "=== train+eval qwen@GPU0 + bibo_min@GPU1 (parallel, seed $SEED) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m ablate.common.train --arm qwen     $COMMON &
CUDA_VISIBLE_DEVICES=1 $PY -m ablate.common.train --arm bibo_min $COMMON &
wait
echo "t4_x2 ablation done (seed $SEED). Watch bpb[hi]/bpb[en] + acc[hi]/acc[en] curves in W&B."
