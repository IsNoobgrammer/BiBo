#!/usr/bin/env bash
# RTX 6000 (single GPU, 48/96GB): both arms sequentially at global batch 40, 1B tokens, bf16, then eval.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # -> BiBo repo root
PY=${PY:-.venv/Scripts/python.exe}             # override PY=python on linux box
SEED=${SEED:-0}
TOKENS=${TOKENS:-1000000000}
OUT=ablate/rtx_6000/runs
# train.py evals PERIODICALLY (every 500 steps) + a full final eval, all logged to W&B as curves.
COMMON="--seed $SEED --tokens $TOKENS --batch 48 --grad_accum 1 --seq_len 1024 \
        --precision bf16 --attn flash_attention_4 --patches liger_norm,liger_rope,ce,moe \
        --eval_every 500 --out $OUT --wandb"

for arm in qwen bibo_min; do
  echo "=== train+eval $arm (seed $SEED) ==="
  $PY -m ablate.common.train --arm "$arm" $COMMON
done
echo "rtx_6000 ablation done (seed $SEED). Watch bpb[hi]/bpb[en] + acc[hi]/acc[en] curves in W&B."
