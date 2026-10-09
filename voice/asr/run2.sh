#!/bin/bash
# run2: run1's exact schedule (9,600 steps of 1,200 s batches = ~3,000 audio hours, 1,000 linear warm-up steps = 10.4 %,
# cosine to 1e-5) on the 1,200 h mix (asr-english + asr-hindi, every source x1), lr 1e-3 (one-pass LR sweep:
# 2.5e-4 51.5 / 5e-4 27.5 / 1e-3 21.8 / 2e-3 21.0 val WER), all tkf kernels incl. the deterministic CTC loss.
# run2 v1 (name "run2") replayed its first ~280 h every eval segment (train_asr iterator bug, fixed 7b6533b): invalid.
# From the NVIDIA base model with run1's tokenizer. Eval every 300 h; checkpoints synced to the HF ckpt repo.
#   bash voice/asr/run2.sh          (needs sweep_run2.sh's data: run2/train.jsonl + val sets)
set -euo pipefail
W=/home/marimo/work; A=$W/asr; R=$A/run2; E=$A/eval_sets; P=/home/marimo/asrenv/bin/python
export TKF=${TKF:-$W/tkf_ctc}                                  # triton-kernel-fused with kernels/sm120/ctc_loss.py
cd $W/BiBo && git log --oneline -1 && git -C $TKF log --oneline -1
TOK=$A/run1/tok/tokenizer_spe_bpe_v4096   # the joint en/hi tokenizer every run uses (trained for run1; run1 deleted from HF)
[ -f $TOK/tokenizer.model ] || { mkdir -p $TOK && $P -c "from huggingface_hub import hf_hub_download as h; import shutil
for f in ('tokenizer.model', 'tokenizer.vocab', 'vocab.txt'): shutil.copy(h('fhai50032/bibo-asr-ckpt', 'run2v2/tok/' + f), '$TOK/' + f)"; }
VALS="$(ls $R/val_*.jsonl | grep -v -e '/val_en.jsonl' -e '/val_hi.jsonl') $E/fleurs_en.jsonl $E/fleurs_hi.jsonl"
$P voice/asr/train_asr.py --run ${RUN:-run2v2} --out $A/exp --seed 23 --train $R/train.jsonl \
  --tok $TOK --val $VALS --lr 1e-3 --warmup 1000 --batch_sec 1200 \
  --total_hours 3200 --eval_hours 300 --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc "$@"
echo RUN2_DONE
