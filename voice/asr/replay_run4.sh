#!/bin/bash
# run4 divergence forensics: replay from its last good checkpoint (step 9000, val 19.16 %) to step 10500 with run4's
# exact schedule (cosine over 17625 steps), one factor changed per arm, per-module grad norms logged every 50 steps
# (W&B bibo-asr-ab, diag/*). In run4 the grad-norm median went 49 -> 132 -> 176 -> 720 -> 3445 from step 9000.
#   bash voice/asr/replay_run4.sh                ARMS="same lr05" picks arms
set -uo pipefail
W=/home/marimo/work; A=$W/asr; R=$A/run2; E=$A/eval_sets; P=/home/marimo/asrenv/bin/python
export TKF=${TKF:-$W/tkf_ctc}
cd $W/BiBo && git log --oneline -1
CK=$(ls $A/exp/run4/step=9000-*.ckpt)
VALS="$(ls $R/val_*.jsonl | grep -v -e '/val_en.jsonl' -e '/val_hi.jsonl') $E/fleurs_en.jsonl $E/fleurs_hi.jsonl"
BASE="--out $A/exp_replay --project bibo-asr-ab --no_hf --seed 23 --train $R/train.jsonl --tok $A/run1/tok/tokenizer_spe_bpe_v4096
  --val $VALS --lr 1e-3 --warmup 1000 --batch_sec 1200 --total_hours 5875 --eval_hours 600
  --lookaheads 13 6 3 1 0 --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc
  --ckpt $CK --stop_step 10500 --grad_diag 50"
MIX="--lookahead_probs 0:0.05 1:0.20 3:0.40 6:0.15 13:0.20"
for arm in ${ARMS:-same lr05 fe005 unimix}; do
  case $arm in
    same)   X="$MIX --fastemit 0.01" ;;
    lr05)   X="$MIX --fastemit 0.01 --lr_scale 0.5" ;;
    fe005)  X="$MIX --fastemit 0.005" ;;
    unimix) X="--lookahead_probs 0:0.2 1:0.2 3:0.2 6:0.2 13:0.2 --fastemit 0.01" ;;
  esac
  rm -rf $A/exp_replay/rp-$arm
  $P voice/asr/train_asr.py --run rp-$arm $BASE $X > $A/replay_$arm.log 2>&1
  echo "ARM_END $arm $?"
done
echo REPLAY_DONE
