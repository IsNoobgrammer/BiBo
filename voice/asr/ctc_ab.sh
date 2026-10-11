#!/bin/bash
# CTC weight A/B, clean (Oct 11 2026): the old ep2-ctc04 resumed mid-run under the broken resume (frozen lr), so it
# is rerun here next to a CTC 0.6 arm on the SAME code (exact resume, 5% warm-up = 562 steps), seed 23, 2 epochs
# (11,250 steps = 2 x 5,625) from NVIDIA's stock pretrained streaming FastConformer (train_asr's default --init; user,
# Oct 11: not from run5, whose CTC/RNN-T balance was already trained in). INIT=<.nemo> overrides. Arms run one after
# the other; re-running the script resumes an unfinished arm and skips a finished one.
#   nohup bash voice/asr/ctc_ab.sh > /home/marimo/work/asr/ctc_ab.out 2>&1 &
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python
[ -f $A/.markers/ready ] || { echo "en3 not ready (run setup_en3.sh)"; exit 1; }
INIT=${INIT:-stt_en_fastconformer_hybrid_large_streaming_multi}
case "$INIT" in *.nemo) [ -f "$INIT" ] || { echo "missing init $INIT"; exit 1; };; esac
TAG=${TAG:-stock}
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo
echo "BiBo $(git log --oneline -1 | cut -c1-60) | tkf $(git -C /home/marimo/work/tkf_ctc log --oneline -1 | cut -c1-60)"
for W in ${ARMS:-0.4 0.6}; do
  R=ctcab-$TAG-w${W/./}
  [ -f $A/exp_ctcab_$TAG/$R/$R.nemo ] && { echo "ARM $R already done"; continue; }
  TKF=/home/marimo/work/tkf_ctc $P voice/asr/train_asr.py --project asr-ctc-weight --run $R --init $INIT \
    --seed 23 --new_vocab --train $E3/train.jsonl --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL \
    --lr 5e-4 --total_hours 3750 --eval_hours 750 \
    --ctc_head glu_glu --ctc_weight $W --fastemit 0.01 --batch_sec 1200 \
    --lookaheads 13 6 3 1 0 -1 --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20 \
    --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $A/exp_ctcab_$TAG > $A/$R.out 2>&1
  rc=$?; echo "ARM $R TRAIN rc $rc"
  [ $rc -eq 0 ] && $P voice/asr/ab_meeting.py --rnnt --head glu_glu --las 3 13 -1 --bp 0 0.5 0.75 1.0 1.25 1.5 \
    --nemo $A/exp_ctcab_$TAG/$R/$R.nemo > $A/meet_$R.out 2>&1; echo "ARM $R MEET rc $?"
done
echo "CTCAB_DONE"
