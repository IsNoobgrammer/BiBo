#!/bin/bash
# Overnight ablations (user, Oct 12 2026) vs the FastEmit-0 baseline fe0-w04 (CTC 0.4, stock init, old data mix and
# look-ahead mix, seed 23, 11,250 steps, 5% warm-up, encoder dropout 0.1 / joint 0.2, Muown + AdamW wd 1e-3).
# ONE change per arm; validation at 240 ms with both heads also at blank penalty 1.0; full meeting grid after.
#   ARMS="wd0 drop0 drop01" nohup bash voice/asr/abl.sh > /home/marimo/work/asr/abl.out 2>&1 &
#   wd0     Muown weight decay 0 (AdamW keeps 1e-3)
#   drop0   every Dropout 0.0
#   drop01  every Dropout 0.1 (only the joint changes: 0.2 -> 0.1)
#   tdt     TDT instead of RNN-T as the CTC head's training partner (NeMo TDT loss; durations 0-4)
# Re-running skips finished arms and resumes an unfinished one (HF checkpoints).
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc
[ -f $A/.markers/ready ] || { echo "en3 not ready"; exit 1; }
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo
echo "BiBo $(git log --oneline -1 | cut -c1-60) | tkf $(git -C $T log --oneline -1 | cut -c1-60)"
for arm in ${ARMS:-wd0 drop0 drop01}; do
  case $arm in
    wd0) X="--muon_wd 0";; drop0) X="--dropout_all 0.0";; drop01) X="--dropout_all 0.1";;
    tdt) X="--aux tdt";; base) X="";; *) echo "unknown arm $arm"; exit 1;;
  esac
  R=$arm-fe0; FJ="--fused_joint"; [ $arm = tdt ] && FJ=""          # tkf's fused joint is RNN-T only
  if [ ! -f $A/exp_abl/$R/$R.nemo ]; then
    TKF=$T $P voice/asr/train_asr.py --project asr-abl --run $R --seed 23 --new_vocab --train $E3/train.jsonl \
      --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL --lr 5e-4 --total_hours 3750 --eval_hours 750 \
      --ctc_head glu_glu --ctc_weight 0.4 --fastemit 0 --batch_sec 1200 --lookaheads 13 6 3 1 0 -1 \
      --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20 $X \
      $FJ --fused_layer --fused_attn --fused_conv --fused_ctc --out $A/exp_abl > $A/$R.out 2>&1
    echo "ARM $R TRAIN rc $?"
  fi
  [ -f $A/exp_abl/$R/$R.nemo ] && TKF=$T $P voice/asr/ab_meeting.py --rnnt --head glu_glu --las 0 1 3 6 13 -1 \
    --bp 0 0.5 0.75 1.0 1.25 1.5 2.0 --nemo $A/exp_abl/$R/$R.nemo --dump $A/meetstudy/v2_$R.json \
    > $A/meetstudy/v2_$R.out 2>&1; echo "ARM $R MEET rc $?"
done
echo ABL_DONE
