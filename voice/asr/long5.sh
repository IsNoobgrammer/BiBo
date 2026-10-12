#!/bin/bash
# 5-epoch runs (user, Oct 12 2026): does a longer run still hurt the meeting clips (mixla4ep: validation 21.35 -> 20.67,
# meetings 240 ms 23.60 -> 25.98) once the schedule is gentler? Recipe = the FastEmit-0 baseline (fe0-w04: CTC 0.4,
# stock init, original data + look-ahead mixes, dropout / wd as pretrained), defaults since Oct 12: FastEmit 0,
# validation also at blank penalty 1.0 (checkpoints select on CTC-bp). 28,125 steps = 5 x 5,625; validation, an HF
# checkpoint and a local step checkpoint every epoch; then the meeting eval of every epoch.
#   RUN=ep5-w10  WU=0.10 LR=5e-4   nohup bash voice/asr/long5.sh > /home/marimo/work/asr/long5_w10.out 2>&1 &
#   RUN=ep5-lr25 WU=0.05 LR=2.5e-4 nohup bash voice/asr/long5.sh > /home/marimo/work/asr/long5_lr25.out 2>&1 &
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc
[ -f $A/.markers/ready ] || { echo "en3 not ready"; exit 1; }
R=${RUN:?}; WU=${WU:?}; LR=${LR:?}; X=${EXTRA:-}; O=$A/exp_long5        # EXTRA: more train_asr flags, e.g. "--dropout_all 0.0"
TR=${TR:-$E3/train.jsonl}                                       # TR: training manifest (e.g. vmix/train.jsonl)
LAP=${LAP:-"0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20"}       # LAP: look-ahead mix (80 ms frames: 0 1 3 6 13 full)
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo
mkdir -p $A/meetstudy
echo "BiBo $(git log --oneline -1 | cut -c1-60) | tkf $(git -C $T log --oneline -1 | cut -c1-60) | $R train $TR la-mix [$LAP] warm-up $WU lr $LR extra [$X]"
if [ ! -f $O/$R/$R.nemo ]; then
  TKF=$T $P voice/asr/train_asr.py --project asr-long5 --run $R --seed 23 --new_vocab --train $TR \
    --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL --lr $LR --warmup_frac $WU --total_hours 9375 --eval_hours 1875 \
    --ctc_head glu_glu --ctc_weight 0.4 --fastemit 0 --batch_sec 1200 --lookaheads 13 6 3 1 0 -1 \
    --lookahead_probs $LAP --save_steps 5625 11250 16875 22500 \
    --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc $X --out $O > $A/$R.out 2>&1
  echo "TRAIN $R rc $?"
fi
[ -f $O/$R/$R.nemo ] || { echo "no final model"; exit 1; }
$P voice/asr/restore_steps.py --run $R --dir $O/$R --steps 5625 11250 16875 22500   # a dead box lost them
for s in 5625 11250 16875 22500 final; do
  CK=""; [ $s != final ] && { [ -f $O/$R/step$s.ckpt ] && CK="--ckpt $O/$R/step$s.ckpt" || { echo "no step$s.ckpt"; continue; }; }
  TKF=$T $P voice/asr/ab_meeting.py --head glu_glu --las 0 1 3 6 13 -1 --bp 0 0.5 0.75 1.0 1.25 1.5 2.0 \
    --nemo $O/$R/$R.nemo $CK --dump $A/meetstudy/${R}_$s.json > $A/meetstudy/${R}_$s.out 2>&1; echo "MEET $R $s rc $?"
done
echo LONG5_DONE
