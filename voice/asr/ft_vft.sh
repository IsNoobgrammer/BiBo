#!/bin/bash
# Verbatim fine-tune (user, Oct 12 2026): does a short low-lr pass over the verbatim set (build_vft.py: AMI / NOTSOFAR /
# Earnings22 / phone + acoustic copies, Svarah x3; 604 h) undo the deletions that more epochs on the cleaned-transcript
# mix added on the meeting clips (ep5-w10: meetings best at epoch 3, worse at 5)? Starts from ep5-w10's final weights
# (--init_weights), fresh Muown + cosine, peak lr $LR (default 1e-4), 2 data epochs, validation every epoch, then the
# meeting eval of the epoch-1 checkpoint and the final model.
#   RUN=ft-vft-w10 INIT=/home/marimo/work/asr/exp_long5/ep5-w10/ep5-w10.nemo nohup bash voice/asr/ft_vft.sh > $A/ft_vft.out 2>&1 &
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc
R=${RUN:-ft-vft-w10}; INIT=${INIT:-$A/exp_long5/ep5-w10/ep5-w10.nemo}; LR=${LR:-1e-4}; EP=${EP:-2}; X=${EXTRA:-}; O=$A/exp_ft
TR=$A/vft/train.jsonl
[ -f $TR ] || { echo "no $TR (run build_vft.py)"; exit 1; }
[ -f $INIT ] || { echo "no $INIT"; exit 1; }
H=$($P -c "import json; print(round(sum(json.loads(l)['duration'] for l in open('$TR')) / 3600, 1))")
TOT=$($P -c "print(round($H * $EP, 1))"); S1=$($P -c "print(int($H * 3600 / 1200 * 0.99))")
SAVE=$($P -c "print(' '.join(str($S1 * k) for k in range(1, $EP)))")      # a checkpoint at every epoch boundary
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo; mkdir -p $A/meetstudy
echo "BiBo $(git log --oneline -1 | cut -c1-60) | $R init $INIT lr $LR epochs $EP: $H h x $EP = $TOT h, saves at [$SAVE], extra [$X]"
if [ ! -f $O/$R/$R.nemo ]; then
  TKF=$T $P voice/asr/train_asr.py --project asr-ft --run $R --seed 23 --new_vocab --train $TR \
    --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL --init_weights $INIT --lr $LR --warmup_frac 0.05 \
    --total_hours $TOT --eval_hours 0 --save_steps $SAVE $X \
    --ctc_head glu_glu --ctc_weight 0.4 --fastemit 0 --batch_sec 1200 --lookaheads 13 6 3 1 0 -1 \
    --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20 \
    --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $O > $A/$R.out 2>&1
  echo "TRAIN $R rc $?"
fi
[ -f $O/$R/$R.nemo ] || { echo "no final model"; exit 1; }
for s in $SAVE final; do
  CK=""; [ $s != final ] && { [ -f $O/$R/step$s.ckpt ] && CK="--ckpt $O/$R/step$s.ckpt" || { echo "no step$s.ckpt"; continue; }; }
  TKF=$T $P voice/asr/ab_meeting.py --head glu_glu --las 0 1 3 6 13 -1 --bp 0 0.5 0.75 1.0 1.25 1.5 2.0 \
    --nemo $O/$R/$R.nemo $CK --dump $A/meetstudy/${R}_$s.json > $A/meetstudy/${R}_$s.out 2>&1; echo "MEET $R $s rc $?"
done
echo FTVFT_DONE
