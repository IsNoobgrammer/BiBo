#!/bin/bash
# Longer training on the hard-set mix + look-ahead recipe (user, Oct 12 2026): does more training help the OOD meeting
# clips, or only validation (run6's pattern)? Same as mixla-w04 except the length: EPOCHS x 5,625 steps (default 4 ->
# 22,500), cosine to the end, 5% warm-up, validation every epoch (so every epoch's checkpoint is also in HF history),
# then a meeting eval of EVERY epoch's checkpoint (v2 scoring, all look-aheads, blank penalty grid).
#   EPOCHS=4 nohup bash voice/asr/long_mix.sh > /home/marimo/work/asr/long_mix.out 2>&1 &
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc
[ -f $A/.markers/ready ] || { echo "en3 not ready (run setup_en3.sh)"; exit 1; }
N=${EPOCHS:-4}; R=mixla${N}ep-w04; O=$A/exp_long
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
set -f
cd /home/marimo/work/BiBo
echo "BiBo $(git log --oneline -1 | cut -c1-60) | tkf $(git -C $T log --oneline -1 | cut -c1-60) | $R"
SAVE=$(seq -s ' ' 5625 5625 $((5625 * (N - 1))))
if [ ! -f $O/$R/$R.nemo ]; then
  TKF=$T $P voice/asr/train_asr.py --project asr-mix --run $R --seed 23 --new_vocab --train $E3/train.jsonl \
    --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL --lr 5e-4 --total_hours $((1875 * N)) --eval_hours 1875 \
    --ctc_head glu_glu --ctc_weight 0.4 --fastemit 0.01 --batch_sec 1200 --lookaheads 13 6 3 1 0 -1 \
    --lookahead_probs 0:0.10 1:0.23 3:0.32 6:0.15 13:0 full:0.20 \
    --source_weight notsofar:3 multispk_ami:2 spotify*:4 svarah*:4 medical*:4 phone:6 nptel:0.5 \
    --save_steps $SAVE --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $O > $A/$R.out 2>&1
  echo "TRAIN $R rc $?"
fi
[ -f $O/$R/$R.nemo ] || { echo "no final model"; exit 1; }
CK=""; for s in $SAVE; do f=$(ls $O/$R/*step${s}*.ckpt $O/$R/step=${s}-*.ckpt 2>/dev/null | head -1); [ -n "$f" ] && CK="$CK $s:$f"; done
echo "epoch checkpoints:$CK"
for c in $CK final:; do s=${c%%:*}; f=${c#*:}
  TKF=$T $P voice/asr/ab_meeting.py --head glu_glu --las 0 1 3 6 13 -1 --bp 0 0.5 0.75 1.0 1.25 1.5 2.0 \
    --nemo $O/$R/$R.nemo ${f:+--ckpt $f} --dump $A/meetstudy/long_$s.json > $A/meetstudy/long_$s.out 2>&1; echo "MEET $s rc $?"
done
echo LONGMIX_DONE
