#!/bin/bash
# Regularisation arm (user, Oct 12 2026): the CTC 0.4 stock-init baseline (ctcab-stock-w04) with RNN-T joint dropout
# 0.1 (config 0.2) and Muown weight decay 0 (was 1e-3; AdamW group keeps 1e-3). Everything else identical (FastEmit
# 0.01, old mix and look-ahead mix, seed 23, 11,250 steps, 5% warm-up), then the full meeting grid (v2 scoring).
#   nohup bash voice/asr/reg_ab.sh > /home/marimo/work/asr/reg_ab.out 2>&1 &
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc
[ -f $A/.markers/ready ] || { echo "en3 not ready"; exit 1; }
R=${RUN:-reg-w04}; VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo
echo "BiBo $(git log --oneline -1 | cut -c1-60) | $R"
if [ ! -f $A/exp_reg/$R/$R.nemo ]; then
  TKF=$T $P voice/asr/train_asr.py --project asr-ctc-weight --run $R --seed 23 --new_vocab --train $E3/train.jsonl \
    --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL --lr 5e-4 --total_hours 3750 --eval_hours 750 \
    --ctc_head glu_glu --ctc_weight 0.4 --fastemit 0.01 --batch_sec 1200 --lookaheads 13 6 3 1 0 -1 \
    --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20 --joint_dropout ${JD:-0.1} --muon_wd ${MWD:-0} \
    --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $A/exp_reg > $A/$R.out 2>&1
  echo "TRAIN $R rc $?"
fi
TKF=$T $P voice/asr/ab_meeting.py --head glu_glu --las 0 1 3 6 13 -1 --bp 0 0.5 0.75 1.0 1.25 1.5 2.0 \
  --nemo $A/exp_reg/$R/$R.nemo --dump $A/meetstudy/v2_$R.json > $A/meetstudy/v2_$R.out 2>&1; echo "MEET $R rc $?"
echo REG_DONE
