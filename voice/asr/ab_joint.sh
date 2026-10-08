#!/bin/bash
# A/B of the fused joint + RNN-T loss kernel in REAL training: 300 steps from run1.nemo (identical start), same
# Lhotse batches (seed 23), lr 5e-4 cosine with 100 warm-up steps. Arms: nemo, nemo again (the run-to-run noise
# floor: dropout masks differ), fused. ARMS="fused2 fused-bf16" bash ... picks arms (fused* = --fused_joint,
# *bf16* = --bf16_master, *layer* = --fused_layer, *r32* = --fp32_residual). W&B project bibo-asr-ab; compare core/ train loss and speed/ audio-s/s.
#   bash voice/asr/ab_joint.sh            HOURS= train hours, EVAL_H= eval every, VALS= val manifests
set -uo pipefail
W=/home/marimo/work; A=$W/asr; R=$A/run1; E=$A/eval_sets; P=/home/marimo/asrenv/bin/python
cd $W/BiBo && git log --oneline -1
for arm in ${ARMS:-nemo nemo2 fused}; do
  rm -rf $A/exp_ab/ab-$arm
  $P voice/asr/train_asr.py --run ab-$arm --out $A/exp_ab --project bibo-asr-ab --no_hf --seed 23 \
    --init $R/../exp/run1/run1.nemo --train $R/train.jsonl --tok $R/tok/tokenizer_spe_bpe_v4096 \
    --val ${VALS:-$E/fleurs_en.jsonl} --total_hours ${HOURS:-100} --eval_hours ${EVAL_H:-100000} --warmup 100 \
    $(case $arm in fused*) echo --fused_joint;; esac) $(case $arm in *bf16*) echo --bf16_master;; esac) \
    $(case $arm in *layer*) echo --fused_layer;; esac) $(case $arm in *r32*) echo --fp32_residual;; esac) \
    > $A/exp_ab_$arm.log 2>&1
  echo "ARM_END $arm $?"
done
echo AB_DONE
