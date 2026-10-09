#!/bin/bash
# run2 sweep: 3 epochs each on the 1,200 h mix (fhai50032/asr-english + asr-hindi, every source x1), from the NVIDIA
# base model with run1's tokenizer, all tkf kernels, eval at every data epoch end + the final model. One factor per arm vs base:
#   base     lr 5e-4, 1200 s batches, bf16 autocast (fp32 weights)
#   bf16m    bf16 weights + fp32 master (--bf16_master)
#   lr1e3 / lr2e4    learning rate
#   b2400    2400 s batches (same lr)
# W&B project bibo-asr-sweep (core/ train loss, val/*/wer per epoch). ARMS="base lr1e3" picks arms.
#   bash voice/asr/sweep_run2.sh
set -uo pipefail
W=/home/marimo/work; A=$W/asr; M=$A/mix1000; R=$A/run2; E=$A/eval_sets; P=/home/marimo/asrenv/bin/python
cd $W/BiBo && git log --oneline -1
# data: two repos, one mix dir (fetch_mix skips a dir that already has manifests, so Hindi lands in a side dir first)
$P voice/asr/fetch_mix.py --repo fhai50032/asr-english --mix $M
[ -f $M/.hindi_fetched ] || { mkdir -p $A/mix1000_hi && $P voice/asr/fetch_mix.py --repo fhai50032/asr-hindi --mix $A/mix1000_hi \
  && mv $A/mix1000_hi/*.jsonl $M/ && touch $M/.hindi_fetched; }
[ -f $R/train.jsonl ] || $P voice/asr/prep_train.py --mix $M --out $R
[ -f $E/fleurs_hi.jsonl ] || $P voice/asr/build_eval.py --out $E
TOK=$A/run1/tok/tokenizer_spe_bpe_v4096                       # run1's joint en/hi tokenizer, same for every arm
H=$($P -c "import json; print(round(sum(json.loads(l)['duration'] for l in open('$R/train.jsonl')) / 3600))")
echo "train hours per epoch: $H"
VALS="$(ls $R/val_*.jsonl | grep -v -e '/val_en.jsonl' -e '/val_hi.jsonl') $E/fleurs_en.jsonl $E/fleurs_hi.jsonl"
for arm in ${ARMS:-base bf16m lr1e3 lr2e4 b2400}; do
  case $arm in
    base)  X="" ;;
    bf16m) X="--bf16_master" ;;
    lr1e3) X="--lr 1e-3" ;;
    lr2e4) X="--lr 2.5e-4" ;;
    b2400) X="--batch_sec 2400" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  rm -rf $A/exp_sweep/s-$arm
  $P voice/asr/train_asr.py --run s-$arm --out $A/exp_sweep --project bibo-asr-sweep --no_hf --seed 23 \
    --train $R/train.jsonl --tok $TOK --val $VALS --total_hours $((3 * H)) --eval_hours 0 \
    --fused_joint --fused_layer --fused_attn --fused_conv $X > $A/sweep_$arm.log 2>&1
  echo "ARM_END $arm $?"
done
echo SWEEP_DONE
