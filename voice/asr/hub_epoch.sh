#!/bin/bash
# One fine-tuning epoch of a hub encoder arm (Nemotron / Parakeet, Oct 2026 study), then the same evaluation as
# bench_asr.py: meeting clips + en3 validation per look-ahead, blank penalties, scored by score_ctc_decode.py.
#   RUN=hub-nt-hybrid INIT=nvidia/nemotron-speech-streaming-en-0.6b HEADS="rnnt ctc" \
#     EXTRA="--hub_head hybrid --ctc_head glu_glu --ctc_weight 0.4" bash voice/asr/hub_epoch.sh
# HEADS: which decoders to evaluate (rnnt = the RNN-T/TDT head via bench_asr; ctc = our glu_glu head via dump_ctc +
# the app's Rust ctc-decode). Prints HUB_DONE <run>. Recipe: run6's data + look-ahead mix, AdamW (cosine: user rule),
# lr ${LR:-1e-4}, 5% warm-up, final lr 0, 1 epoch = 1875 h = 5625 steps at 1200 s batches; HF checkpoint at the end.
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc; O=$A/exp_hub; B=$A/bench
R=${RUN:?}; I=${INIT:?}; H=${HEADS:-rnnt}; X=${EXTRA:-}; LR=${LR:-1e-4}
# our kernels (probe: Nemotron 0.41 s/step fused vs 0.67 NeMo); --fused_ctc only with our CTC head
F=${FUSED:---fused_joint --fused_layer --fused_attn --fused_conv}; [[ "$X" == *"--ctc_head"* ]] && F="$F --fused_ctc"
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
mkdir -p $B $A/ctcdump                    # fresh boxes have neither (the first Parakeet arm's evals were lost to it)
D=/home/marimo/work/rtc/tools/ctc-decode/target/release/ctc-decode
[[ "$H" == *ctc* && ! -x $D ]] && echo "HUB_WARN $R: no ctc-decode binary at $D (build it: awfull.sh)"
cd /home/marimo/work/BiBo
echo "$(date +%H:%M) HUB_START $R | BiBo $(git log --oneline -1 | cut -c1-40) | init $I lr $LR extra [$X] fused [$F] heads [$H]"
if [ ! -f $O/$R/$R.nemo ]; then
  TKF=$T $P voice/asr/train_asr.py --project asr-hub --run $R --seed 23 --init $I --keep_vocab $X $F \
    --train ${TRAIN:-$E3/train.jsonl} --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL --optim adamw --lr $LR --warmup_frac 0.05 \
    --min_lr 0 --total_hours ${HOURS:-1875} --eval_hours ${HOURS:-1875} --fastemit 0 --batch_sec 1200 --lookaheads 13 6 3 1 0 -1 \
    --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20 --out $O > $A/$R.out 2>&1
  echo "$(date +%H:%M) HUB_TRAIN $R rc $? $(grep -E 'it/s|s/it' $A/$R.out | tail -1 | tr '\r' '\n' | tail -1 | grep -oE '[0-9.]+ ?(it/s|s/it)')"
fi
[ -f $O/$R/$R.nemo ] || { echo "HUB_FAIL $R no .nemo"; exit 1; }
sc() { $P voice/asr/score_ctc_decode.py $1/refs.jsonl $1/hyp.tsv --tag " $(basename $1)" > $1/score.txt 2>&1
       echo "$(date +%H:%M) HUB_SCORE $(basename $1) $(grep meeting $1/score.txt | awk '{print $4, $5, $6, $7, $8, $9, $10}') val $(grep SCORE_MEAN $1/score.txt | awk '{print $4}')"; }
SC=none; [[ "$X" == *"--ctc_head glu_glu"* ]] && SC=glu_glu
for head in $H; do
  if [ $head = rnnt ]; then
    for la in 0 1 3 6 13 -1; do n=$R-rnnt-la$la
      $P voice/asr/bench_asr.py --model $O/$R/$R.nemo --ctc_head $SC --head rnnt --la $la --out $B/$n --bps 0 0.5 1.0 > $B/$n.log 2>&1
      for bp in 0 0.5 1; do [ -f $B/${n}_bp$bp/hyp.tsv ] && sc $B/${n}_bp$bp; done
    done
  else                                    # our CTC head: logits once per look-ahead, the app's decoder per penalty
    D=/home/marimo/work/rtc/tools/ctc-decode/target/release/ctc-decode
    for la in 3 -1; do M=$A/ctcdump/$R-la$la
      TKF=$T $P voice/asr/dump_ctc.py --nemo $O/$R/$R.nemo --head glu_glu --out $M --val $E3/val_*.jsonl --la $la \
        --meeting --las $la --tok $O/hub_tok/${I//\//_}.model > $M.log 2>&1      # the hub model's own pieces
      for bp in 0.0 0.5 1.0; do n=$B/$R-ctc-la${la}_bp$bp; mkdir -p $n; cp $M/refs.jsonl $n/
        $D --vocab $M/pieces.txt --beam 1 --blank-penalty $bp --chunk 0 --classes $(($(wc -l < $M/pieces.txt) + 1)) @$M/list.txt > $n/hyp.tsv 2> $n/err.txt
        sc $n; done
    done
  fi
done
echo "$(date +%H:%M) HUB_DONE $R"
