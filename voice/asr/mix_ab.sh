#!/bin/bash
# Hard-set mix and look-ahead mix A/B (user, Oct 11 2026), from NVIDIA's stock model, seed 23, 11,250 steps (the CTC A/B's budget, so
# equal compute: ~1.65 passes over the bigger mix), 5% warm-up, validation at 240 ms (--val_lookahead default).
#   CTCW=0.4 [ARMS="la mix fe0"] nohup bash voice/asr/mix_ab.sh > /home/marimo/work/asr/mix_ab.out 2>&1 &
#   0. re-validate the two CTC-weight arms at 240 ms (they validated at 1040 ms) -> the baseline for arms A/B
#   A. mix-w*      : hard sets up-weighted (NOTSOFAR x3, AMI windows x2, Spotify/Svarah/Medical x4, phone x6, NPTEL x0.5)
#   B. la-w*       : OLD data mix, look-ahead mix 0:10% 80ms:23% 240ms:32% 480ms:15% 1040ms:0% full:20%
#   C. fe0-w*      : old data mix, old look-ahead mix, --fastemit 0 (baseline 0.01; never ablated)
#   Each arm changes ONE thing against the CTC-weight winner's arm (the baseline, already trained).
# Re-running skips finished arms and resumes an unfinished one.
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; T=/home/marimo/work/tkf_ctc
[ -f $A/.markers/ready ] || { echo "en3 not ready (run setup_en3.sh)"; exit 1; }
W=${CTCW:?set CTCW to the CTC-weight winner, e.g. 0.6}
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo
echo "BiBo $(git log --oneline -1 | cut -c1-60) | tkf $(git -C $T log --oneline -1 | cut -c1-60) | CTC weight $W"
COMMON="--seed 23 --new_vocab --train $E3/train.jsonl --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL
  --lr 5e-4 --total_hours 3750 --eval_hours 750 --ctc_head glu_glu --batch_sec 1200
  --lookaheads 13 6 3 1 0 -1 --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc"
LA_OLD="0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20"
LA_NEW="0:0.10 1:0.23 3:0.32 6:0.15 13:0 full:0.20"
MIX="notsofar:3 multispk_ami:2 spotify*:4 svarah*:4 medical*:4 phone:6 nptel:0.5"
for w in 0.4 0.6; do                                     # 0. baselines at 240 ms
  R=ctcab-stock-w${w/./}; V=$R-val240
  grep -q VALIDATE_DONE $A/$V.out 2>/dev/null && { echo "VAL $V already done"; continue; }
  CK=$A/exp_ctcab_stock/$R/last.ckpt; [ -f $CK ] || CK=$A/hf_ckpt/$R/last.ckpt
  [ -f $CK ] || $P -c "from huggingface_hub import hf_hub_download as d; d('fhai50032/bibo-asr-ckpt', '$R/last.ckpt', local_dir='$A/hf_ckpt')"
  TKF=$T $P voice/asr/train_asr.py --project asr-ctc-weight --run $V --no_hf --validate_only --ckpt $CK $COMMON \
    --ctc_weight $w --fastemit 0.01 --lookahead_probs $LA_OLD --out $A/exp_val240 > $A/$V.out 2>&1; echo "VAL $V rc $?"
done
for arm in ${ARMS:-la}; do                           # B first; A (mix) only when asked: ARMS=mix
  R=$arm-w${W/./}
  LA=$LA_OLD; SW=""; FE=0.01
  case $arm in mix) SW="--source_weight $MIX";; la) LA=$LA_NEW;; fe0) FE=0;; *) echo "unknown arm $arm"; exit 1;; esac
  [ -f $A/exp_mix/$R/$R.nemo ] && { echo "ARM $R already done"; continue; }
  TKF=$T $P voice/asr/train_asr.py --project asr-mix --run $R --ctc_weight $W $COMMON --lookahead_probs $LA --fastemit $FE \
    --source_weight $MIX --out $A/exp_mix > $A/$R.out 2>&1
  rc=$?; echo "ARM $R TRAIN rc $rc"
  [ $rc -eq 0 ] && TKF=$T $P voice/asr/ab_meeting.py --rnnt --head glu_glu --las 3 13 -1 --bp 0 0.5 1.0 1.25 1.5 \
    --nemo $A/exp_mix/$R/$R.nemo > $A/meet_$R.out 2>&1; echo "ARM $R MEET rc $?"
done
echo "MIXAB_DONE"
