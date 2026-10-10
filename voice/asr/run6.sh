#!/bin/bash
# run6: 10 epochs of en3 from NVIDIA's pretrained streaming FastConformer (train_asr's default --init), Muown cosine
# (the default optimizer), CTC 0.6 / RNN-T 0.4, glu_glu self-conditioned CTC head, all lookaheads + full context.
# Re-running resumes from the latest checkpoint (local, else HF bibo-asr-ckpt/run6). Needs setup_en3.sh's EN3_READY.
#   nohup bash voice/asr/run6.sh > /home/marimo/work/asr/run6.out 2>&1 &
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3
[ -f $A/.markers/ready ] || { echo "en3 not ready (run setup_en3.sh)"; exit 1; }
VAL=$(ls $E3/val_*.jsonl | sort | tr '\n' ' ')
cd /home/marimo/work/BiBo
TKF=/home/marimo/work/tkf_ctc /home/marimo/asrenv/bin/python voice/asr/train_asr.py --project asr-run6 --run run6 \
  --seed 23 --new_vocab --train $E3/train.jsonl --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $VAL \
  --lr 5e-4 --warmup 200 --total_hours 18750 --eval_hours 1875 \
  --ctc_head glu_glu --ctc_weight 0.6 --fastemit 0.01 --batch_sec 1200 \
  --lookaheads 13 6 3 1 0 -1 --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20 \
  --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $A/exp_run6
rc=$?; echo "TRAIN rc $rc"
[ $rc -eq 0 ] && /home/marimo/asrenv/bin/python voice/asr/ab_meeting.py --rnnt --head glu_glu --las 3 13 -1 \
  --nemo $A/exp_run6/run6/run6.nemo > $A/meet_run6.out 2>&1; echo "RUN6_DONE"
