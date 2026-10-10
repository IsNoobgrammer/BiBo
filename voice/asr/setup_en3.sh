#!/bin/bash
# Fresh molab box, after setup_box.sh -> ready to train on en3: tkf kernels, run5.nemo, the en3 mix, the en3
# manifests + tokenizer (rebuilt by prep_en, then checked against the old box's exact copy on HF), kernel parity.
# Idempotent: every step leaves a marker in $A/.markers and is skipped on a re-run. Run DETACHED (the data_en3
# notebook cell does): a foreground cell is interrupted when the calling client goes away and leaves a half box.
#   nohup bash voice/asr/setup_en3.sh > /home/marimo/setup_en3.log 2>&1 &     -> last line EN3_READY
set -uo pipefail
W=/home/marimo/work; A=$W/asr; P=/home/marimo/asrenv/bin/python; M=$A/.markers
mkdir -p "$M"
until [ -f /home/marimo/asrenv/.setup_done ]; do sleep 15; done; echo "asrenv ready"
cd $W && { [ -d tkf_ctc ] || git clone -q https://github.com/IsNoobgrammer/triton-kernel-fused.git tkf_ctc; } \
  && git -C tkf_ctc pull -q && git -C tkf_ctc log --oneline -1
if [ ! -f "$M/run5" ]; then
  mkdir -p $A/exp/run5
  $P -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('fhai50032/bibo-asr-ckpt', 'run5/run5.nemo'), '$A/exp/run5/run5.nemo')" \
    && touch "$M/run5" || { echo "RUN5 FAILED"; exit 1; }
fi
echo "run5.nemo ok"
if [ ! -f "$M/fetch" ]; then
  # en3's sources live in THREE repos (base / v2: meetings, far-field, read, VITW / nc: SPGI 2.0, TED-LIUM), merged
  # into one mix dir. Oct 10: fetching only asr-english left most of en3's rows without audio.
  for R in asr-english asr-english-v2 asr-english-nc; do
    (cd $W/BiBo && HF_HUB_ENABLE_HF_TRANSFER=1 $P voice/asr/fetch_mix.py --repo fhai50032/$R --mix $A/mix_en --merge) \
      || { echo "FETCH FAILED $R"; exit 1; }
  done
  touch "$M/fetch"
fi
echo "mix_en ok"
if [ ! -f "$M/en3" ]; then
  (cd $W/BiBo && $P voice/asr/prep_en.py --mix $A/mix_en --out $A/en3 --vocab 2047) \
    && $P $W/BiBo/voice/asr/en3_verify.py && touch "$M/en3" || { echo "EN3 FAILED"; exit 1; }
fi
echo "en3 ok: $(wc -l < $A/en3/train.jsonl) train rows"
for t in parity_relpos_attn parity_rnnt_joint parity_muon_rc; do
  $P $W/tkf_ctc/parity_check/$t.py 2>&1 | tail -1 | grep -q "ALL PASS" || { echo "PARITY FAILED: $t"; exit 1; }
  echo "$t: ALL PASS"
done
touch "$M/ready"; echo EN3_READY
