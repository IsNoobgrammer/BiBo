#!/bin/bash
# Score a trained run on the internal meeting clips (EVAL ONLY: never train on them, never push them) as a TRUE stream
# (NeMo cache-aware streaming loop, the way RT Captions would run it), at every lookahead, with both heads.
# Same scorer + normalizer as the teacher / Nemotron baselines in voice/README.md.
#   bash voice/asr/eval_meeting.sh run1         (waits for exp/run1/run1.nemo)   BP=0.5 bash ... : blank penalty
set -uo pipefail
RUN=${1:-run1}
BP=${BP:-}                                        # BP=0.5: decode-time blank penalty (blank_penalty.py), both heads
LOCK=${LOCK:-}                                    # LOCK=en: English script lock (no Devanagari tokens), as RT Captions
[ -n "$LOCK" ] && BP=${BP:-0}
TAG=$RUN${BP:+.bp$BP}${LOCK:+.$LOCK}
W=/home/marimo/work; A=$W/asr; M=$A/eval_meeting; P=/home/marimo/asrenv/bin/python
NEMO=$A/exp/$RUN/$RUN.nemo
until [ -f "$NEMO" ]; do sleep 30; done
sleep 20                                         # let save_to finish writing
cd $W/BiBo && git log --oneline -1
$P voice/asr/nemo_ctx_fix.py $NEMO ${NEMO%.nemo}.eval.nemo && NEMO=${NEMO%.nemo}.eval.nemo   # [70,3] -> [68,3]
$P - <<EOF
import json, soundfile as sf
with open("$M/meeting.jsonl", "w") as f:
    for wav in ("c2m.wav", "clip16k.wav", "g5_16k.wav"):
        x, sr = sf.read("$M/" + wav)
        f.write(json.dumps({"audio_filepath": "$M/" + wav, "duration": len(x) / sr, "text": ""}) + "\n")
EOF
for att in 0 1 3 6 13; do
  left=$((70 - 70 % (att + 1)))                  # the training mask's effective left context
  for dec in rnnt ctc; do
    out=$M/$TAG.la$att.$dec.jsonl
    rm -rf $out
    $P ${BP:+voice/asr/blank_penalty.py $dec:$BP${LOCK:+:$LOCK}} $W/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
      model_path=$NEMO dataset_manifest=$M/meeting.jsonl output_path=$out batch_size=1 \
      "att_context_size=[$left,$att]" decoder_type=$dec > $M/$TAG.la$att.$dec.log 2>&1 || { echo "FAILED la$att $dec"; tail -5 $M/$TAG.la$att.$dec.log; continue; }
    $P - <<EOF
import glob, json, sys
sys.path.insert(0, "voice/asr")
from score import normalize, wer
# NeMo's streaming script treats output_path as a DIRECTORY and writes streaming_out_*.json in manifest order
order = [json.loads(l)["audio_filepath"].rsplit("/", 1)[-1] for l in open("$M/meeting.jsonl")]
rows = dict(zip(order, map(json.loads, open(glob.glob("$out/*.json")[0], encoding="utf-8"))))
res = []
for wav, ref in (("c2m.wav", "ref2m.txt"), ("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt")):
    rw, hw = normalize(open("$M/" + ref, encoding="utf-8").read()), normalize(rows[wav].get("pred_text", ""))
    res.append(f"{wav.split('.')[0]} {100 * wer(rw, hw):5.1f}% ({len(hw)}/{len(rw)} words)")
print(f"MEETING $TAG lookahead {int($att) * 80:4d} ms {'$dec':4s} |", " | ".join(res), flush=True)
EOF
  done
done
echo MEETING_DONE
