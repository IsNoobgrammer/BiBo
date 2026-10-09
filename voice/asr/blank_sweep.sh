#!/bin/bash
# RNN-T blank-penalty sweep (decode only, blank_penalty.py): does penalising blank recover dropped words without
# paying it back in insertions? Per delta: the internal meeting clips as a TRUE stream (RNN-T, 480 ms lookahead =
# att [70,6]; WER + words out / words in the reference) and the val sets offline (diag_deletions TOTAL S/D/I).
#   bash voice/asr/blank_sweep.sh run2v2 "0 0.5 1 1.5 2 3"
set -uo pipefail
RUN=${1:-run2v2}; DELTAS=${2:-0 0.5 1 1.5 2 3}
W=/home/marimo/work; A=$W/asr; M=$A/eval_meeting; R=$A/run2; E=$A/eval_sets; P=/home/marimo/asrenv/bin/python
NEMO=$A/exp/$RUN/$RUN.nemo
cd $W/BiBo && git log --oneline -1
[ -f $M/meeting.jsonl ] || $P - <<EOF
import json, soundfile as sf
with open("$M/meeting.jsonl", "w") as f:
    for wav in ("c2m.wav", "clip16k.wav"):
        x, sr = sf.read("$M/" + wav)
        f.write(json.dumps({"audio_filepath": "$M/" + wav, "duration": len(x) / sr, "text": ""}) + "\n")
EOF
for d in $DELTAS; do
  out=$M/$RUN.bp$d.jsonl
  rm -rf $out
  $P voice/asr/blank_penalty.py $d $W/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=$NEMO dataset_manifest=$M/meeting.jsonl output_path=$out batch_size=1 "att_context_size=[70,6]" \
    decoder_type=rnnt > $M/$RUN.bp$d.log 2>&1 || { echo "FAILED bp $d"; tail -5 $M/$RUN.bp$d.log; continue; }
  $P - <<EOF
import glob, json, sys
sys.path.insert(0, "voice/asr")
from score import normalize, wer
order = [json.loads(l)["audio_filepath"].rsplit("/", 1)[-1] for l in open("$M/meeting.jsonl")]
rows = dict(zip(order, map(json.loads, open(glob.glob("$out/*.json")[0], encoding="utf-8"))))
res = []
for wav, ref in (("c2m.wav", "ref2m.txt"), ("clip16k.wav", "reference.txt")):
    rw, hw = normalize(open("$M/" + ref, encoding="utf-8").read()), normalize(rows[wav].get("pred_text", ""))
    res.append(f"{wav.split('.')[0]} {100 * wer(rw, hw):5.1f}% ({len(hw)}/{len(rw)} words)")
print(f"MEETING $RUN blank penalty {float($d):g} rnnt 480 ms |", " | ".join(res), flush=True)
EOF
  $P voice/asr/diag_deletions.py --nemo $NEMO --val $(ls $R/val_*.jsonl) $E/fleurs_en.jsonl $E/fleurs_hi.jsonl \
    --max_per_source 200 --pad 0 --blank_penalty $d 2>&1 | grep -a "^TOTAL"
done
echo SWEEP_DONE
