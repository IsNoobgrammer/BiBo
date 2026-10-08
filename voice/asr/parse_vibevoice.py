"""VibeVoice-ASR demo output (.raw) -> plain transcript (.txt) and segments (.json: Start / End / Speaker / Content).

    python voice/asr/parse_vibevoice.py teachers/vibevoice_clip5m.raw
"""
import json
import re
import sys

for raw in sys.argv[1:]:
    t = open(raw, encoding="utf-8").read()
    m = re.search(r"--- Raw Output ---\s*assistant\s*(\[.*?\])\s*\n\s*\n", t, re.S)
    segs = json.loads(m.group(1))
    open(raw.replace(".raw", ".json"), "w", encoding="utf-8").write(json.dumps(segs, ensure_ascii=False, indent=0))
    open(raw.replace(".raw", ".txt"), "w", encoding="utf-8").write(" ".join(s["Content"] for s in segs))
    print(raw, len(segs), "segments,", len({s["Speaker"] for s in segs}), "speakers")
