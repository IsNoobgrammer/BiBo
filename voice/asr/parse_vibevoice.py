"""VibeVoice-ASR demo output (.raw) -> plain transcript (.txt) and segments (.json: Start / End / Speaker / Content).

    python voice/asr/parse_vibevoice.py teachers/vibevoice_clip5m.raw
"""
import json
import sys

for raw in sys.argv[1:]:
    lines = open(raw, encoding="utf-8").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip() == "--- Raw Output ---")
    body = next(l for l in lines[i + 1:] if l.strip() and l.strip() != "assistant")   # the JSON array is one line
    segs = json.loads(body)
    open(raw.replace(".raw", ".json"), "w", encoding="utf-8").write(json.dumps(segs, ensure_ascii=False, indent=0))
    open(raw.replace(".raw", ".txt"), "w", encoding="utf-8").write(" ".join(s["Content"] for s in segs))
    print(raw, len(segs), "segments,", len({s["Speaker"] for s in segs}), "speakers")
