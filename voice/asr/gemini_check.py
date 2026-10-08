"""Second-opinion label audit with Gemini (default gemini-3.8-flash): per-row WER of the dataset transcript vs Gemini's.
Measures only -- nothing is filtered. Uses the SAME rows as qwen_check.py's audit when that file exists, so the two
judges can be compared row by row: a row both judges disagree with is a bad LABEL (misalignment), a row only one judge
disagrees with is that judge's error.

    GEMINI_API_KEY=... python voice/asr/gemini_check.py --mix /home/marimo/work/asr/audit_nptel [--per_source 300]

Public dataset audio only -- never point this at the internal meeting clips.
"""
import argparse
import base64
import glob
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import normalize, wer  # noqa: E402

PROMPT = ("Transcribe this audio verbatim. Write Hindi words in Devanagari and English words in Latin script, exactly as "
          "spoken; do not translate, summarise or add anything. Output only the transcript text.")


def gemini(path, model, key):
    body = {"contents": [{"parts": [{"text": PROMPT},
                                    {"inline_data": {"mime_type": "audio/flac",
                                                     "data": base64.b64encode(open(path, "rb").read()).decode()}}]}],
            "generationConfig": {"temperature": 0}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as e:
            if attempt == 5:
                return f"<ERROR {type(e).__name__}>"
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--per_source", type=int, default=300)
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--worst", type=int, default=0, help="> 0: only the N rows Qwen scored worst (cheap spot check)")
    a = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        src = os.path.basename(man)[:-6]
        qp = os.path.join(a.mix, "qwen", f"{src}_audit.jsonl")
        if os.path.exists(qp):                                     # same rows as the Qwen audit
            rows = [json.loads(l) for l in open(qp, encoding="utf-8")]
            if a.worst:
                rows = sorted(rows, key=lambda r: -r["wer"])[:a.worst]
        else:
            rows = [json.loads(l) for l in open(man, encoding="utf-8")]
            rows = random.Random(23).sample(rows, min(a.per_source, len(rows)))
        with ThreadPoolExecutor(a.workers) as pool:
            hyps = list(pool.map(lambda r: gemini(r["audio_filepath"], a.model, key), rows))
        edits = words = 0
        both = only_g = only_q = 0
        out = []
        for r, h in zip(rows, hyps):
            ref = normalize(r["text"])
            r["gemini_text"], r["gemini_wer"] = h, round(wer(ref, normalize(h)), 4)
            edits += r["gemini_wer"] * len(ref)
            words += len(ref)
            if "wer" in r:                                          # Qwen audit value present
                g, q = r["gemini_wer"] > 0.5, r["wer"] > 0.5
                both, only_g, only_q = both + (g and q), only_g + (g and not q), only_q + (q and not g)
            out.append(r)
        os.makedirs(os.path.join(a.mix, "gemini"), exist_ok=True)
        with open(os.path.join(a.mix, "gemini", f"{src}_audit.jsonl"), "w", encoding="utf-8") as fo:
            fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
        ws = sorted(r["gemini_wer"] for r in out)
        errs = sum(h.startswith("<ERROR") for h in hyps)
        n = len(out)
        print(f"GEMINI {src}: n={n} errors={errs} corpusWER={100 * edits / max(words, 1):.1f}% "
              f"median={100 * ws[n // 2]:.1f}%  <=10%: {100 * sum(w <= .1 for w in ws) / n:.0f}%  "
              f">50%: {100 * sum(w > .5 for w in ws) / n:.0f}%", flush=True)
        if any("wer" in r for r in out):
            print(f"  >50% WER by both judges (bad label): {both}  only Gemini: {only_g}  only Qwen: {only_q}", flush=True)
        for r in sorted(out, key=lambda r: -r["gemini_wer"])[:3]:
            print(f"  WORST g={r['gemini_wer']:.2f} q={r.get('wer', float('nan')):.2f}\n    ref:    {r['text'][:150]}"
                  f"\n    gemini: {r['gemini_text'][:150]}", flush=True)


if __name__ == "__main__":
    main()
