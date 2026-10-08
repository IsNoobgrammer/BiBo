"""Quality report of an ASR training mix with Qwen3-ASR-1.7B: per-utterance WER of the dataset transcript vs Qwen's.
Measures only, unless --keep_below is given (then rows above that WER are REMOVED from the manifest).

    python voice/asr/qwen_check.py --mix /home/marimo/work/asr/mix100 [--per_source 300]

High WER means the transcript and the audio disagree (bad alignment, wrong text, wrong language) OR Qwen failed; the
report shows the worst rows per source so a person can tell which. Qwen's cased + punctuated text is kept as `qwen_text`
in <mix>/qwen/<source>.jsonl. Number spelling differs (dataset "nineteen ninety" vs Qwen "1990"), so a few points of
WER per source are formatting, not label noise.
"""
import argparse
import glob
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import normalize, wer  # noqa: E402

LANG = {"en": "English", "hi": "Hindi"}


def load():
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    mid = "Qwen/Qwen3-ASR-1.7B-hf"
    return AutoProcessor.from_pretrained(mid), AutoModelForMultimodalLM.from_pretrained(
        mid, device_map="cuda", dtype=torch.bfloat16).eval()


@torch.inference_mode()
def transcribe(proc, m, rows):
    import soundfile as sf
    audio = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rows]
    inputs = proc.apply_transcription_request(audio=audio, language=[LANG[r["lang"]] for r in rows]).to(m.device, m.dtype)
    out = m.generate(**inputs, max_new_tokens=448)
    return proc.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--per_source", type=int, default=0, help="random rows per source (0 = all)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--sources", default="", help="comma list of manifests to check (default: all)")
    ap.add_argument("--keep_below", type=float, default=0,
                    help="> 0: REMOVE rows above this WER (bad labels): <src>.jsonl -> <src>_raw.jsonl, kept rows -> <src>.jsonl")
    a = ap.parse_args()
    proc, m = load()
    qdir = os.path.join(a.mix, "qwen")
    os.makedirs(qdir, exist_ok=True)
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        src = os.path.basename(man)[:-6]
        if (a.sources and src not in a.sources.split(",")) or src.endswith("_raw"):
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        if a.per_source:
            rows = random.Random(23).sample(rows, min(a.per_source, len(rows)))
        rows.sort(key=lambda r: r["duration"])                 # similar lengths per batch -> less padding
        edits = words = 0
        tag = "audit" if a.per_source else "all"
        with open(os.path.join(qdir, f"{src}_{tag}.jsonl"), "w", encoding="utf-8") as fo:
            for i in range(0, len(rows), a.bs):
                batch = rows[i:i + a.bs]
                for r, h in zip(batch, transcribe(proc, m, batch)):
                    ref, hyp = normalize(r["text"]), normalize(h)
                    r["qwen_text"], r["wer"] = h, round(wer(ref, hyp), 4)
                    edits += r["wer"] * len(ref)
                    words += len(ref)
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        ws = sorted(r["wer"] for r in rows)
        print(f"SOURCE {src}: n={len(rows)} {sum(r['duration'] for r in rows) / 3600:.1f} h  corpusWER="
              f"{100 * edits / max(words, 1):.1f}% median={100 * ws[len(ws) // 2]:.1f}%  <=10%: "
              f"{100 * sum(w <= .1 for w in ws) / len(ws):.0f}%  >50%: {100 * sum(w > .5 for w in ws) / len(ws):.0f}%", flush=True)
        for sc in sorted({r.get("scenario") for r in rows} - {None}):
            sub = [r for r in rows if r.get("scenario") == sc]
            e = sum(r["wer"] * len(normalize(r["text"])) for r in sub)
            print(f"  scenario {sc}: n={len(sub)} corpusWER={100 * e / max(sum(len(normalize(r['text'])) for r in sub), 1):.1f}%", flush=True)
        for r in sorted(rows, key=lambda r: -r["wer"])[:3]:
            print(f"  WORST wer={r['wer']:.2f} {os.path.basename(r['audio_filepath'])}\n    ref: {r['text'][:160]}\n    qwen: {r['qwen_text'][:160]}", flush=True)
        if a.keep_below and not a.per_source:
            keep = [r for r in rows if r["wer"] <= a.keep_below]
            os.replace(man, os.path.join(a.mix, f"{src}_raw.jsonl"))
            with open(man, "w", encoding="utf-8") as fo:
                fo.writelines(json.dumps({k: v for k, v in r.items() if k not in ("qwen_text", "wer")},
                                         ensure_ascii=False) + "\n" for r in keep)
            print(f"KEPT {src}: {len(keep)}/{len(rows)} rows, {sum(r['duration'] for r in keep) / 3600:.1f}"
                  f"/{sum(r['duration'] for r in rows) / 3600:.1f} h (dropped WER > {a.keep_below})", flush=True)


if __name__ == "__main__":
    main()
