"""Untranscribed-speech audit of a training manifest (deletions hypothesis, Oct 2026).

Per source: hours, words per second of transcript (wps), the share of clips / hours under 1.0 and 1.5 wps. Then, on
the lowest-wps clips of each source (duration >= 3 s), an energy VAD (25 ms frames, speech = within 35 dB of the clip's
loudest frame and above -55 dBFS) separates SILENCE-padded clips (few speech seconds: harmless) from UNTRANSCRIBED
speech (many speech seconds, few words: teaches the model to delete). words per speech-second (wpss) < 1.0 on a clip
with >= 3 s of speech = flagged.

    python voice/asr/data_audit.py en3/train.jsonl [--per_source 150] [--out audit.tsv]
"""
import argparse
import collections
import json
import random

import numpy as np


def vad_speech_sec(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    hop = int(0.025 * sr)
    n = len(x) // hop
    if n == 0:
        return 0.0, 0.0
    e = (x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(e)
    sp = (db > db.max() - 35) & (db > -55)
    return float(sp.sum() * 0.025), float(n * 0.025)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--per_source", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("source", "?")].append(r)
    print(f"{len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.1f} h, {len(by)} sources")
    print(f"{'source':<22}{'clips':>8}{'hours':>8}{'wps':>6}{'<1.0 clips':>11}{'<1.0 h':>8}{'<1.5 clips':>11}{'<1.5 h':>8}")
    tot = [0, 0.0, 0, 0.0]
    for s, rs in sorted(by.items(), key=lambda kv: -sum(r["duration"] for r in kv[1])):
        d = np.array([r["duration"] for r in rs])
        w = np.array([len(r["text"].split()) for r in rs])
        wps = w / np.maximum(d, 1e-3)
        l1, l15 = wps < 1.0, wps < 1.5
        tot[0] += l1.sum(); tot[1] += d[l1].sum() / 3600; tot[2] += l15.sum(); tot[3] += d[l15].sum() / 3600
        print(f"{s:<22}{len(rs):>8}{d.sum() / 3600:>8.1f}{w.sum() / d.sum():>6.2f}{100 * l1.mean():>10.1f}%{d[l1].sum() / 3600:>8.1f}"
              f"{100 * l15.mean():>10.1f}%{d[l15].sum() / 3600:>8.1f}")
    print(f"{'ALL':<22}{len(rows):>8}{'':>8}{'':>6}{100 * tot[0] / len(rows):>10.1f}%{tot[1]:>8.1f}{100 * tot[2] / len(rows):>10.1f}%{tot[3]:>8.1f}")
    print(f"\nVAD on the {a.per_source} lowest-wps clips (>= 3 s) per source: speech s / clip s, words per speech-second (wpss)")
    print(f"{'source':<22}{'n':>5}{'speech%':>9}{'wpss med':>9}{'flagged':>9}{'flag h (est, source-wide)':>27}")
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    random.seed(0)
    for s, rs in sorted(by.items(), key=lambda kv: -sum(r["duration"] for r in kv[1])):
        cand = sorted((r for r in rs if r["duration"] >= 3), key=lambda r: len(r["text"].split()) / r["duration"])
        low = [r for r in cand if len(r["text"].split()) / r["duration"] < 1.5]
        samp = low[:a.per_source]
        if not samp:
            print(f"{s:<22}{0:>5}")
            continue
        sp, wpss, flag = [], [], 0
        for r in samp:
            try:
                ss, tt = vad_speech_sec(r["audio_filepath"])
            except Exception as e:                      # unreadable audio: report, skip
                print(f"  unreadable {r['audio_filepath']}: {e}")
                continue
            words = len(r["text"].split())
            sp.append(ss / max(tt, 1e-3))
            q = words / max(ss, 1e-3)
            wpss.append(q)
            f = ss >= 3 and q < 1.0
            flag += f
            if out:
                out.write(f"{s}\t{r['audio_filepath']}\t{r['duration']:.2f}\t{ss:.2f}\t{words}\t{q:.2f}\t{int(f)}\t{r['text'][:200]}\n")
        frac = flag / max(len(sp), 1)
        low_h = sum(r["duration"] for r in low) / 3600
        print(f"{s:<22}{len(sp):>5}{100 * np.median(sp):>8.0f}%{np.median(wpss):>9.2f}{100 * frac:>8.0f}%"
              f"{frac * low_h:>14.1f} of {low_h:.1f} h <1.5 wps")
    if out:
        out.close()


if __name__ == "__main__":
    main()
