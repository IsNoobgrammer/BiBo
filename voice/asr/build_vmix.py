"""Verbatim-heavy training mix (user, Oct 12 2026): shrink the big clean / edited sources, keep every verbatim source,
add the augmented verbatim copies.

The verbatim fine-tune set (vft, build_vft.py) reuses en3's own rows for its originals (same audio, same text: all
98,758 matched), so only its augmented copies (far field + noise, echo + reverb + obstruction: 289.5 h) are new
audio. This mix = en3 train with --cap sources randomly cut to N hours (seeded per source) + vft rows whose source
is not an en3 source (the copies).

    python voice/asr/build_vmix.py --cap spgi2:100 nptel:100 emilia:100 tedlium:100 librispeech:50 \
        --out /home/marimo/work/asr/vmix/train.jsonl
"""
import argparse
import collections
import json
import os
import random

A = "/home/marimo/work/asr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en3", default=f"{A}/en3/train.jsonl")
    ap.add_argument("--vft", default=f"{A}/vft/train.jsonl")
    ap.add_argument("--cap", nargs="*", default=[], help="SOURCE:HOURS, e.g. spgi2:100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    cap = {s: float(h) for s, h in (c.split(":") for c in a.cap)}
    by = collections.defaultdict(list)
    for l in open(a.en3, encoding="utf-8"):
        r = json.loads(l)
        by[r["source"]].append(r)
    miss = [s for s in cap if s not in by]
    assert not miss, f"--cap sources not in en3: {miss}"
    rows, before, after = [], {}, {}
    for s in sorted(by):
        rs = by[s]
        before[s] = sum(r["duration"] for r in rs) / 3600
        if s in cap:
            random.Random(f"{a.seed}:{s}").shuffle(rs)
            keep, h = [], 0.0
            for r in rs:
                if h >= cap[s] * 3600:
                    break
                keep.append(r)
                h += r["duration"]
            rs = keep
        rows += rs
        after[s] = sum(r["duration"] for r in rs) / 3600
    add = collections.Counter()
    for l in open(a.vft, encoding="utf-8"):
        r = json.loads(l)
        if r["source"] not in by:                       # vft originals are en3 rows: only the copies are new
            rows.append(r)
            add[r["source"]] += r["duration"] / 3600
    missing = sum(1 for r in rows if not os.path.exists(r["audio_filepath"]))
    random.Random(a.seed).shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in sorted(before, key=lambda s: -before[s]):
        print(f"  {s:24s} {before[s]:7.1f} h -> {after[s]:7.1f} h" + ("  (capped)" if s in cap else ""))
    for s, h in add.most_common():
        print(f"  + {s:22s} {h:7.1f} h (augmented copy)")
    tot = sum(r["duration"] for r in rows) / 3600
    print(f"VMIX rows {len(rows)} hours {tot:.1f} (en3 {sum(before.values()):.1f}) missing audio {missing}")


if __name__ == "__main__":
    main()
