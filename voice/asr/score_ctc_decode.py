"""Score an external CTC decoder's output (`file<TAB>text` lines, e.g. RT Captions' tools/ctc-decode) against
dump_ctc.py's refs.jsonl: pooled WER (S D I) per set, the mean over validation sets and the OOD-like mean.

    python voice/asr/score_ctc_decode.py DUMPDIR/refs.jsonl DECODED.tsv [--tag NAME] [--norm v1|v2]

v1 = score.normalize (data prep / validation); v2 = normalize_v2 (meeting eval: contractions expanded). Meeting sets
(meeting_la*) are always scored with v2, as ab_meeting.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import edits, normalize, normalize_v2  # noqa: E402

OOD = ("notsofar", "sdm", "medical_room", "spotify_room", "svarah_room")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs")
    ap.add_argument("decoded")
    ap.add_argument("--tag", default="")
    ap.add_argument("--norm", choices=["v1", "v2"], default="v1")
    a = ap.parse_args()
    hyp = {}
    for l in open(a.decoded, encoding="utf-8"):
        f, _, t = l.rstrip("\n").partition("\t")
        hyp[os.path.normpath(f)] = t
    agg, missing = {}, 0
    for l in open(a.refs, encoding="utf-8"):
        r = json.loads(l)
        k = os.path.normpath(r["file"])
        if k not in hyp:
            missing += 1; continue
        nz = normalize_v2 if r["set"].startswith("meeting") or a.norm == "v2" else normalize
        s, d, i = edits(nz(r["ref"]), nz(hyp[k]))
        x = agg.setdefault(r["set"], [0, 0, 0, 0])
        x[0] += s; x[1] += d; x[2] += i; x[3] += len(nz(r["ref"]))
    w = {k: 100 * (v[0] + v[1] + v[2]) / max(v[3], 1) for k, v in agg.items()}
    for k in sorted(agg):
        s, d, i, n = agg[k]
        print(f"SCORE{a.tag} {k:24s} {w[k]:6.2f} (S {100 * s / n:5.2f} D {100 * d / n:5.2f} I {100 * i / n:5.2f}) words {n}")
    val = [k for k in w if not k.startswith("meeting")]
    ood = [k for k in val if any(o in k for o in OOD)]
    if val:
        print(f"SCORE_MEAN{a.tag} mean {sum(w[k] for k in val) / len(val):6.2f} over {len(val)} sets | ood-like "
              f"{sum(w[k] for k in ood) / max(len(ood), 1):6.2f} over {len(ood)}")
    print(f"SCORE_DONE{a.tag} missing {missing}")


if __name__ == "__main__":
    main()
