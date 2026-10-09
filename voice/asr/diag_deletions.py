"""Deletion-bias diagnostic: does a model drop words, where, and is it the data or the decoder?

    python voice/asr/diag_deletions.py --nemo exp/run2/run2.nemo --val R/val_*.jsonl E/fleurs_*.jsonl \
        [--max_per_source 400] [--pad 1.0] [--att_context 70 13] [--out diag_run2.json]

Per val set (word-level Levenshtein alignment):
  S / D / I rates      deletions >> insertions = a deletion bias
  D by position        first 10 % / middle / last 10 % of the reference words. End-heavy = the streaming encoder
                       never "flushes" the last words (fix at inference); spread = model / data
  D by speaking rate   words per second buckets: fast-speech deletions = too many tokens per frame
  end-pad A/B          the same audio + `pad` s of silence: if D(last 10 %) falls, it is the flush, not the data
  worst rows           highest deletion count (path, ref, hyp): label rows that miss spoken words show up here
"""
import argparse
import collections
import json
import os
import random



def load_rows(path, n, rng):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rng.sample(rows, n) if n and len(rows) > n else rows


def transcribe(m, rows, pad, bs):
    import numpy as np
    import soundfile as sf
    import torch
    audio = []
    for r in rows:
        x, sr = sf.read(r["audio_filepath"], dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        off, dur = r.get("offset"), r.get("duration")
        if off is not None:
            x = x[int(off * sr): int((off + dur) * sr)]
        if pad:
            x = np.concatenate([x, np.zeros(int(pad * sr), np.float32)])
        audio.append(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m.transcribe(audio, batch_size=bs, verbose=False)
    out = out[0] if isinstance(out, tuple) else out
    return [h.text if hasattr(h, "text") else h for h in out]


def align(ref, hyp):
    """Levenshtein over words -> (S, D, I, deleted ref indices). Ties prefer substitution, like jiwer."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    s = de = ins = 0
    dels = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            de += 1
            dels.append(i - 1)
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, de, ins, dels


def analyse(rows, hyps):
    acc = collections.Counter()
    pos = collections.Counter()
    rate_d, rate_n = collections.Counter(), collections.Counter()
    worst = []
    for r, h in zip(rows, hyps):
        ref = r["text"].split()
        if not ref:
            continue
        s, de, ins, dels = align(ref, h.split())
        acc["N"] += len(ref)
        acc["S"] += s
        acc["D"] += de
        acc["I"] += ins
        nd = de
        for i in dels:
            f = i / len(ref)
            pos["first10" if f < 0.1 else "last10" if f >= 0.9 else "middle"] += 1
        wps = len(ref) / max(r["duration"], 1e-3)
        b = "<1.5" if wps < 1.5 else "1.5-2.5" if wps < 2.5 else "2.5-3.5" if wps < 3.5 else ">3.5"
        rate_d[b] += nd
        rate_n[b] += len(ref)
        worst.append((nd, r["audio_filepath"], r["text"], h))
    worst.sort(key=lambda x: -x[0])
    return acc, pos, rate_d, rate_n, worst[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--max_per_source", type=int, default=400)
    ap.add_argument("--pad", type=float, default=1.0, help="seconds of trailing silence for the end-pad A/B (0 = off)")
    ap.add_argument("--att_context", type=int, nargs=2, default=None, help="e.g. 70 0 (no look-ahead) or 70 13")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--blank_penalty", type=float, default=0.0, help="RNN-T decode: blank logit -= this (blank_penalty.py)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(a.nemo).cuda().eval()
    if a.att_context:
        m.encoder.set_default_att_context_size(list(a.att_context))
    if a.blank_penalty:
        import blank_penalty
        blank_penalty.apply(a.blank_penalty)
    tot = collections.Counter()
    rng = random.Random(0)
    report = {}
    print(f"{'set':18s} {'words':>6s} {'WER':>6s} {'S':>6s} {'D':>6s} {'I':>6s} | D first10/mid/last10 "
          f"| D last10 with pad | D rate by words/s (<1.5, 1.5-2.5, 2.5-3.5, >3.5)", flush=True)
    for p in a.val:
        stem = os.path.basename(p)[:-6]
        if stem in ("val_en", "val_hi", "val_multispk"):
            continue
        rows = load_rows(p, a.max_per_source, rng)
        acc, pos, rd, rn, worst = analyse(rows, transcribe(m, rows, 0, a.bs))
        N = max(acc["N"], 1)
        pad_last = None
        if a.pad:
            _, ppos, *_ = analyse(rows, transcribe(m, rows, a.pad, a.bs))
            pad_last = ppos["last10"] / N
        tot.update(acc)
        rates = [rd[b] / rn[b] if rn[b] else float("nan") for b in ("<1.5", "1.5-2.5", "2.5-3.5", ">3.5")]
        print(f"{stem:18s} {acc['N']:6d} {(acc['S'] + acc['D'] + acc['I']) / N:6.3f} {acc['S'] / N:6.3f} "
              f"{acc['D'] / N:6.3f} {acc['I'] / N:6.3f} | {pos['first10'] / N:.3f}/{pos['middle'] / N:.3f}/"
              f"{pos['last10'] / N:.3f} | {pad_last if pad_last is None else round(pad_last, 3)} | "
              + " ".join(f"{x:.3f}" for x in rates), flush=True)
        report[stem] = {"words": acc["N"], "S": acc["S"] / N, "D": acc["D"] / N, "I": acc["I"] / N,
                        "D_pos": {k: v / N for k, v in pos.items()}, "D_last10_padded": pad_last,
                        "D_by_wps": dict(zip(("<1.5", "1.5-2.5", "2.5-3.5", ">3.5"), rates)),
                        "worst": [{"deleted": d, "audio": f, "ref": r, "hyp": h} for d, f, r, h in worst]}
    T = max(tot["N"], 1)
    print(f"{'TOTAL':18s} {tot['N']:6d} {(tot['S'] + tot['D'] + tot['I']) / T:6.3f} {tot['S'] / T:6.3f} "
          f"{tot['D'] / T:6.3f} {tot['I'] / T:6.3f}   (blank penalty {a.blank_penalty:g})", flush=True)
    if a.out:
        json.dump(report, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {a.out} (per-set numbers + 10 worst rows each)", flush=True)


if __name__ == "__main__":
    main()
