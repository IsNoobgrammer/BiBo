"""Drop training clips whose reference disagrees with a strong teacher's transcript (teacher_transcribe.py output):
  missing speech : teacher words >= --ratio x reference words AND >= --extra more words (untranscribed speech -> the
                   model learns deletions)
  garbage        : WER(reference vs teacher) > --max_wer with >= --min_words reference words
Both after score.py's normalize_v2 (lowercase, contractions expanded, punctuation and fillers dropped). Writes the clean manifest and a
per-source table (clips / hours dropped by reason), plus a TSV of the dropped rows for inspection.

    python voice/asr/clean_by_teacher.py --manifest en3/train.jsonl --hyp en3/teacher_pk --out en3/train_clean.jsonl
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import normalize_v2 as normalize  # noqa: E402  (v2: fillers dropped both sides)


def words(t):
    n = normalize(t)                                        # score.normalize_v2 returns the word list
    return n if isinstance(n, list) else n.split()


def wer_counts(ref, hyp):
    """word-level Levenshtein: (errors, ref words)"""
    r, h = ref, hyp
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        ri = r[i - 1]
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ri != h[j - 1]))
        prev = cur
    return prev[-1], len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp", default=None, help="teacher_transcribe.py --out dir (hyp_*.tsv), rows by manifest index")
    ap.add_argument("--flags", nargs="*", default=None,
                    help="hf_add_columns.py flags.tsv files: rows by '<source>/<file>' key (any manifest built from "
                         "those repos, e.g. en3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop_sources", default=None, help="regex of sources to drop whole, e.g. our synthetic "
                    "augmentation copies '_(room|phone)$' (vitw_* are real re-recordings: keep)")
    ap.add_argument("--drop_empty", action="store_true", help="drop rows whose reference is blank after normalize_v2")
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--extra", type=int, default=5)
    ap.add_argument("--max_wer", type=float, default=0.6)
    ap.add_argument("--min_words", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    if a.flags:                                             # key-based: the per-repo teacher flags (hf_add_columns.py)
        fl = {}
        for p in a.flags:
            for l in open(p, encoding="utf-8"):
                k, mi, ga = l.rstrip("\n").split("\t")[:3]
                fl[k] = (mi == "1", ga == "1")
        keep, by = [], collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])

        def flag(src, fn):
            """re-recorded / far-field variants: the teacher's own WER there measures the TEACHER failing on
            reverberant audio (medical_room 36.9 % / svarah_room 36.1 % 'garbage' vs ~0-2 % on their close-mic
            originals with the same transcripts). Take the close-mic original's flags when it exists; else trust only
            missing_speech (noise makes the teacher hear fewer words, never more)."""
            base = {"ami_sdm": "ami_ihm"}.get(src) or (src.rsplit("_", 1)[0] if src.endswith(("_room", "_phone")) else None)
            if base and f"{base}/{fn}" in fl:
                return fl[f"{base}/{fn}"]
            f = fl.get(f"{src}/{fn}")
            if f is not None and (base or src.startswith("vitw_")):
                return (f[0], False)
            return f

        drop_src = re.compile(a.drop_sources) if a.drop_sources else None
        n_src = n_empty = 0
        h_src = h_empty = 0.0
        for r in rows:
            s = by[r.get("source", "?")]
            s[0] += 1
            s[1] += r["duration"]
            if drop_src and drop_src.search(r.get("source") or ""):   # our own synthetic augmentation copies
                n_src += 1
                h_src += r["duration"]
                continue
            if a.drop_empty and not words(r["text"]):                 # blank / untranscribed reference
                n_empty += 1
                h_empty += r["duration"]
                continue
            f = flag(r.get("source"), os.path.basename(r["audio_filepath"]))
            if f is None:
                s[4] += 1
                keep.append(r)
            elif f[0] or f[1]:
                s[2] += f[0] * r["duration"]
                s[3] += f[1] * r["duration"]
            else:
                keep.append(r)
        with open(a.out, "w", encoding="utf-8") as fo:
            for r in keep:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{'source':<22}{'clips':>8}{'hours':>8}{'missing h':>11}{'garbage h':>11}{'no flag':>9}")
        for src, s in sorted(by.items(), key=lambda kv: -kv[1][1]):
            print(f"{src:<22}{s[0]:>8}{s[1] / 3600:>8.1f}{s[2] / 3600:>11.2f}{s[3] / 3600:>11.2f}{s[4]:>9}")
        print(f"dropped sources /{a.drop_sources}/: {n_src} clips {h_src / 3600:.1f} h; blank references: {n_empty} clips "
              f"{h_empty / 3600:.2f} h")
        print(f"kept {len(keep)} of {len(rows)} clips ({sum(r['duration'] for r in keep) / 3600:.1f} h) -> {a.out}")
        return
    hyp = {}
    for p in glob.glob(os.path.join(a.hyp, "hyp_*.tsv")):
        for l in open(p, encoding="utf-8"):
            i, _, t = l.rstrip("\n").partition("\t")
            hyp[int(i)] = t
    keep, stats = [], collections.defaultdict(lambda: [0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0])
    drop_f = open(a.out + ".dropped.tsv", "w", encoding="utf-8")
    for i, r in enumerate(rows):
        s = stats[r.get("source", "?")]
        s[0] += 1
        s[1] += r["duration"]
        if i not in hyp:                                    # not transcribed (e.g. over max_dur): keep, count
            s[6] += 1
            s[7] += r["duration"]
            keep.append(r)
            continue
        ref, h = words(r["text"]), words(hyp[i])
        missing = len(h) >= a.ratio * max(len(ref), 1) and len(h) - len(ref) >= a.extra
        e, n = wer_counts(ref, h) if not missing else (0, len(ref))
        garbage = not missing and n >= a.min_words and e / n > a.max_wer
        if missing or garbage:
            k = 2 if missing else 4
            s[k] += 1
            s[k + 1] += r["duration"]
            drop_f.write(f"{'missing' if missing else 'garbage'}\t{r.get('source')}\t{r['duration']:.2f}\t{len(ref)}\t{len(h)}"
                         f"\t{r['text'][:160]}\t{hyp[i][:160]}\n")
        else:
            keep.append(r)
    drop_f.close()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{'source':<22}{'clips':>8}{'hours':>8}{'missing h':>11}{'missing %':>10}{'garbage h':>11}{'garbage %':>10}{'untested h':>11}")
    T = [0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0]
    for src, s in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        T = [x + y for x, y in zip(T, s)]
        print(f"{src:<22}{s[0]:>8}{s[1] / 3600:>8.1f}{s[3] / 3600:>11.2f}{100 * s[3] / s[1]:>9.1f}%{s[5] / 3600:>11.2f}"
              f"{100 * s[5] / s[1]:>9.1f}%{s[7] / 3600:>11.2f}")
    print(f"{'ALL':<22}{T[0]:>8}{T[1] / 3600:>8.1f}{T[3] / 3600:>11.2f}{100 * T[3] / T[1]:>9.1f}%{T[5] / 3600:>11.2f}"
          f"{100 * T[5] / T[1]:>9.1f}%{T[7] / 3600:>11.2f}")
    print(f"kept {len(keep)} of {len(rows)} clips -> {a.out}; dropped rows -> {a.out}.dropped.tsv")


if __name__ == "__main__":
    main()
