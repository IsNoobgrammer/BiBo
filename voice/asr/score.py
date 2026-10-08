"""WER of ASR outputs against references, with the RT Captions normalizer (lowercase, punctuation stripped, fillers
dropped). Works for Devanagari: Unicode punctuation (category P*) is removed, letters / matras / digits are kept.

    python voice/asr/score.py --ref eval/reference.txt --hyp out/a.json [--hyp out/b.json ...]

A hyp is a .txt (whole transcript) or a NeMo output manifest (.json lines with "pred_text"; lines are joined).
"""
import argparse
import json
import re
import unicodedata

FILLERS = {"uh", "um", "uhm", "umm", "hmm", "hm", "mm", "mhm", "ah", "er", "erm"}


def normalize(text):
    text = re.sub(r"<[^>]+>", " ", text)                     # language / speaker / eou tags
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.lower())
    return [w for w in text.split() if w not in FILLERS]


def read_hyp(path):
    if path.endswith(".txt"):
        return open(path, encoding="utf-8").read()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return " ".join(r.get("pred_text", "") for r in rows)


def wer(ref, hyp):
    """Word error rate by edit distance (no external dependency)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", action="append", required=True)
    a = ap.parse_args()
    ref = normalize(open(a.ref, encoding="utf-8").read())
    for h in a.hyp:
        hw = normalize(read_hyp(h))
        print(f"WER {100 * wer(ref, hw):6.2f}%  ref {len(ref)} words  hyp {len(hw)} words  {h}")


if __name__ == "__main__":
    assert wer(["a", "b", "c"], ["a", "x", "c", "d"]) == 2 / 3
    assert normalize("Hello, Uh world! <en-US> आँखें पीली हैं।") == ["hello", "world", "आँखें", "पीली", "हैं"]
    main()
