"""build_mix.py output -> train / val manifests with ONE text convention + balanced tokenizer text.

    python voice/asr/prep_train.py --mix /home/marimo/work/asr/mix100 --out /home/marimo/work/asr/run0

v0 convention: lowercase, no punctuation (only VoxPopuli has casing + punctuation, AMI is UPPERCASE; mixing them would
teach the model random casing). Apostrophes inside English words are kept ("don't"). Devanagari is kept as written.
ponytail: punctuation + casing come back as a later stage once more cased data (VoxPopuli, IndicVoices verbatim) is in.
Val = 2% of each source (max 400 rows), seeded. tokenizer.txt = Hindi and English text balanced 50/50 by characters.
"""
import argparse
import glob
import json
import os
import random
import re
import unicodedata

KEEP = re.compile(r"[^a-z0-9'ऀ-ॿ ]")      # Latin, digits, apostrophe, Devanagari block


def clean(text):
    t = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    t = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c for c in t)
    t = KEEP.sub(" ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)     # quote marks, not apostrophes
    return " ".join(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(23)
    train, val = [], []
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        for r in rows:
            r["text"] = clean(r["text"])
        rows = [r for r in rows if r["text"]]
        rng.shuffle(rows)
        k = min(400, len(rows) // 50)
        val += rows[:k]
        train += rows[k:]
        print(f"{os.path.basename(man)}: train {len(rows) - k}  val {k}", flush=True)
    write = lambda name, rs: open(os.path.join(a.out, name), "w", encoding="utf-8").writelines(
        json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    write("train.jsonl", train)
    write("val.jsonl", val)
    for lang in ("en", "hi"):
        write(f"val_{lang}.jsonl", [r for r in val if r["lang"] == lang])
    hi = [r["text"] for r in train if r["lang"] == "hi"]
    en = [r["text"] for r in train if r["lang"] == "en"]
    budget = min(sum(map(len, hi)), sum(map(len, en)))
    pick = lambda xs: [x for x, c in zip(xs, __import__("itertools").accumulate(map(len, xs))) if c <= budget]
    open(os.path.join(a.out, "tokenizer.txt"), "w", encoding="utf-8").write("\n".join(pick(hi) + pick(en)) + "\n")
    hrs = lambda rs, l: sum(r["duration"] for r in rs if r["lang"] == l) / 3600
    print(f"train en {hrs(train, 'en'):.1f} h / hi {hrs(train, 'hi'):.1f} h; val {len(val)} rows; tokenizer text "
          f"{budget} chars per language", flush=True)


if __name__ == "__main__":
    assert clean("Don't STOP, “it's” 1990 — ok!") == "don't stop it's 1990 ok"
    assert clean("कल की Meeting cancel हो गई।") == "कल की meeting cancel हो गई"
    main()
