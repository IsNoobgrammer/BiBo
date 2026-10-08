"""build_mix.py output -> train / val manifests with ONE text convention, speaker-disjoint val, repeats, multi-speaker
windows, balanced tokenizer text.

    python voice/asr/prep_train.py --mix /home/marimo/work/asr/mix1000 --out /home/marimo/work/asr/run2 --variant nptel50

Text v0: lowercase, no punctuation (sources disagree on casing: AMI UPPERCASE, VoxPopuli / Svarah cased). Apostrophes
inside English words stay. Vaani markup: `फोन {phone}` keeps the Latin form (our code-switch convention: English words
in Latin), `पे {पर}` keeps what was said; <tags> and [events] are dropped; rows marked unintelligible are dropped.
ponytail: punctuation + casing return as a later stage once more cased data is in.
Val = whole SPEAKERS per source (2%, 10% for the small accent sets Svarah / Lahaja), never repeated, never in a
multi-speaker window. Train rows are repeated per build_mix.SOURCES[*].repeat: copy 0 is the original, every further
copy is a perturbed rendering (augment.py: speed, reverb, babble / coloured noise, gain). multispk.py adds <spk> windows.
Sources marked windows_only (AMI) enter training ONLY inside full-meeting windows: a single-speaker AMI row labels one
speaker while the others are audible, which taught run1 to drop other people's "okay" / "yes".
--variant picks the English pools (emilia / nptel) by hours with pool_split(), the same split push_mix.py uses.
"""
import argparse
import glob
import itertools
import json
import os
import random
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mix import SOURCES  # noqa: E402
import multispk  # noqa: E402
import augment  # noqa: E402
from functools import partial  # noqa: E402
from multiprocessing import Pool  # noqa: E402

KEEP = re.compile(r"[^a-z0-9'ऀ-ॿ ]")      # Latin, digits, apostrophe, Devanagari block
ANNOT = {"noise", "pause", "breathing", "inhaling", "unintelligible"}
VAL_SHARE = {"svarah": 0.10, "lahaja": 0.10}
REPEAT = {s["name"]: s.get("repeat", 1) for s in SOURCES}
WINDOWS_ONLY = {s["name"] for s in SOURCES if s.get("windows_only")}
# English variants: hours taken from each pool (None = all of it). core = first N h of pool_split, shared by both.
VARIANTS = {"nptel50": {"emilia": None, "nptel": 50}, "nptel150": {"emilia": 100, "nptel": None}, "all": {}}
MULTISPK_H = {"ami": None, "hi": 25, "cs": 20, "bc": 30}      # None = every AMI window available


def pool_split(rows, hours):
    """Deterministic (core, rest): rows in a fixed shuffled order, core = the first `hours` of audio."""
    rows = sorted(rows, key=lambda r: r["audio_filepath"])
    random.Random(7).shuffle(rows)
    core, t = [], 0.0
    for r in rows:
        if hours is not None and t >= hours * 3600:
            break
        core.append(r)
        t += r["duration"]
    return core, rows[len(core):]


def brace(m):
    said, alt = m.group(1), m.group(2).strip()
    latin = re.search("[A-Za-z]", alt) and alt.lower() not in ANNOT
    return alt if latin else said


def clean(text):
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"(\S+)\s*\{([^{}]*)\}", brace, t)                  # Vaani: spoken {standard / English spelling}
    t = re.sub(r"<[^<>]*>|\[[^\[\]]*\]|\{[^{}]*\}", " ", t)         # <tags>, [events], leftover braces
    t = t.lower().replace("’", "'")
    t = "".join(" " if unicodedata.category(c).startswith("P") and c != "'" else c for c in t)
    t = KEEP.sub(" ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)                  # quote marks, not apostrophes
    return " ".join(t.split())


def _copy(row, k, out_dir, babble):
    return augment.make_copy(row, k, out_dir, babble)


def split_by_speaker(rows, share, rng):
    spk = sorted({r["speaker"] for r in rows})
    rng.shuffle(spk)
    val_spk, n = set(), 0
    for s in spk:                                                   # whole speakers until the share is reached
        if n >= share * len(rows):
            break
        val_spk.add(s)
        n += sum(r["speaker"] == s for r in rows)
    return [r for r in rows if r["speaker"] not in val_spk], [r for r in rows if r["speaker"] in val_spk]


def write_val_sets(val, out, rng):
    """One manifest per source (val_<source>.jsonl -> W&B val/<source>/*) + val_multispk (speaker-turn windows built
    from val rows only, so no train audio leaks in)."""
    by_src = {}
    for r in val:
        by_src.setdefault(r["source"], []).append(r)
    for src, rs in by_src.items():
        with open(os.path.join(out, f"val_{src}.jsonl"), "w", encoding="utf-8") as fo:
            fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    ms = multispk.make(val, os.path.join(out, "multispk_val"), {"ami": 0.5, "hi": 0.3, "cs": 0.3, "bc": 0.4}, rng)
    with open(os.path.join(out, "val_multispk.jsonl"), "w", encoding="utf-8") as fo:
        fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ms)
    print(f"val sets: {sorted(by_src)} + multispk ({len(ms)} windows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="all")
    ap.add_argument("--val_only", action="store_true", help="only (re)write the per-source val sets from out/val.jsonl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(23)
    if a.val_only:
        write_val_sets([json.loads(l) for l in open(os.path.join(a.out, "val.jsonl"), encoding="utf-8")], a.out, rng)
        return
    base, val = [], []
    pool = VARIANTS[a.variant]
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        if man.endswith("_raw.jsonl"):                            # pre-filter copy kept by qwen_check --keep_below
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        rows = [r for r in rows if "unintelligible" not in r["text"].lower()]
        for r in rows:
            r["text"] = clean(r["text"])
        rows = [r for r in rows if r["text"]]
        src = rows[0]["source"]
        if src in pool:                                            # same split as push_mix.py (before the val split)
            rows = pool_split(rows, pool[src])[0]
        tr, va = split_by_speaker(rows, VAL_SHARE.get(src, 0.02), rng)
        base += tr
        val += va
        print(f"{src}: train {len(tr)} ({sum(r['duration'] for r in tr) / 3600:.1f} h) x{REPEAT.get(src, 1)}"
              f"{' windows only' if src in WINDOWS_ONLY else ''}  val {len(va)} ({len({r['speaker'] for r in va})} speakers)",
              flush=True)
    single = [r for r in base if r["source"] not in WINDOWS_ONLY]      # rows trained on as-is
    jobs = [(r, k) for r in single for k in range(1, REPEAT.get(r["source"], 1))]
    babble = [r["audio_filepath"] for r in rng.sample(base, min(3000, len(base)))]
    adir = os.path.join(a.out, "aug")
    os.makedirs(adir, exist_ok=True)
    with Pool(16) as pool:
        copies = pool.starmap(partial(_copy, out_dir=adir, babble=babble), jobs, chunksize=64)
    train = single + copies
    print(f"augmented copies: {len(copies)} rows, {sum(r['duration'] for r in copies) / 3600:.1f} h", flush=True)
    ms = multispk.make(base, os.path.join(a.out, "multispk"), MULTISPK_H, rng)
    train += ms
    rng.shuffle(train)
    write = lambda name, rs: open(os.path.join(a.out, name), "w", encoding="utf-8").writelines(
        json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    write("train.jsonl", train)
    write("val.jsonl", val)
    for lang in ("en", "hi"):
        write(f"val_{lang}.jsonl", [r for r in val if r["lang"] == lang])
    write_val_sets(val, a.out, random.Random(24))
    hi = [r["text"] for r in single if r["lang"] == "hi"]
    en = [r["text"] for r in base if r["lang"] == "en"]                # AMI text is fine for the tokenizer
    budget = min(sum(map(len, hi)), sum(map(len, en)))
    pick = lambda xs: [x for x, c in zip(xs, itertools.accumulate(map(len, xs))) if c <= budget]
    open(os.path.join(a.out, "tokenizer.txt"), "w", encoding="utf-8").write("\n".join(pick(hi) + pick(en)) + "\n")
    hrs = lambda rs, key, v: sum(r["duration"] for r in rs if r.get(key) == v) / 3600
    print(f"train: en {hrs(train, 'lang', 'en'):.1f} h / hi {hrs(train, 'lang', 'hi'):.1f} h / multi-speaker "
          f"{hrs(train, 'lang', 'mix'):.1f} h; val {len(val)} rows; tokenizer text {budget} chars per language", flush=True)


if __name__ == "__main__":
    assert clean("Don't STOP, “it's” 1990 — ok!") == "don't stop it's 1990 ok"
    assert clean("कल की Meeting cancel हो गई।") == "कल की meeting cancel हो गई"
    assert clean("<noise>एक फ्रीज़ {fridge} रखा [breathing] है पे {पर} दे {Noise}</noise>") == "एक fridge रखा है पे दे"
    assert clean("<hi-en> दोस्तों bash में nested") == "दोस्तों bash में nested"
    main()
