"""English-only dataset straight from the English mix: the recipe for every English (re)build.

    python voice/asr/fetch_mix.py --repo fhai50032/asr-english --mix /home/marimo/work/asr/mix_en
    python voice/asr/prep_en.py --mix /home/marimo/work/asr/mix_en --out /home/marimo/work/asr/en2 --vocab 2047

Same text convention, speaker-disjoint val shares, repeats and AMI meeting windows as prep_train.py, but every random
choice has its OWN generator seeded by (seed, purpose, source): adding, dropping or reordering other data (Hindi, new
English sources) never moves an existing source's val speakers. A mix dir that also holds Hindi manifests is fine:
only lang == "en" rows (no Devanagari) are read. Tokenizer: English SentencePiece BPE on the training text, <spk1..4>.
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multispk  # noqa: E402
from prep_train import REPEAT, VAL_SHARE, WINDOWS_ONLY, _copy, clean, split_by_speaker  # noqa: E402

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=2047, help="BPE pieces (+ blank = 2048 classes)")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--nemo", default="/home/marimo/work/NeMo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = lambda *key: random.Random(":".join(map(str, (a.seed,) + key)))     # str seeds are deterministic
    base, val = [], []
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        if man.endswith("_raw.jsonl"):
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        rows = [r for r in rows if r.get("lang") == "en" and "unintelligible" not in r["text"].lower()]
        for r in rows:
            r["text"] = clean(r["text"])
        rows = [r for r in rows if r["text"] and not DEVANAGARI.search(r["text"])]
        if not rows:
            continue
        src = rows[0]["source"]
        tr, va = split_by_speaker(rows, VAL_SHARE.get(src, 0.02), rng("val", src))
        base += tr
        val += va
        print(f"{src}: train {len(tr)} ({sum(r['duration'] for r in tr) / 3600:.1f} h) x{REPEAT.get(src, 1)}"
              f"{' windows only' if src in WINDOWS_ONLY else ''}  val {len(va)} ({len({r['speaker'] for r in va})} speakers)",
              flush=True)
    single = [r for r in base if r["source"] not in WINDOWS_ONLY]
    jobs = [(r, k) for r in single for k in range(1, REPEAT.get(r["source"], 1))]
    copies = []
    if jobs:                                   # ponytail: every source is x1 today, so normally no audio is written
        babble = [r["audio_filepath"] for r in rng("babble").sample(base, min(3000, len(base)))]
        os.makedirs(os.path.join(a.out, "aug"), exist_ok=True)
        with Pool(16) as p:
            copies = p.starmap(partial(_copy, out_dir=os.path.join(a.out, "aug"), babble=babble), jobs, chunksize=64)
    no_sim = {"hi": 0, "cs": 0, "bc": 0}       # English-only: real AMI meeting windows, no simulated mixes
    train = single + copies + multispk.make(base, os.path.join(a.out, "multispk"), {"ami": None, **no_sim}, rng("multispk"))
    rng("shuffle").shuffle(train)
    write = lambda name, rs: open(os.path.join(a.out, name), "w", encoding="utf-8").writelines(
        json.dumps(r, ensure_ascii=False) + "\n" for r in rs)
    write("train.jsonl", train)
    by_src = {}
    for r in val:
        by_src.setdefault(r["source"], []).append(r)
    for src, rs in by_src.items():
        write(f"val_{src}.jsonl", rs)
    # tokenizer text: the training transcripts once each, <spk> tags out
    seen = dict.fromkeys(re.sub(r"<spk\d>", " ", r["text"]).strip() for r in train)
    open(os.path.join(a.out, "tokenizer.txt"), "w", encoding="utf-8").write("\n".join(t for t in seen if t) + "\n")
    tok = os.path.join(a.out, "tok")
    if not os.path.isdir(os.path.join(tok, f"tokenizer_spe_bpe_v{a.vocab}")):
        subprocess.run([sys.executable, os.path.join(a.nemo, "scripts/tokenizers/process_asr_text_tokenizer.py"),
                        "--data_file", os.path.join(a.out, "tokenizer.txt"), "--data_root", tok,
                        "--vocab_size", str(a.vocab), "--tokenizer", "spe", "--spe_type", "bpe", "--no_lower_case",
                        "--spe_user_defined_symbols", "<spk1>", "<spk2>", "<spk3>", "<spk4>"], check=True)
    by = {}
    for r in train:
        by[r.get("source")] = by.get(r.get("source"), 0) + r["duration"] / 3600
    print(f"{os.path.basename(a.out)}: {len(train)} rows, {sum(by.values()):.1f} h "
          f"({', '.join(f'{k} {v:.1f}' for k, v in sorted(by.items()))}); val sets {sorted(by_src)}; "
          f"tokenizer {tok}/tokenizer_spe_bpe_v{a.vocab}", flush=True)


if __name__ == "__main__":
    main()
