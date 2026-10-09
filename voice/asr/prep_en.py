"""English-only dataset from a prep_train.py build (run2): realtime English ASR first, Hindi later.

    python voice/asr/prep_en.py --src /home/marimo/work/asr/run2 --out /home/marimo/work/asr/en1 [--vocab 1024]

Keeps: every lang == "en" row + the AMI multi-speaker windows (English meetings, <spk> tags). Drops every Hindi row and
the Hindi / code-switch / bilingual windows (multispk_hi / _cs / _bc). Val: only the English per-source sets (+ FLEURS
en is passed separately). Tokenizer: English SentencePiece BPE trained on the kept training text, the <spk1..4>
symbols kept, no Devanagari (so no script lock is needed downstream).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

KEEP_SOURCES = {"multispk_ami"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def keep(r):
    return (r["lang"] == "en" or r.get("source") in KEEP_SOURCES) and not DEVANAGARI.search(r["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=1024, help="BPE size (Nemotron / NVIDIA English models use 1024)")
    ap.add_argument("--nemo", default="/home/marimo/work/NeMo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(a.src, "train.jsonl"), encoding="utf-8")]
    train = [r for r in rows if keep(r)]
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
    for p in sorted(glob.glob(os.path.join(a.src, "val_*.jsonl"))):
        vr = [json.loads(l) for l in open(p, encoding="utf-8")]
        stem = os.path.basename(p)
        if stem not in ("val_en.jsonl", "val_hi.jsonl", "val_multispk.jsonl") and vr and all(r["lang"] == "en" for r in vr):
            shutil.copy(p, os.path.join(a.out, stem))
    # tokenizer text: the training transcripts once each (copies / perturbed repeats share a text), <spk> tags out
    seen = dict.fromkeys(re.sub(r"<spk\d>", " ", r["text"]).strip() for r in train)
    open(os.path.join(a.out, "tokenizer.txt"), "w", encoding="utf-8").write("\n".join(t for t in seen if t) + "\n")
    tok = os.path.join(a.out, "tok")
    if not os.path.isdir(os.path.join(tok, f"tokenizer_spe_bpe_v{a.vocab}")):
        subprocess.run([sys.executable, os.path.join(a.nemo, "scripts/tokenizers/process_asr_text_tokenizer.py"),
                        "--data_file", os.path.join(a.out, "tokenizer.txt"), "--data_root", tok,
                        "--vocab_size", str(a.vocab), "--tokenizer", "spe", "--spe_type", "bpe", "--no_lower_case",
                        "--spe_user_defined_symbols", "<spk1>", "<spk2>", "<spk3>", "<spk4>"], check=True)
    hrs = sum(r["duration"] for r in train) / 3600
    by = {}
    for r in train:
        by[r.get("source")] = by.get(r.get("source"), 0) + r["duration"] / 3600
    print(f"en1: {len(train)} rows, {hrs:.1f} h ({', '.join(f'{k} {v:.1f}' for k, v in sorted(by.items()))}); "
          f"val sets {sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.out, 'val_*.jsonl')))}; "
          f"tokenizer {tok}/tokenizer_spe_bpe_v{a.vocab}", flush=True)


if __name__ == "__main__":
    main()
