"""en3 on a fresh box must be the SAME split the old boxes trained / validated on (val WER is only comparable on
identical val sets). Compares the rebuilt /home/marimo/work/asr/en3 (prep_en.py) with the copy pushed to HF
fhai50032/bibo-asr-ckpt en3/ and replaces any differing file with the HF one; then checks that manifest audio exists.
Exit 1 if audio is missing (the manifests point at files this box does not have).

    python voice/asr/en3_verify.py [--dst /home/marimo/work/asr/en3]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

from huggingface_hub import snapshot_download


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="/home/marimo/work/asr/en3")
    a = ap.parse_args()
    src = os.path.join(snapshot_download("fhai50032/bibo-asr-ckpt", allow_patterns=["en3/*"]), "en3")
    diff = []
    for root, _, files in os.walk(src):
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(a.dst, os.path.relpath(s, src))
            if md5(s) != md5(d):
                diff.append(os.path.relpath(s, src))
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy(s, d)
    miss, n = 0, 0
    for f in os.listdir(a.dst):
        if f.endswith(".jsonl"):
            for line in open(os.path.join(a.dst, f)):
                n += 1
                miss += not os.path.exists(json.loads(line)["audio_filepath"])
    print(f"[en3_verify] files replaced with the HF copy: {diff or 'none'} | audio missing: {miss} of {n} rows", flush=True)
    sys.exit(1 if miss else 0)


if __name__ == "__main__":
    main()
