"""100 h English/Hindi ASR training mix, 55/45, from HF parquet shards (no `datasets` dependency). No filtering: sources
are chosen for clean human labels, and voice/asr/qwen_check.py only MEASURES each source's WER against Qwen3-ASR.

    python voice/asr/build_mix.py --out /home/marimo/work/asr/mix100 [--scale 1.0] [--only voxpopuli]

Writes <out>/audio/<source>/<n>.flac (16 kHz mono) and <out>/<source>.jsonl manifests (NeMo style: audio_filepath,
duration, text, lang, source, + scenario when the source has one). Shards are visited in a seeded random order and only
FRAC of each shard's rows are kept, so the hours come from many recordings / speakers instead of the first few.
Utterances outside 1-30 s are dropped. IndicVoices is HF-gated: the box needs a token whose account accepted it.
"""
import argparse
import io
import json
import os
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

CONV = "refs/convert/parquet"
# name, repo, revision, shard prefix, audio column, text column, lang, hours. EN 55 h / HI 45 h.
# Qwen3-ASR audit (300 rows each): VoxPopuli 6.4%, People's Speech 8.6% (mostly number spelling), AMI 9.4% WER.
# Shrutilipi-hi was dropped: 27.8% WER, 11% of rows carry a different news sentence than the audio.
SOURCES = [
    ("peoples_speech", "MLCommons/peoples_speech", CONV, "clean/train/", "audio", "text", "en", 30),
    ("voxpopuli", "facebook/voxpopuli", CONV, "en/train/", "audio", "raw_text", "en", 18),   # accented, cased + punct
    ("ami_ihm", "edinburghcstr/ami", CONV, "ihm/train/", "audio", "text", "en", 7),          # meetings, close-talk
    ("indicvoices_hi", "ai4bharat/IndicVoices", "main", "hindi/train-", "audio_filepath", "text", "hi", 45),  # conv. / extempore / read
]
FRAC = 0.25
MIN_S, MAX_S = 1.0, 30.0


def shard_rows(repo, rev, path, audio_col, text_col, rng):
    local = hf_hub_download(repo, path, repo_type="dataset", revision=rev)
    cols = pq.read_schema(local).names
    t = pq.read_table(local, columns=[audio_col, text_col] + (["scenario"] if "scenario" in cols else [])).to_pylist()
    os.remove(os.path.realpath(local))                  # keep the HF cache from filling the disk
    return [r for r in t if rng.random() < FRAC]


def build(src, out, scale, seed):
    name, repo, rev, prefix, audio_col, text_col, lang, hours = src
    budget = hours * scale * 3600
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
                   if f.startswith(prefix) and f.endswith(".parquet"))
    rng = random.Random(seed)
    rng.shuffle(files)
    adir = os.path.join(out, "audio", name)
    os.makedirs(adir, exist_ok=True)
    total, n, skipped = 0.0, 0, 0
    with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as man, ThreadPoolExecutor(4) as pool:
        jobs = iter(enumerate(files))
        window = deque(pool.submit(shard_rows, repo, rev, f, audio_col, text_col, random.Random(seed + i)) for i, f in islice(jobs, 4))
        while window:                                    # at most 4 shards in flight / in RAM
            rows = window.popleft().result()
            for i, f in islice(jobs, 1):
                window.append(pool.submit(shard_rows, repo, rev, f, audio_col, text_col, random.Random(seed + i)))
            for r in rows:
                text = (r[text_col] or "").strip()
                x, sr = sf.read(io.BytesIO(r[audio_col]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                dur = len(x) / sr
                if sr != 16000 or not text or not MIN_S <= dur <= MAX_S:
                    skipped += 1
                    continue
                p = os.path.join(adir, f"{n:07d}.flac")
                sf.write(p, x, sr)
                man.write(json.dumps({"audio_filepath": p, "duration": round(dur, 3), "text": text, "lang": lang,
                                      "source": name}, ensure_ascii=False) + "\n")
                total += dur
                n += 1
                if total >= budget:
                    break
            print(f"{name}: {total / 3600:7.1f} / {budget / 3600:.0f} h  {n} utts  {skipped} skipped", flush=True)
            if total >= budget:
                break
    return name, total / 3600, n


def main():
    global FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every hour budget (0.01 = smoke test)")
    ap.add_argument("--only", default="", help="comma list of source names")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--frac", type=float, default=FRAC, help="share of each shard kept (lower = more recordings per hour)")
    a = ap.parse_args()
    FRAC = a.frac
    srcs = [s for s in SOURCES if not a.only or s[0] in a.only.split(",")]
    res = [build(s, a.out, a.scale, a.seed) for s in srcs]
    hrs = {"en": 0.0, "hi": 0.0}
    for s, (_, h, _n) in zip(srcs, res):
        hrs[s[6]] += h
    tot = sum(hrs.values()) or 1
    print("MIX", {k: f"{v:.1f} h ({100 * v / tot:.0f}%)" for k, v in hrs.items()}, flush=True)


if __name__ == "__main__":
    main()
