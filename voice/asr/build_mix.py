"""400 h English/Hindi ASR training mix (HI ~225 : EN ~175 after repeats) from HF parquet shards, no `datasets` dependency.
No filtering: sources are chosen for clean human labels.

    python voice/asr/build_mix.py --out /home/marimo/work/asr/mix400 [--scale 1.0] [--only svarah,lahaja]

Writes <out>/audio/<source>/<n>.flac (16 kHz mono) and <out>/<source>.jsonl manifests (NeMo style: audio_filepath,
duration, text, lang, source, speaker, + scenario / meeting / begin when the source has them). Each source stores its
UNIQUE audio once; `repeat` (small, high-value sets) is applied to the train manifest by prep_train.py. Shards are
visited in a seeded random order and only `frac` of each shard's rows are kept, so the hours come from many recordings
and speakers. Utterances outside 1-30 s are dropped. Gated sets need an HF token whose account accepted them.
"""
import argparse
import io
import json
import os
import random
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

CONV = "refs/convert/parquet"
# hours = unique hours to take (None = everything); repeat = times each train row appears (prep_train.py).
# effective train hours: EN svarah 9.4x2 + people's speech 80 + ami 65 + voxpopuli 10 = ~175
#                        HI indicvoices 60 + vaani 45 + kathbath 40 + hinglish 45 + lahaja 11x3 = ~223
SOURCES = [
    # --- English (Indian accent first) ---
    dict(name="svarah", repo="ai4bharat/Svarah", rev="main", prefix="data/test-", audio="audio_filepath", text="text",
         lang="en", hours=None, frac=1.0, repeat=2,
         speaker=lambda r: f"{r.get('native_place_district')}|{r.get('gender')}|{r.get('age-group')}"),  # no speaker id column
    dict(name="peoples_speech", repo="MLCommons/peoples_speech", rev=CONV, prefix="clean/train/", audio="audio",
         text="text", lang="en", hours=80, frac=0.25, speaker=lambda r: r["id"].rsplit("_", 1)[0]),
    dict(name="voxpopuli", repo="facebook/voxpopuli", rev=CONV, prefix="en/train/", audio="audio", text="raw_text",
         lang="en", hours=10, frac=0.25, speaker=lambda r: r["speaker_id"]),
    dict(name="ami_ihm", repo="edinburghcstr/ami", rev=CONV, prefix="ihm/train/", audio="audio", text="text", lang="en",
         hours=65, frac=1.0, speaker=lambda r: r["speaker_id"],          # whole meetings: real turns for <spk> windows
         extra={"meeting": "meeting_id", "begin": "begin_time"}),
    # --- Hindi ---
    dict(name="indicvoices_hi", repo="ai4bharat/IndicVoices", rev="main", prefix="hindi/train-", audio="audio_filepath",
         text="text", lang="hi", hours=60, frac=0.25, speaker=lambda r: r["speaker_id"], extra={"scenario": "scenario"},
         quota={"Conversation": 25, "Extempore": 25, "Read": 10}),       # conversation boosted over its natural 23%
    dict(name="vaani_hi", repo="psk/vaani-asr", rev="main", prefix="hindi/train-", audio="audio", text="transcript",
         lang="hi", hours=45, frac=0.5, speaker=lambda r: r["file_name"].split("_")[5]),  # spontaneous, loanwords {latin}
    dict(name="kathbath_hi", repo="ai4bharat/Kathbath", rev="main", prefix="hindi/train-", audio="audio_filepath",
         text="text", lang="hi", hours=40, frac=0.5, speaker=lambda r: str(r["speaker_id"])),
    dict(name="hinglish", repo="agarwalayushi/hinglish", rev="main", prefix="data/train-", audio="audio", text="text",
         lang="hi", hours=45, frac=0.1, speaker=lambda r: r["source"]),  # code-switched, English already in Latin
    dict(name="lahaja", repo="ai4bharat/Lahaja", rev="main", prefix="data/test-", audio="audio_filepath", text="text",
         lang="hi", hours=None, frac=1.0, repeat=3, speaker=lambda r: str(r["sp_id"])),  # Hindi from non-native speakers
]
MIN_S, MAX_S = 1.0, 30.0
SPEAKER_COLS = {"native_place_district", "gender", "age-group", "id", "speaker_id", "file_name", "source", "sp_id"}


def shard_rows(src, path, rng):
    local = hf_hub_download(src["repo"], path, repo_type="dataset", revision=src["rev"])
    names = set(pq.read_schema(local).names)
    want = {src["audio"], src["text"]} | (SPEAKER_COLS & names) | (set(src.get("extra", {}).values()) & names)
    t = pq.read_table(local, columns=sorted(want)).to_pylist()
    os.remove(os.path.realpath(local))                  # keep the HF cache from filling the disk
    return [r for r in t if rng.random() < src["frac"]]


def build(src, out, scale, seed):
    name = src["name"]
    budget = src["hours"] * scale * 3600 if src["hours"] else float("inf")
    quota = {k: v * scale * 3600 for k, v in (src.get("quota") or {}).items()}
    files = sorted(f for f in HfApi().list_repo_files(src["repo"], repo_type="dataset", revision=src["rev"])
                   if f.startswith(src["prefix"]) and f.endswith(".parquet"))
    random.Random(seed).shuffle(files)
    adir = os.path.join(out, "audio", name)
    os.makedirs(adir, exist_ok=True)
    total, n, skipped, per_q = 0.0, 0, 0, defaultdict(float)
    full = lambda: total >= budget or (quota and all(per_q[k] >= v for k, v in quota.items()))
    with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as man, ThreadPoolExecutor(4) as pool:
        jobs = iter(enumerate(files))
        window = deque(pool.submit(shard_rows, src, f, random.Random(seed + i)) for i, f in islice(jobs, 4))
        while window:                                    # at most 4 shards in flight / in RAM
            rows = window.popleft().result()
            for i, f in islice(jobs, 1):
                window.append(pool.submit(shard_rows, src, f, random.Random(seed + i)))
            for r in rows:
                text = (r[src["text"]] or "").strip()
                q = r.get("scenario") if quota else None
                if not text or (quota and (q not in quota or per_q[q] >= quota[q])):
                    skipped += 1
                    continue
                try:
                    x, sr = sf.read(io.BytesIO(r[src["audio"]]["bytes"]), dtype="float32")
                except Exception:                            # corrupt / empty audio bytes (seen in psk/vaani-asr)
                    skipped += 1
                    continue
                if x.ndim > 1:
                    x = x.mean(1)
                dur = len(x) / sr
                if sr != 16000 or not MIN_S <= dur <= MAX_S:
                    skipped += 1
                    continue
                p = os.path.join(adir, f"{n:07d}.flac")
                sf.write(p, x, sr)
                row = {"audio_filepath": p, "duration": round(dur, 3), "text": text, "lang": src["lang"], "source": name,
                       "speaker": f"{name}:{src['speaker'](r)}"}
                for k, col in src.get("extra", {}).items():
                    row[k] = r.get(col)
                man.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += dur
                n += 1
                if q:
                    per_q[q] += dur
                if full():
                    break
            print(f"{name}: {total / 3600:7.1f} / {budget / 3600:.0f} h  {n} utts  {skipped} skipped"
                  + (f"  {dict((k, round(v / 3600, 1)) for k, v in per_q.items())}" if quota else ""), flush=True)
            if full():
                break
    return total / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every hour budget (0.01 = smoke test)")
    ap.add_argument("--only", default="", help="comma list of source names")
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    srcs = [s for s in SOURCES if not a.only or s["name"] in a.only.split(",")]
    hrs = defaultdict(float)
    for s in srcs:
        hrs[s["lang"]] += build(s, a.out, a.scale, a.seed) * s.get("repeat", 1)
    print("MIX (effective, with repeats)", {k: f"{v:.1f} h" for k, v in hrs.items()}, flush=True)


if __name__ == "__main__":
    main()
