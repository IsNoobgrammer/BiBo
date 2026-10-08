"""1000 h English/Hindi ASR training superset (HI 600 : EN 400 per variant) from HF parquet shards, no `datasets`.
Bad rows are REMOVED, never relabelled: each source is human-verified or kept only where its own quality signal says the
label matches the audio (Emilia: two independent transcripts agree + audio quality; Numo: its own WER / CER /
synthetic-speech score; NPTEL: qwen_check.py --keep_below after the build).

    python voice/asr/build_mix.py --out /home/marimo/work/asr/mix1000 [--scale 0.01] [--only emilia,numo_hi]

Writes <out>/audio/<source>/<n>.flac (16 kHz mono) + <out>/<source>.jsonl (audio_filepath, duration, text, lang,
source, speaker [+ scenario / meeting / begin]). Unique audio only; `repeat` and `windows_only` are applied by
prep_train.py, which also cuts the two English variants (nptel50 / nptel150) from the emilia + nptel pools.
The 400 h mix (run1) is in git history + the manifests in HF fhai50032/asr-en-hi-400h.
"""
import argparse
import csv
import io
import json
import os
import random
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import normalize, wer  # noqa: E402

CONV = "refs/convert/parquet"
SR = 16000


def emilia_ok(r):
    """'Very good' Emilia: ElevenLabs Scribe and Emilia's own transcript agree within 5% WER, audio quality PQ >= 6.5
    (~34% of rows, ~204 h of 605 h on a 2-shard sample)."""
    a, b = normalize(r.get("text_emilia") or ""), normalize(r.get("text_scribe") or "")
    return bool(a) and wer(a, b) <= 0.05 and (r.get("PQ") or 0) >= 6.5


def numo_ok(r):
    """Numo's own validation: per-recording WER <= 6% (median 4%) and synthetic-speech suspicion <= 0.2."""
    return float(r.get("wer") or 1) <= 0.06 and float(r.get("synthetic_suspicion_score") or 1) <= 0.2


# hours = unique hours to take (None = all); repeat = train copies (copy 1+ augmented, prep_train.py)
# EN per variant ~400: [emilia + nptel pools: nptel50 = emilia ~200 + nptel 50, nptel150 = emilia 100 + nptel ~150]
#   + ami windows ~80 + spotify + phone + voxpopuli 20 + people's speech 15 + svarah ~9 (x2) + medical ~8
# HI 600: indicvoices 300 + numo 120 + vaani 100 + kathbath 50 + hinglish 20 + lahaja ~11 (x2)
SOURCES = [
    # --- English pools for the two variants ---
    dict(name="emilia", repo="MrDragonFox/EN_Emilia_Yodas_616h", rev="main", prefix="data/train-", audio="audio",
         text="text_scribe", lang="en", hours=210, frac=1.0, keep=emilia_ok, speaker=lambda r: r["file_id"].rsplit("_W", 1)[0],
         cols=["text_emilia", "PQ"]),                                       # spontaneous YouTube English, cased + punct
    dict(name="nptel", repo="skbose/indian-english-nptel-v0", rev="main", prefix="data/train-", audio="audio",
         text="transcription_normalised", lang="en", hours=170, frac=0.15, speaker=lambda r: r["speaker_name"]),
    # --- English, shared ---
    dict(name="ami_ihm", repo="edinburghcstr/ami", rev=CONV, prefix="ihm/train/", audio="audio", text="text", lang="en",
         hours=None, frac=1.0, windows_only=True, speaker=lambda r: r["speaker_id"],
         extra={"meeting": "meeting_id", "begin": "begin_time"}),           # ONLY as full-meeting <spk> windows
    dict(name="spotify", loader="spotify", repo="SALT-NLP/spotify_podcast_ASR", lang="en", hours=None, max_s=60,
         speaker=lambda r: r["filename"]),                                  # human verbatim podcast talk, 2-3 speakers
    dict(name="phone", repo="sawradip/phone-asr-data", rev="main", prefix="data/", audio="audio", text="transcription",
         lang="en", hours=None, frac=1.0, speaker=lambda r: r["filename"].split("-")[0]),  # phone chat, "yeah"s
    dict(name="voxpopuli", repo="facebook/voxpopuli", rev=CONV, prefix="en/train/", audio="audio", text="raw_text",
         lang="en", hours=20, frac=0.25, speaker=lambda r: r["speaker_id"]),
    dict(name="peoples_speech", repo="MLCommons/peoples_speech", rev=CONV, prefix="clean/train/", audio="audio",
         text="text", lang="en", hours=15, frac=0.1, speaker=lambda r: r["id"].rsplit("_", 1)[0]),
    dict(name="svarah", repo="ai4bharat/Svarah", rev="main", prefix="data/test-", audio="audio_filepath", text="text",
         lang="en", hours=None, frac=1.0, repeat=2,
         speaker=lambda r: f"{r.get('native_place_district')}|{r.get('gender')}|{r.get('age-group')}"),
    dict(name="medical", repo="yashtiwari/PaulMooney-Medical-ASR-Data", rev="main", prefix="data/", audio="path",
         text="sentence", lang="en", hours=None, frac=1.0, speaker=lambda r: str(r["speaker_id"])),
    # --- Hindi ---
    dict(name="indicvoices_hi", repo="ai4bharat/IndicVoices", rev="main", prefix="hindi/train-", audio="audio_filepath",
         text="text", lang="hi", hours=300, frac=1.0, speaker=lambda r: r["speaker_id"], extra={"scenario": "scenario"},
         quota={"Conversation": 120, "Extempore": 130, "Read": 50}),       # conversation is only ~147 h of ~640 h
    dict(name="numo_hi", repo="psdn-ai/numo-indic-speech", rev="main", prefix="data/hindi/", audio="audio",
         text="transcript", lang="hi", hours=120, frac=0.5, keep=numo_ok, max_s=70, speaker=lambda r: r["speaker_id"],
         cols=["wer", "synthetic_suspicion_score"]),                         # read, 36-67 s, 48 kHz stereo
    dict(name="vaani_hi", repo="psk/vaani-asr", rev="main", prefix="hindi/train-", audio="audio", text="transcript",
         lang="hi", hours=100, frac=0.9, speaker=lambda r: r["file_name"].split("_")[5]),
    dict(name="kathbath_hi", repo="ai4bharat/Kathbath", rev="main", prefix="hindi/train-", audio="audio_filepath",
         text="text", lang="hi", hours=50, frac=0.5, speaker=lambda r: str(r["speaker_id"])),
    dict(name="hinglish", repo="agarwalayushi/hinglish", rev="main", prefix="data/train-", audio="audio", text="text",
         lang="hi", hours=20, frac=0.05, speaker=lambda r: r["source"]),  # spoken tutorials (= MUCS; one copy only)
    dict(name="lahaja", repo="ai4bharat/Lahaja", rev="main", prefix="data/test-", audio="audio_filepath", text="text",
         lang="hi", hours=None, frac=1.0, repeat=2, speaker=lambda r: str(r["sp_id"])),
]
MIN_S, MAX_S = 1.0, 30.0
SPEAKER_COLS = {"native_place_district", "gender", "age-group", "id", "speaker_id", "file_name", "source", "sp_id",
                "speaker_name", "file_id", "filename"}


def shard_rows(src, path, rng):
    local = hf_hub_download(src["repo"], path, repo_type="dataset", revision=src["rev"])
    names = set(pq.read_schema(local).names)
    want = ({src["audio"], src["text"]} | (SPEAKER_COLS & names) | (set(src.get("extra", {}).values()) & names)
            | (set(src.get("cols", [])) & names))
    t = pq.read_table(local, columns=sorted(want)).to_pylist()
    os.remove(os.path.realpath(local))                  # keep the HF cache from filling the disk
    return [r for r in t if rng.random() < src["frac"]]


def spotify_rows():
    """SALT-NLP/spotify_podcast_ASR: .ogg clips + metadata.csv (human transcription per clip)."""
    for attempt in range(30):                     # 1,690 small files: HF rate-limits (429) bursts of per-file requests
        try:
            d = snapshot_download("SALT-NLP/spotify_podcast_ASR", repo_type="dataset", max_workers=2)
            break
        except Exception as e:
            if "429" not in str(e) or attempt == 29:
                raise
            time.sleep(60)
    rows = []
    for r in csv.DictReader(open(os.path.join(d, "metadata.csv"), encoding="utf-8")):
        p = os.path.join(d, r["file_name"])
        if os.path.exists(p) and r.get("transcription"):
            rows.append({"audio": {"bytes": open(p, "rb").read()}, "transcription": r["transcription"],
                         "filename": r["filename"]})
    return rows


def build(src, out, scale, seed):
    name = src["name"]
    budget = src["hours"] * scale * 3600 if src["hours"] else float("inf")
    quota = {k: v * scale * 3600 for k, v in (src.get("quota") or {}).items()}
    max_s = src.get("max_s", MAX_S)
    if src.get("loader") == "spotify":
        src = {**src, "audio": "audio", "text": "transcription"}
        shards = [lambda: spotify_rows()]
    else:
        files = sorted(f for f in HfApi().list_repo_files(src["repo"], repo_type="dataset", revision=src["rev"])
                       if f.startswith(src["prefix"]) and f.endswith(".parquet"))
        random.Random(seed).shuffle(files)
        shards = [(lambda f=f, i=i: shard_rows(src, f, random.Random(seed + i))) for i, f in enumerate(files)]
    adir = os.path.join(out, "audio", name)
    os.makedirs(adir, exist_ok=True)
    total, n, skipped, dropped, per_q = 0.0, 0, 0, 0, defaultdict(float)
    full = lambda: total >= budget or (quota and all(per_q[k] >= v for k, v in quota.items()))
    with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as man, ThreadPoolExecutor(4) as pool:
        jobs = iter(shards)
        window = deque(pool.submit(j) for j in islice(jobs, 4))
        while window:                                    # at most 4 shards in flight / in RAM
            rows = window.popleft().result()
            for j in islice(jobs, 1):
                window.append(pool.submit(j))
            for r in rows:
                text = (r[src["text"]] or "").strip()
                q = r.get("scenario") if quota else None
                if not text or (quota and (q not in quota or per_q[q] >= quota[q])):
                    skipped += 1
                    continue
                if src.get("keep") and not src["keep"](r):  # the source's own quality signal says: bad row
                    dropped += 1
                    continue
                try:
                    x, sr = sf.read(io.BytesIO(r[src["audio"]]["bytes"]), dtype="float32")
                except Exception:                            # corrupt / empty audio bytes (seen in psk/vaani-asr)
                    skipped += 1
                    continue
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != SR:
                    x = AF.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, SR).numpy()
                dur = len(x) / SR
                if not MIN_S <= dur <= max_s:
                    skipped += 1
                    continue
                p = os.path.join(adir, f"{n:07d}.flac")
                sf.write(p, x, SR)
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
            print(f"{name}: {total / 3600:7.1f} / {budget / 3600:.0f} h  {n} utts  {skipped} skipped  {dropped} dropped"
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
    failed = []
    for s in srcs:
        try:
            hrs[s["lang"]] += build(s, a.out, a.scale, a.seed)
        except Exception as e:                          # one source failing must not stop the others
            failed.append(s["name"])
            print(f"SOURCE FAILED {s['name']}: {type(e).__name__} {str(e)[:200]}", flush=True)
    print("BUILT unique hours", {k: f"{v:.1f} h" for k, v in hrs.items()}, "failed:", failed, flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
