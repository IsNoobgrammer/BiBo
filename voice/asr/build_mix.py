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


# hours = unique hours to take (None = all); every source x1 here (repeat = train copies, copy 1+ augmented in prep_train.py: set it for post-training)
# EN per variant ~400: [emilia + nptel pools: nptel50 = emilia ~200 + nptel 50, nptel150 = emilia 100 + nptel ~150]
#   + ami windows ~80 + spotify + phone + voxpopuli 20 + people's speech 15 + svarah ~9 + medical ~8 + phone
# HI 600: indicvoices 300 + numo 120 + vaani 100 + kathbath 50 + hinglish 20 + lahaja ~11
SOURCES = [
    # --- English pools for the two variants ---
    dict(name="emilia", repo="MrDragonFox/EN_Emilia_Yodas_616h", rev="main", prefix="data/train-", audio="audio",
         text="text_scribe", lang="en", hours=210, frac=1.0, keep=emilia_ok, speaker=lambda r: r["file_id"].rsplit("_W", 1)[0],
         cols=["text_emilia", "PQ"]),                                       # spontaneous YouTube English, cased + punct
    dict(name="nptel", repo="skbose/indian-english-nptel-v0", rev="main", prefix="data/train-", audio="audio",
         text="transcription_normalised", lang="en", hours=None, frac=0.57, speaker=lambda r: r["speaker_name"]),
    # --- English, shared ---
    dict(name="ami_ihm", repo="edinburghcstr/ami", rev=CONV, prefix="ihm/train/", audio="audio", text="text", lang="en",
         hours=None, frac=1.0, windows_only=True, speaker=lambda r: r["speaker_id"],
         extra={"meeting": "meeting_id", "begin": "begin_time"}),           # ONLY as full-meeting <spk> windows
    dict(name="ami_sdm", repo="edinburghcstr/ami", rev=CONV, prefix="sdm/train/", audio="audio", text="text", lang="en",
         hours=None, frac=1.0, windows_only=True, speaker=lambda r: r["speaker_id"],
         extra={"meeting": "meeting_id", "begin": "begin_time"}),           # same meetings, single distant mic (far-field)
    dict(name="librispeech", repo="openslr/librispeech_asr", rev="main", prefix="all/train.clean.100/", audio="audio",
         text="text", lang="en", hours=None, frac=1.0, speaker=lambda r: str(r["speaker_id"])),  # read, train-clean-100
    dict(name="earnings22", loader="earnings22", repo="anton-l/earnings22_baseline_5_gram", lang="en", hours=None,
         speaker=lambda r: r["call"]),                                      # earnings calls, many accents (CC BY-SA)
    dict(name="tedlium", loader="tedlium", repo="kfajdsl/tedlium", lang="en", hours=300,
         speaker=lambda r: r["talk"]),                                      # TED talks, wide vocabulary (CC BY-NC-ND)
    dict(name="spgi2", loader="spgi2", repo="kensho/SPGISpeech2.0", lang="en", hours=300,
         speaker=lambda r: str(r["spk"])),                                  # earnings calls (Kensho: non-commercial)
    # Voices-in-the-Wild (Apache-2.0): LibriSpeech-train / Common Voice sentences re-recorded through simulated
    # acoustics; English rows only (its question field), 24 kHz -> resampled. No speaker ids (each row its own).
    *[dict(name=f"vitw_{s}", repo="zhifeixie/Voices-in-the-Wild-2M", rev="main", prefix=f"data/{s}-", audio="audio",
           text="answer", lang="en", hours=50, frac=1.0, cols=["question", "name"],
           keep=lambda r: r["question"].startswith("Please transcribe"), speaker=lambda r: r["name"])
      for s in ("distortion", "dropout", "echo", "far_field", "noise", "obstructed", "recording", "far_field_noise")],
    dict(name="spotify", loader="spotify", repo="SALT-NLP/spotify_podcast_ASR", lang="en", hours=None, max_s=60,
         speaker=lambda r: r["filename"]),                                  # human verbatim podcast talk, 2-3 speakers
    dict(name="phone", repo="sawradip/phone-asr-data", rev="main", prefix="data/", audio="audio", text="transcription",
         lang="en", hours=None, frac=1.0, speaker=lambda r: r["filename"].split("-")[0]),  # phone chat, "yeah"s
    dict(name="voxpopuli", repo="facebook/voxpopuli", rev=CONV, prefix="en/train/", audio="audio", text="raw_text",
         lang="en", hours=20, frac=0.25, speaker=lambda r: r["speaker_id"]),
    dict(name="peoples_speech", repo="MLCommons/peoples_speech", rev=CONV, prefix="clean/train/", audio="audio",
         text="text", lang="en", hours=15, frac=0.1, speaker=lambda r: r["id"].rsplit("_", 1)[0]),
    dict(name="svarah", repo="ai4bharat/Svarah", rev="main", prefix="data/test-", audio="audio_filepath", text="text",
         lang="en", hours=None, frac=1.0,
         speaker=lambda r: f"{r.get('native_place_district')}|{r.get('gender')}|{r.get('age-group')}"),
    dict(name="medical", repo="yashtiwari/PaulMooney-Medical-ASR-Data", rev="main", prefix="data/", audio="path",
         text="sentence", lang="en", hours=None, frac=1.0, speaker=lambda r: str(r["speaker_id"])),
    # --- Hindi ---
    dict(name="indicvoices_hi", repo="ai4bharat/IndicVoices", rev="main", prefix="hindi/train-", audio="audio_filepath",
         text="text", lang="hi", hours=300, frac=1.0, speaker=lambda r: r["speaker_id"], extra={"scenario": "scenario"},
         quota={"Conversation": 120, "Extempore": 130, "Read": 50}),       # conversation is only ~147 h of ~640 h
    dict(name="numo_hi", repo="psdn-ai/numo-indic-speech", rev="main", prefix="data/hindi/", audio="audio",
         text="transcript", lang="hi", hours=120, frac=1.0, keep=numo_ok, max_s=70, speaker=lambda r: r["speaker_id"],
         cols=["wer", "synthetic_suspicion_score"]),                         # read, 36-67 s, 48 kHz stereo
    dict(name="vaani_hi", repo="psk/vaani-asr", rev="main", prefix="hindi/train-", audio="audio", text="transcript",
         lang="hi", hours=100, frac=0.9, speaker=lambda r: r["file_name"].split("_")[5]),
    dict(name="kathbath_hi", repo="ai4bharat/Kathbath", rev="main", prefix="hindi/train-", audio="audio_filepath",
         text="text", lang="hi", hours=50, frac=0.5, speaker=lambda r: str(r["speaker_id"])),
    dict(name="hinglish", repo="agarwalayushi/hinglish", rev="main", prefix="data/train-", audio="audio", text="text",
         lang="hi", hours=20, frac=0.05, speaker=lambda r: r["source"]),  # spoken tutorials (= MUCS)
    dict(name="lahaja", repo="ai4bharat/Lahaja", rev="main", prefix="data/test-", audio="audio_filepath", text="text",
         lang="hi", hours=None, frac=1.0, speaker=lambda r: str(r["sp_id"])),
]
MIN_S, MAX_S = 1.0, 30.0
SPEAKER_COLS = {"native_place_district", "gender", "age-group", "id", "speaker_id", "file_name", "source", "sp_id",
                "speaker_name", "file_id", "filename"}


def hf(fn, *args, **kw):
    """Every HF call retries on 429: a 1000 h build makes thousands of requests and the Hub rate-limits bursts."""
    for attempt in range(40):
        try:
            return fn(*args, **kw)
        except Exception as e:
            if "429" not in str(e) or attempt == 39:
                raise
            time.sleep(min(30 * (attempt + 1), 300))


def shard_rows(src, path, rng):
    local = hf(hf_hub_download, src["repo"], path, repo_type="dataset", revision=src["rev"])
    names = set(pq.read_schema(local).names)
    want = ({src["audio"], src["text"]} | (SPEAKER_COLS & names) | (set(src.get("extra", {}).values()) & names)
            | (set(src.get("cols", [])) & names))
    t = pq.read_table(local, columns=sorted(want)).to_pylist()
    os.remove(os.path.realpath(local))                  # keep the HF cache from filling the disk
    return [r for r in t if rng.random() < src["frac"]]


def spotify_rows():
    """SALT-NLP/spotify_podcast_ASR: .ogg clips + metadata.csv (human transcription per clip)."""
    d = hf(snapshot_download, "SALT-NLP/spotify_podcast_ASR", repo_type="dataset", max_workers=2)  # 1,690 small files
    rows = []
    for r in csv.DictReader(open(os.path.join(d, "metadata.csv"), encoding="utf-8")):
        p = os.path.join(d, r["file_name"])
        if os.path.exists(p) and r.get("transcription"):
            rows.append({"audio": {"bytes": open(p, "rb").read()}, "transcription": r["transcription"],
                         "filename": r["filename"]})
    return rows


def _wav(x, sr=SR):
    b = io.BytesIO()
    sf.write(b, x, sr, format="WAV")
    return {"bytes": b.getvalue()}


DROP_TAGS = ("<inaudible", "<crosstalk", "inaudible", "ignore_time_segment_in_scoring", "<unk>")  # label != audio


def earnings22_shards(src, seed):
    """anton-l/earnings22_baseline_5_gram (the files ESB's earnings22 reads): one tar per call of segment wavs +
    metadata.csv (cased, punctuated sentence per segment). Speaker = the call (no per-speaker labels)."""
    meta = {}
    for r in csv.DictReader(open(hf(hf_hub_download, src["repo"], "metadata.csv", repo_type="dataset"), encoding="utf-8")):
        meta[r["file"]] = r
    tars = sorted({f"data/chunked/{r['source_id']}.tar.gz" for r in meta.values()})
    random.Random(seed).shuffle(tars)

    def shard(path):
        import tarfile
        local = hf(hf_hub_download, src["repo"], path, repo_type="dataset")
        rows = []
        with tarfile.open(local) as t:
            for m in t:
                r = meta.get(m.name.lstrip("./"))
                if m.isfile() and r and not any(k in r["sentence"].lower() for k in DROP_TAGS):
                    rows.append({"audio": {"bytes": t.extractfile(m).read()}, "text": r["sentence"], "call": r["source_id"]})
        os.remove(os.path.realpath(local))
        return rows
    return [(lambda p=p: shard(p)) for p in tars]


def tedlium_shards(src, seed):
    """kfajdsl/tedlium (mirror of LIUM/tedlium) release 3 legacy train_1 (~300 h of 452, all of it): talk .sph + .stm segments.
    Text fixes as ESB: lowercase, "it 's" -> "it's"; a segment with <unk> (an unknown spoken word) is dropped, not
    kept with the word deleted. Speaker = the talk."""
    import re
    import tarfile
    root = os.path.join(src["_out"], "_tedlium_extract")
    if not os.path.isdir(root):
        local = hf(hf_hub_download, src["repo"], "TEDLIUM_release3/legacy/train_1.tar.gz", repo_type="dataset")
        with tarfile.open(local) as t:
            t.extractall(root)
        os.remove(os.path.realpath(local))
    stms = sorted(glob_files(root, ".stm"))
    random.Random(seed).shuffle(stms)

    def shard(stm):
        rows, audio = [], {}
        for line in open(stm, encoding="utf-8"):
            fn, _ch, spk, t0, t1, _label, text = line.strip().split(" ", 6)
            text = text.rsplit(" (", 1)[0].lower() if text.endswith(")") else text.lower()
            if not text or any(k in text for k in DROP_TAGS):
                continue
            text = re.sub(r" '(?=[a-z])", "'", text)
            if fn not in audio:
                audio[fn], _ = sf.read(os.path.join(os.path.dirname(os.path.dirname(stm)), "sph", fn + ".sph"),
                                       dtype="float32")
            x = audio[fn][int(float(t0) * SR): int(float(t1) * SR)]
            rows.append({"audio": _wav(x), "text": text, "talk": fn})
        return rows
    return [(lambda s=s: shard(s)) for s in stms]


def glob_files(root, ext):
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(ext)]


def cut_aligned(words, max_s=25.0, min_gap=0.15):
    """Word alignment [(word, start, end, speaker)] -> [(start, end, [words])]: a new piece at every speaker change,
    and at the first pause >= min_gap once a piece is past 2/3 of max_s (hard cut at max_s). Cut points sit in the
    middle of the gap, so no piece holds part of a neighbour's word."""
    pieces, cur = [], []
    for w in words:
        if cur:
            gap, dur = w[1] - cur[-1][2], w[2] - cur[0][1]
            if w[3] != cur[-1][3] or dur > max_s or (gap >= min_gap and dur > max_s * 2 / 3):
                pieces.append(cur)
                cur = []
        cur.append(w)
    if cur:
        pieces.append(cur)
    out = []
    for i, p in enumerate(pieces):
        lo = (pieces[i - 1][-1][2] + p[0][1]) / 2 if i else max(p[0][1] - 0.2, 0.0)
        hi = (p[-1][2] + pieces[i + 1][0][1]) / 2 if i + 1 < len(pieces) else p[-1][2] + 0.2
        out.append((lo, max(hi, lo), [w[0] for w in p], p[0][3]))
    return out


def spgi2_shards(src, seed):
    """kensho/SPGISpeech2.0: ~75 s multi-speaker snippets (cased, punctuated raw_transcript) + per-word alignment
    (word, start, end, speaker) in supplemental_data/alignment_files.tar.gz. Each snippet is cut by cut_aligned into
    single-speaker pieces <= 25 s; text = the aligned words. Speaker = the alignment's speaker id."""
    import tarfile
    root = os.path.join(src["_out"], "_spgi2_align")
    if not os.path.isdir(root):
        local = hf(hf_hub_download, src["repo"], "supplemental_data/alignment_files.tar.gz", repo_type="dataset")
        with tarfile.open(local) as t:
            t.extractall(root)
    files = sorted(f for f in hf(HfApi().list_repo_files, src["repo"], repo_type="dataset")
                   if f.startswith("data/train_part_") and f.endswith(".parquet"))
    random.Random(seed).shuffle(files)

    def shard(path):
        local = hf(hf_hub_download, src["repo"], path, repo_type="dataset")
        t = pq.read_table(local, columns=["call_id", "snippet_id", "audio"]).to_pylist()
        os.remove(os.path.realpath(local))
        rows = []
        for r in t:
            ap = os.path.join(root, "alignment_files", str(r["call_id"]), f"{r['snippet_id']}.json")
            if not os.path.exists(ap):
                continue
            al = json.load(open(ap))
            words = [(w["word"], w["start_time"], w["end_time"], w["speaker"]) for _, w in sorted(al.items(), key=lambda kv: int(kv[0]))]
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            for lo, hi, ws, spk in cut_aligned(words):
                rows.append({"audio": _wav(x[int(lo * sr): int(hi * sr)], sr), "text": " ".join(ws), "spk": spk})
        return rows
    return [(lambda p=p: shard(p)) for p in files]


LOADERS = {"earnings22": earnings22_shards, "tedlium": tedlium_shards, "spgi2": spgi2_shards}


def build(src, out, scale, seed):
    name = src["name"]
    budget = src["hours"] * scale * 3600 if src["hours"] else float("inf")
    quota = {k: v * scale * 3600 for k, v in (src.get("quota") or {}).items()}
    max_s = src.get("max_s", MAX_S)
    if src.get("loader") == "spotify":
        src = {**src, "audio": "audio", "text": "transcription"}
        shards = [lambda: spotify_rows()]
    elif src.get("loader") in LOADERS:
        src = {**src, "audio": "audio", "text": "text", "_out": out}
        shards = LOADERS[src["loader"]](src, seed)
    else:
        files = sorted(f for f in hf(HfApi().list_repo_files, src["repo"], repo_type="dataset", revision=src["rev"])
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
    _w = [("a", 0.0, 0.5, 1), ("b", 0.6, 1.0, 1), ("c", 1.4, 1.8, 2)] + [(f"w{i}", 2 + i, 2.9 + i, 2) for i in range(30)]
    _p = cut_aligned(_w)
    assert [p[2][:2] for p in _p[:2]] == [["a", "b"], ["c", "w0"]] and _p[0][1] == _p[1][0] == 1.2, _p[:2]
    assert all(hi - lo <= 26 for lo, hi, _, _ in _p) and sum(len(p[2]) for p in _p) == len(_w)
    main()
