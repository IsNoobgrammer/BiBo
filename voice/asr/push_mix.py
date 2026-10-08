"""Pack a build_mix.py output into parquet shards and push them to an HF dataset (private unless --public).

    python voice/asr/push_mix.py --mix /home/marimo/work/asr/mix1000 --lang hi --repo fhai50032/asr-hindi
    python voice/asr/push_mix.py --mix /home/marimo/work/asr/mix1000 --lang en --repo fhai50032/asr-english
One language per repo. Hindi: one default config (data/shared). English: the groups + configs below.
Set HF_XET_HIGH_PERFORMANCE=1 for faster Xet transfers.

Unique audio, stored ONCE, in groups; the card defines one config per English variant (prep_train.VARIANTS):
    data/shared/        every source except the emilia / nptel pools (all Hindi, AMI, Spotify, phone, ...)
    data/emilia_core/   first 100 h of Emilia by prep_train.pool_split    data/emilia_extra/  the rest
    data/nptel_core/    first 50 h of NPTEL                               data/nptel_extra/   the rest
    config nptel50  = shared + emilia_core + emilia_extra + nptel_core   (Emilia-heavy, NPTEL 50 h)
    config nptel150 = shared + emilia_core + nptel_core + nptel_extra    (NPTEL ~150 h)
Columns: audio (FLAC bytes, HF Audio), text (source transcript, original casing), lang, source, speaker, scenario,
meeting, begin, duration. manifests/ holds the original build rows for fetch_mix.py. upload_large_folder (resumable).
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mix import SOURCES  # noqa: E402
from prep_train import VARIANTS, pool_split  # noqa: E402

ROWS_PER_SHARD = 4000
STR = {"dtype": "string", "_type": "Value"}
FEATURES = {"audio": {"_type": "Audio", "sampling_rate": 16000}, "text": STR, "lang": STR, "source": STR, "speaker": STR,
            "scenario": STR, "meeting": STR, "begin": {"dtype": "float64", "_type": "Value"},
            "duration": {"dtype": "float64", "_type": "Value"}}
GROUPS = {"nptel50": ["shared", "emilia_core", "emilia_extra", "nptel_core"],
          "nptel150": ["shared", "emilia_core", "nptel_core", "nptel_extra"]}


def groups(rows_by_src):
    core = {src: min(h for v in VARIANTS.values() for s, h in v.items() if s == src and h is not None)
            for src in ("emilia", "nptel")}
    out = {g: [] for g in ("shared", "emilia_core", "emilia_extra", "nptel_core", "nptel_extra")}
    for src, rows in rows_by_src.items():
        if src in core:
            c, rest = pool_split(rows, core[src])
            out[f"{src}_core"] += c
            out[f"{src}_extra"] += rest
        else:
            out["shared"] += rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--lang", choices=["hi", "en"], required=True, help="one language per repo")
    a = ap.parse_args()
    repeat = {s["name"]: s.get("repeat", 1) for s in SOURCES}
    windows_only = {s["name"] for s in SOURCES if s.get("windows_only")}
    rows_by_src = {}
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        if man.endswith("_raw.jsonl"):
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        if rows and rows[0]["lang"] == a.lang:
            rows_by_src[rows[0]["source"]] = rows
    grp = groups(rows_by_src)
    out = os.path.join(a.mix, f"hf_{a.lang}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "manifests"))
    for src in rows_by_src:
        shutil.copy(os.path.join(a.mix, f"{src}.jsonl"), os.path.join(out, "manifests", f"{src}.jsonl"))
    meta = {b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()}
    for g, rows in grp.items():
        if not rows:
            continue
        random.Random(23).shuffle(rows)
        os.makedirs(os.path.join(out, "data", g), exist_ok=True)
        n = (len(rows) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
        for k in range(n):
            part = rows[k * ROWS_PER_SHARD:(k + 1) * ROWS_PER_SHARD]
            cols = {"audio": [{"bytes": open(r["audio_filepath"], "rb").read(),
                               "path": f"{r['source']}/{os.path.basename(r['audio_filepath'])}"} for r in part]}
            for c in list(FEATURES)[1:]:
                cols[c] = [None if r.get(c) is None else (str(r[c]) if FEATURES[c].get("dtype") == "string" else r[c])
                           for r in part]
            pq.write_table(pa.table(cols).replace_schema_metadata(meta),
                           os.path.join(out, "data", g, f"train-{k:05d}.parquet"))
        print(f"group {g}: {len(rows)} rows, {sum(r['duration'] for r in rows) / 3600:.1f} h, {n} shards", flush=True)
    hrs = lambda gs: sum(r["duration"] for g in gs for r in grp[g]) / 3600
    configs = GROUPS if a.lang == "en" else {"default": ["shared"]}
    table = "\n".join(f"| {src} | {rows[0]['lang']} | {sum(r['duration'] for r in rows) / 3600:.1f} | {len(rows)} | "
                      f"x{repeat.get(src, 1)}{' (meeting windows only)' if src in windows_only else ''} |"
                      for src, rows in sorted(rows_by_src.items()))
    cfg = "\n".join(f"- config_name: {v}\n  data_files:\n" + "\n".join(f"  - data/{g}/*.parquet" for g in gs)
                    for v, gs in configs.items())
    card = f"""---
license: other
language: [{a.lang}]
task_categories: [automatic-speech-recognition]
configs:
{cfg}
---
# {a.repo.split('/')[-1]}

{"Hindi" if a.lang == "hi" else "English"} ASR audio (16 kHz FLAC), one half of the BiBo voice 1000 h mix (600 h
Hindi / 400 h English), built by BiBo `voice/asr/build_mix.py`. Bad rows were REMOVED, never
relabelled: Emilia keeps rows whose two independent transcripts agree (<= 5% WER) with audio quality PQ >= 6.5; Numo
keeps its own WER <= 6% and synthetic-speech score <= 0.2; NPTEL drops rows a Qwen3-ASR pass scores above 25% WER
(and 25-50% rows, which Gemini also rejects 90% of the time: labels running past the audio cut). Text keeps each source's original casing / markup; the training
normalisation is in `voice/asr/prep_train.py`.

Configs: {", ".join(f"**{v}** {hrs(gs):.1f} h" for v, gs in configs.items())}.

| source | lang | hours | rows | train use |
|---|---|---|---|---|
{table}

Licenses: each row keeps its source's terms -- MrDragonFox/EN_Emilia_Yodas_616h CC-BY-4.0, psdn-ai/numo-indic-speech
CC-BY-4.0, skbose/indian-english-nptel-v0, edinburghcstr/ami CC-BY-4.0, SALT-NLP/spotify_podcast_ASR,
sawradip/phone-asr-data, facebook/voxpopuli CC0, MLCommons/peoples_speech CC-BY / CC-BY-SA, ai4bharat/Svarah,
yashtiwari/PaulMooney-Medical-ASR-Data, ai4bharat/IndicVoices CC-BY-4.0, psk/vaani-asr (Vaani CC-BY-4.0),
ai4bharat/Kathbath CC-BY-4.0, agarwalayushi/hinglish CC-BY-4.0, ai4bharat/Lahaja MIT. Svarah and Lahaja are public test
sets: models trained on this cannot report their official numbers.
"""
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(card)
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=out)
    print(f"PUSHED https://huggingface.co/datasets/{a.repo}", {v: round(hrs(gs), 1) for v, gs in configs.items()},
          flush=True)


if __name__ == "__main__":
    main()
