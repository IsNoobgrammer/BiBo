"""Pack a build_mix.py output into parquet shards and push them to an HF dataset (private unless --public).

    python voice/asr/push_mix.py --mix /home/marimo/work/asr/mix400 --repo fhai50032/asr-en-hi-400h --public [--exclude svarah]

Rows are shuffled (seed 23) so every shard mixes sources. Columns: audio (FLAC bytes, HF Audio feature), text (the
source's original transcript), lang, source, speaker, scenario, duration. Unique audio only: the training repeats are
listed in the card. Uses upload_large_folder (resumable, many files).
"""
import argparse
import glob
import json
import os
import random

import sys

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_mix import SOURCES  # noqa: E402

ROWS_PER_SHARD = 5000
FEATURES = {"audio": {"_type": "Audio", "sampling_rate": 16000}, "text": {"dtype": "string", "_type": "Value"},
            "lang": {"dtype": "string", "_type": "Value"}, "source": {"dtype": "string", "_type": "Value"},
            "speaker": {"dtype": "string", "_type": "Value"}, "scenario": {"dtype": "string", "_type": "Value"}, "duration": {"dtype": "float64", "_type": "Value"}}
CARD = """---
license: other
language: [en, hi]
task_categories: [automatic-speech-recognition]
configs:
- config_name: default
  data_files: data/train-*.parquet
---
# {name}

{hours:.1f} h of English + Hindi ASR audio (16 kHz FLAC, 1-30 s), built by BiBo `voice/asr/build_mix.py`. No filtering:
every row keeps its source's human transcript (original casing / markup; the training text normalisation is in
`voice/asr/prep_train.py`). `speaker` is source-scoped; IndicVoices `scenario` = Conversation / Extempore / Read.
Unique audio only -- `repeat` is how often the training mix uses each row.

| source | lang | hours | rows | repeat |
|---|---|---|---|---|
{table}

Licenses (each row keeps its source's terms; cite the sources): MLCommons/peoples_speech CC-BY / CC-BY-SA,
facebook/voxpopuli CC0, edinburghcstr/ami CC-BY-4.0, ai4bharat/IndicVoices CC-BY-4.0 (arXiv 2403.01926),
ai4bharat/Kathbath CC-BY-4.0, ai4bharat/Lahaja MIT, ai4bharat/Svarah (no license on its card), psk/vaani-asr
(from ARTPARK-IISc Vaani, CC-BY-4.0), agarwalayushi/hinglish CC-BY-4.0. Note: Svarah and Lahaja are public test sets;
models trained on this mix cannot report their official numbers.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--exclude", default="", help="comma list of sources to leave out")
    a = ap.parse_args()
    repeat = {s["name"]: s.get("repeat", 1) for s in SOURCES}
    rows, stats = [], []
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        src = os.path.basename(man)[:-6]
        if src in a.exclude.split(","):
            continue
        src_rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        rows += src_rows
        stats.append(f"| {src} | {src_rows[0]['lang']} | {sum(r['duration'] for r in src_rows) / 3600:.1f} | {len(src_rows)} | x{repeat.get(src, 1)} |")
    random.Random(23).shuffle(rows)
    out = os.path.join(a.mix, "hf")
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    meta = {b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()}
    n_shards = (len(rows) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
    for k in range(n_shards):
        part = rows[k * ROWS_PER_SHARD:(k + 1) * ROWS_PER_SHARD]
        cols = {"audio": [{"bytes": open(r["audio_filepath"], "rb").read(), "path": os.path.basename(r["audio_filepath"])}
                          for r in part]}
        for c in list(FEATURES)[1:]:
            cols[c] = [r.get(c) for r in part]
        t = pa.table(cols).replace_schema_metadata(meta)
        pq.write_table(t, os.path.join(out, "data", f"train-{k:05d}-of-{n_shards:05d}.parquet"))
        print(f"shard {k + 1}/{n_shards}", flush=True)
    hours = sum(r["duration"] for r in rows) / 3600
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(
        CARD.replace("{name}", a.repo.split("/")[-1]).replace("{hours:.1f}", f"{hours:.1f}").replace("{table}", "\n".join(stats)))
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=out)
    print(f"PUSHED https://huggingface.co/datasets/{a.repo}  {len(rows)} rows  {hours:.1f} h", flush=True)


if __name__ == "__main__":
    main()
