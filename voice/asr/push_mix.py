"""Pack a build_mix.py output (+ qwen_check.py report columns) into parquet shards and push them to a PRIVATE HF dataset.

    python voice/asr/push_mix.py --mix /home/marimo/work/asr/mix100 --repo fhai50032/asr-en-hi-100h

Rows are shuffled (seed 23) so every shard mixes sources. Columns: audio (FLAC bytes, HF Audio feature), text, lang,
source, scenario, duration, qwen_text, qwen_wer. Private because IndicVoices is a gated set (CC-BY-4.0, attribution in
the card); flip visibility on the Hub only after checking each source's terms.
"""
import argparse
import glob
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

ROWS_PER_SHARD = 5000
FEATURES = {"audio": {"_type": "Audio", "sampling_rate": 16000}, "text": {"dtype": "string", "_type": "Value"},
            "lang": {"dtype": "string", "_type": "Value"}, "source": {"dtype": "string", "_type": "Value"},
            "scenario": {"dtype": "string", "_type": "Value"}, "duration": {"dtype": "float64", "_type": "Value"},
            "qwen_text": {"dtype": "string", "_type": "Value"}, "qwen_wer": {"dtype": "float64", "_type": "Value"}}
CARD = """---
license: cc-by-4.0
language: [en, hi]
task_categories: [automatic-speech-recognition]
configs:
- config_name: default
  data_files: data/train-*.parquet
---
# {name}

{hours:.1f} h of English + Hindi ASR audio (16 kHz FLAC, 1-30 s), built by BiBo `voice/asr/build_mix.py`. No filtering:
every row keeps its source's human transcript. `qwen_text` / `qwen_wer` are Qwen3-ASR-1.7B's transcript and its WER
against `text` (lowercase, punctuation and fillers stripped) -- a quality signal, not a label.

| source | lang | hours | rows | corpus WER vs Qwen3-ASR |
|---|---|---|---|---|
{table}

Sources: MLCommons/peoples_speech (clean; CC-BY / CC-BY-SA), facebook/voxpopuli (en; CC0), edinburghcstr/ami (ihm;
CC-BY-4.0), ai4bharat/IndicVoices (hindi; CC-BY-4.0, gated -- cite AI4Bharat IndicVoices, arXiv 2403.01926).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    rows, stats = [], []
    for man in sorted(glob.glob(os.path.join(a.mix, "*.jsonl"))):
        src = os.path.basename(man)[:-6]
        q = {}
        qp = os.path.join(a.mix, "qwen", f"{src}_all.jsonl")
        if os.path.exists(qp):
            q = {r["audio_filepath"]: r for r in map(json.loads, open(qp, encoding="utf-8"))}
        src_rows = [json.loads(l) for l in open(man, encoding="utf-8")]
        for r in src_rows:
            r["qwen_text"] = q.get(r["audio_filepath"], {}).get("qwen_text")
            r["qwen_wer"] = q.get(r["audio_filepath"], {}).get("wer")
        rows += src_rows
        scored = [r for r in src_rows if r["qwen_wer"] is not None]
        words = [len(r["text"].split()) for r in scored]
        cwer = sum(r["qwen_wer"] * w for r, w in zip(scored, words)) / max(sum(words), 1)
        stats.append(f"| {src} | {src_rows[0]['lang']} | {sum(r['duration'] for r in src_rows) / 3600:.1f} | "
                     f"{len(src_rows)} | {f'{100 * cwer:.1f}%' if scored else 'n/a'} |")
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
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=out, commit_message=f"{hours:.1f} h en/hi ASR mix")
    print(f"PUSHED https://huggingface.co/datasets/{a.repo}  {len(rows)} rows  {hours:.1f} h", flush=True)


if __name__ == "__main__":
    main()
