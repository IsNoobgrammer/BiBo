"""Push ONE build_mix.py source to an HF dataset as soon as it is built (a box death then loses only the source in
flight). Layout data/<source>/train-NNNNN.parquet + manifests/<source>.jsonl, the same columns as push_mix.py, so
fetch_mix.py restores it unchanged.

    python voice/asr/push_src.py --mix /home/marimo/work/asr/mix_en --source ami_sdm --repo fhai50032/asr-english-v2
"""
import argparse
import json
import os
import random
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push_mix import FEATURES, ROWS_PER_SHARD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.mix, f"{a.source}.jsonl"), encoding="utf-8")]
    out = os.path.join(a.mix, "_push", a.source)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "manifests"))
    os.makedirs(os.path.join(out, "data", a.source))
    shutil.copy(os.path.join(a.mix, f"{a.source}.jsonl"), os.path.join(out, "manifests", f"{a.source}.jsonl"))
    meta = {b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()}
    random.Random(23).shuffle(rows)
    n = (len(rows) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
    for k in range(n):
        part = rows[k * ROWS_PER_SHARD:(k + 1) * ROWS_PER_SHARD]
        cols = {"audio": [{"bytes": open(r["audio_filepath"], "rb").read(),
                           "path": f"{r['source']}/{os.path.basename(r['audio_filepath'])}"} for r in part]}
        for c in list(FEATURES)[1:]:
            cols[c] = [None if r.get(c) is None else (str(r[c]) if FEATURES[c].get("dtype") == "string" else r[c])
                       for r in part]
        pq.write_table(pa.table(cols).replace_schema_metadata(meta), os.path.join(out, "data", a.source, f"train-{k:05d}.parquet"))
        print(f"packed {a.source} {k + 1}/{n}", flush=True)
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=out)
    shutil.rmtree(out, ignore_errors=True)                  # the staged parquet copy is not needed after the upload
    print(f"PUSHED {a.source} {len(rows)} rows {sum(r['duration'] for r in rows) / 3600:.1f} h -> {a.repo}", flush=True)


if __name__ == "__main__":
    main()
