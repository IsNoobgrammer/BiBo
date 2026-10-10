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
from huggingface_hub import CommitOperationDelete, HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push_mix import FEATURES, ROWS_PER_SHARD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--core_hours", type=float, default=None,
                    help="also split into data/<src>/core<H>/ (prep_train.pool_split, the subset prep_en trains on) + rest/")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.mix, f"{a.source}.jsonl"), encoding="utf-8")]
    out = os.path.join(a.mix, "_push", a.source)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "manifests"))
    shutil.copy(os.path.join(a.mix, f"{a.source}.jsonl"), os.path.join(out, "manifests", f"{a.source}.jsonl"))
    meta ={b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()}
    if a.core_hours:
        from prep_train import pool_split
        core, rest = pool_split(rows, a.core_hours)
        groups = {f"{a.source}/core{a.core_hours:g}": core, f"{a.source}/rest": rest}
    else:
        groups = {a.source: rows}
    for g, grows in groups.items():
        os.makedirs(os.path.join(out, "data", g))
        random.Random(23).shuffle(grows)
        n = (len(grows) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
        for k in range(n):
            part = grows[k * ROWS_PER_SHARD:(k + 1) * ROWS_PER_SHARD]
            cols = {"audio": [{"bytes": open(r["audio_filepath"], "rb").read(),
                               "path": f"{r['source']}/{os.path.basename(r['audio_filepath'])}"} for r in part]}
            for c in list(FEATURES)[1:]:
                cols[c] = [None if r.get(c) is None else (str(r[c]) if FEATURES[c].get("dtype") == "string" else r[c])
                           for r in part]
            pq.write_table(pa.table(cols).replace_schema_metadata(meta), os.path.join(out, "data", g, f"train-{k:05d}.parquet"))
            print(f"packed {g} {k + 1}/{n}", flush=True)
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=out)
    # a re-push with fewer shards must not leave the old ones behind (a config reads every file in the folder)
    mine = {os.path.relpath(os.path.join(d, f), out).replace(os.sep, "/") for d, _, fs in os.walk(out) for f in fs}
    stale = [f for f in api.list_repo_files(a.repo, repo_type="dataset")
             if f.startswith(f"data/{a.source}/") and f.endswith(".parquet") and f not in mine]
    if stale:
        api.create_commit(a.repo, repo_type="dataset", operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
                          commit_message=f"{a.source}: remove {len(stale)} stale shards of an earlier push")
        print(f"removed {len(stale)} stale shards of {a.source}", flush=True)
    shutil.rmtree(out, ignore_errors=True)                  # the staged parquet copy is not needed after the upload
    print(f"PUSHED {a.source} {len(rows)} rows {sum(r['duration'] for r in rows) / 3600:.1f} h -> {a.repo}", flush=True)


if __name__ == "__main__":
    main()
