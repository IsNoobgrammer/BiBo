"""Check a push_src.py repo: every data/<source>/ folder holds exactly its manifest's rows (parquet footers only).
With --prune, a flat folder with extra rows loses the shards past ceil(rows / ROWS_PER_SHARD) (left by an earlier,
larger push of the same source), then is re-checked.

    python voice/asr/hf_verify.py --repo fhai50032/asr-english-v2 [--prune]
"""
import argparse
import math
import re
import sys

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi, HfFileSystem, hf_hub_download

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from push_mix import ROWS_PER_SHARD  # noqa: E402


def check(api, fs, repo):
    files = api.list_repo_files(repo, repo_type="dataset")
    srcs = sorted(f[10:-6] for f in files if f.startswith("manifests/") and f.endswith(".jsonl"))
    out = {}
    for s in srcs:
        man = sum(1 for _ in open(hf_hub_download(repo, f"manifests/{s}.jsonl", repo_type="dataset"), encoding="utf-8"))
        shards = sorted(f for f in files if f.startswith(f"data/{s}/") and f.endswith(".parquet"))
        n = sum(pq.ParquetFile(fs.open(f"datasets/{repo}/{f}")).metadata.num_rows for f in shards)
        out[s] = (man, n, shards)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prune", action="store_true")
    a = ap.parse_args()
    api, fs = HfApi(), HfFileSystem()
    res = check(api, fs, a.repo)
    bad = []
    for s, (man, n, shards) in res.items():
        print(f"{'OK ' if man == n else 'BAD'} {a.repo} {s}: manifest {man}, data {n} rows in {len(shards)} shards", flush=True)
        if man != n:
            bad.append(s)
    if a.prune and bad:
        ops = []
        for s in bad:
            man, n, shards = res[s]
            flat = [f for f in shards if f.count("/") == 2]
            keep = math.ceil(man / ROWS_PER_SHARD)
            ops += [CommitOperationDelete(path_in_repo=f) for f in flat
                    if int(re.search(r"train-(\d+)\.parquet$", f).group(1)) >= keep]
        if ops:
            api.create_commit(a.repo, repo_type="dataset", operations=ops,
                              commit_message=f"remove {len(ops)} stale shards of earlier, larger pushes")
            print(f"PRUNED {len(ops)} stale shards", flush=True)
        res = check(api, fs, a.repo)
        bad = [s for s, (man, n, _) in res.items() if man != n]
        print("AFTER PRUNE:", "all match" if not bad else f"still BAD {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
