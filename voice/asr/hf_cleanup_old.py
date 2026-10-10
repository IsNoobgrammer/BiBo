"""fhai50032/asr-english after the per-source re-pack: verify every data/<source>/ folder against its manifest (row
count from the parquet footers, nothing downloaded but the footers), and only if ALL match delete the old grouped
layout (data/shared, data/emilia_*, data/nptel_* groups) plus manifests whose audio lived only there (ami_ihm: the
improved AMI headset build is in asr-english-v2). Prints what it would delete with --dry_run.

    python voice/asr/hf_cleanup_old.py --repo fhai50032/asr-english --sources emilia nptel ... [--dry_run]
"""
import argparse
import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi, HfFileSystem, hf_hub_download

OLD_GROUPS = ("data/shared/", "data/emilia_core/", "data/emilia_extra/", "data/nptel_core/", "data/nptel_extra/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    api, fs = HfApi(), HfFileSystem()
    files = api.list_repo_files(a.repo, repo_type="dataset")
    ok = True
    for s in a.sources:
        man = sum(1 for _ in open(hf_hub_download(a.repo, f"manifests/{s}.jsonl", repo_type="dataset"), encoding="utf-8"))
        shards = [f for f in files if f.startswith(f"data/{s}/") and f.endswith(".parquet")]
        n = sum(pq.ParquetFile(fs.open(f"datasets/{a.repo}/{f}")).metadata.num_rows for f in shards)
        good = n == man and shards
        ok &= bool(good)
        print(f"{'OK ' if good else 'BAD'} {s}: manifest {man} rows, data/{s}/ {n} rows in {len(shards)} shards", flush=True)
    if not ok:
        print("VERIFY FAILED: nothing deleted", flush=True)
        sys.exit(1)
    keep = set(a.sources)
    dels = [f for f in files if f.startswith(OLD_GROUPS)]
    dels += [f for f in files if f.startswith("manifests/") and f[len("manifests/"):-len(".jsonl")] not in keep
             and not any(g.startswith(f"data/{f[len('manifests/'):-len('.jsonl')]}/") for g in files)]
    print(f"{'WOULD DELETE' if a.dry_run else 'DELETING'} {len(dels)} files:",
          json.dumps(sorted({'/'.join(f.split('/')[:2]) for f in dels})), flush=True)
    if not a.dry_run and dels:
        api.create_commit(a.repo, repo_type="dataset", operations=[CommitOperationDelete(path_in_repo=f) for f in dels],
                          commit_message="remove the old grouped layout (re-packed per source; AMI headset -> asr-english-v2)")
        print("DELETED", flush=True)


if __name__ == "__main__":
    main()
