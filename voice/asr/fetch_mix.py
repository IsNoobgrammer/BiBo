"""Fresh box: restore a build_mix.py output from its HF dataset instead of rebuilding from ~10 source datasets.

    python voice/asr/fetch_mix.py --repo fhai50032/asr-en-hi-400h --mix /home/marimo/work/asr/mix400 [--workers 16]

The repo holds the audio (data/*.parquet: audio bytes + source + file name) and the ORIGINAL manifests
(manifests/<source>.jsonl, with every field build_mix wrote -- e.g. AMI meeting / begin for real speaker turns). Audio
is written back to the exact paths the manifests name, so prep_train.py runs unchanged. Skips if already restored.

Parallel (Oct 12 2026): the parquet shards are restored by a process pool, each worker streaming its shard in record
batches and writing that shard's FLACs (the old single loop wrote ~720k files one at a time: most of the ~30 min
fresh-box data setup, file creation being slow on the gVisor sandbox). Each repo downloads into its OWN folder
(hf_mix/<repo>), so setup_en3.sh can fetch the three en3 repos at the same time.
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

WANT = {}                                                       # "<source>/<file>" -> manifest path (per worker)


def _init(want):
    WANT.update(want)


def _restore(f):
    """One parquet shard -> its audio files; returns (written, shard rows without a manifest row)."""
    n = skipped = 0
    made = set()
    for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["audio", "source"]):
        for row in batch.to_pylist():
            ap_ = row["audio"]["path"]                          # "<source>/<file>" (old 400 h layout: "<file>")
            p = WANT.get(ap_ if "/" in ap_ else f'{row["source"]}/{ap_}')
            if not p:
                skipped += 1
                continue
            d = os.path.dirname(p)
            if d not in made:
                os.makedirs(d, exist_ok=True)
                made.add(d)
            with open(p, "wb") as fo:
                fo.write(row["audio"]["bytes"])
            n += 1
    os.remove(f)                                                # keep the disk to one copy of the audio
    return n, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--merge", action="store_true", help="restore into a mix dir other sources are being built into")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--skip_existing", action="store_true",
                    help="never rewrite a file that already exists (open-for-write truncates it: a training job reading "
                         "that file at the moment would get a partial clip); e.g. the verbatim set over a live en3")
    a = ap.parse_args()
    done = os.path.join(a.mix, f".restored_{a.repo.replace('/', '__')}")
    if os.path.exists(done) or (glob.glob(os.path.join(a.mix, "*.jsonl")) and not a.merge):
        print(f"fetch_mix: {a.mix} already restored / has manifests, skipping", flush=True)
        return
    # HF answers 5xx now and then (Oct 12 2026: a 504 on asr-english-v2's xet read token killed a box's setup):
    # retry with backoff; snapshot_download resumes, skipping the files it already has
    for attempt in range(8):
        try:
            d = snapshot_download(a.repo, repo_type="dataset", max_workers=16,
                                  local_dir=os.path.join(os.path.dirname(a.mix), "hf_mix", a.repo.replace("/", "__")))
            break
        except Exception as e:                                  # HfHubHTTPError, timeouts, connection resets
            if attempt == 7:
                raise
            wait = min(300, 15 * 2 ** attempt)
            print(f"fetch_mix: {a.repo} download failed ({type(e).__name__}: {str(e)[:120]}), retry in {wait} s",
                  flush=True)
            time.sleep(wait)
    for man in sorted(glob.glob(os.path.join(d, "manifests", "*.jsonl"))):
        for l in open(man, encoding="utf-8"):
            r = json.loads(l)
            WANT[f'{r["source"]}/{os.path.basename(r["audio_filepath"])}'] = r["audio_filepath"]
    kept = 0
    if a.skip_existing:
        for k in [k for k, p in WANT.items() if os.path.exists(p) and os.path.getsize(p) > 0]:
            del WANT[k]
            kept += 1
        print(f"fetch_mix: --skip_existing: {kept} files already present are left untouched", flush=True)
    shards = sorted(glob.glob(os.path.join(d, "data", "**", "*.parquet"), recursive=True))
    with Pool(a.workers, initializer=_init, initargs=(WANT,)) as p:
        res = p.map(_restore, shards, chunksize=1)
    n = sum(x for x, _ in res)
    missing = len(WANT) - n
    assert missing == 0, f"{missing} manifest rows have no audio in {a.repo}"
    os.makedirs(a.mix, exist_ok=True)
    for man in glob.glob(os.path.join(d, "manifests", "*.jsonl")):
        os.replace(man, os.path.join(a.mix, os.path.basename(man)))
    open(done, "w").close()
    print(f"fetch_mix: restored {n} files from {len(shards)} shards ({a.workers} workers) into {a.mix}", flush=True)


if __name__ == "__main__":
    main()
