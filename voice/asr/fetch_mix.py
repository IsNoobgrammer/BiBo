"""Fresh box: restore a build_mix.py output from its HF dataset instead of rebuilding from ~10 source datasets.

    python voice/asr/fetch_mix.py --repo fhai50032/asr-en-hi-400h --mix /home/marimo/work/asr/mix400

The repo holds the audio (data/*.parquet: audio bytes + source + file name) and the ORIGINAL manifests
(manifests/<source>.jsonl, with every field build_mix wrote -- e.g. AMI meeting / begin for real speaker turns). Audio
is written back to the exact paths the manifests name, so prep_train.py runs unchanged. Skips if already restored.
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mix", required=True)
    a = ap.parse_args()
    if glob.glob(os.path.join(a.mix, "*.jsonl")):
        print(f"fetch_mix: {a.mix} already has manifests, skipping", flush=True)
        return
    d = snapshot_download(a.repo, repo_type="dataset", local_dir=os.path.join(os.path.dirname(a.mix), "hf_mix"))
    want = {}
    for man in sorted(glob.glob(os.path.join(d, "manifests", "*.jsonl"))):
        for l in open(man, encoding="utf-8"):
            r = json.loads(l)
            want[f'{r["source"]}/{os.path.basename(r["audio_filepath"])}'] = r["audio_filepath"]
    n = 0
    for f in sorted(glob.glob(os.path.join(d, "data", "**", "*.parquet"), recursive=True)):
        for row in pq.read_table(f, columns=["audio", "source"]).to_pylist():
            ap_ = row["audio"]["path"]                              # "<source>/<file>" (old 400 h layout: "<file>")
            p = want.get(ap_ if "/" in ap_ else f'{row["source"]}/{ap_}')
            if p:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fo:
                    fo.write(row["audio"]["bytes"])
                n += 1
        os.remove(f)                                            # keep the disk to one copy of the audio
    missing = len(want) - n
    assert missing == 0, f"{missing} manifest rows have no audio in {a.repo}"
    for man in glob.glob(os.path.join(d, "manifests", "*.jsonl")):
        os.replace(man, os.path.join(a.mix, os.path.basename(man)))
    print(f"fetch_mix: restored {n} files into {a.mix}", flush=True)


if __name__ == "__main__":
    main()
