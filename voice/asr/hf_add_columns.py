"""Add the teacher columns to a whole HF ASR dataset and upload it (after teacher_transcribe.py ran on that repo).

Input: the repo's concatenated manifests (the --manifest teacher_transcribe.py transcribed; audio_filepath basename +
source = the shard row key) and the transcripts (--hyp dir, hyp_*.tsv by manifest row). Every data/**/*.parquet shard
is downloaded (hf_transfer), streamed one at a time, given four columns by key, and written to <work>/<repo>/upload/
at the same path; then the folder goes up in one upload_large_folder (multi-worker, resumable) -- run it in the
background while the next repo transcribes.
  parakeet_transcription   stock Parakeet TDT 0.6B v2, full context, greedy
  parakeet_wer             WER % of the manifest reference against it (normalize_v2); null without a reference
  missing_speech           teacher >= 1.5x and >= 5 more words than the reference (untranscribed speech)
  garbage                  WER > 60 % with >= 4 reference words
Also writes <work>/<repo>/flags.tsv (key, missing, garbage, wer) for clean_by_teacher.py --flags.

    python voice/asr/hf_add_columns.py --repo fhai50032/asr-english --manifest R.jsonl --hyp HYPDIR --work W [--upload]
"""
import argparse
import glob
import json
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_by_teacher import wer_counts  # noqa: E402
from score import normalize_v2  # noqa: E402

COLS = ["parakeet_transcription", "parakeet_wer", "missing_speech", "garbage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--extra", type=int, default=5)
    ap.add_argument("--max_wer", type=float, default=0.6)
    ap.add_argument("--min_words", type=int, default=4)
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    wd = os.path.join(a.work, a.repo.replace("/", "__"))
    up = os.path.join(wd, "upload")
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    hyp = {}
    for p in glob.glob(os.path.join(a.hyp, "hyp_*.tsv")):
        for l in open(p, encoding="utf-8"):
            i, _, t = l.rstrip("\n").partition("\t")
            hyp[int(i)] = t
    col = {}                                                    # key -> (transcript, wer, missing, garbage)
    os.makedirs(wd, exist_ok=True)
    with open(os.path.join(wd, "flags.tsv"), "w", encoding="utf-8") as fl:
        for i, r in enumerate(rows):
            if i not in hyp:
                continue
            key = f'{r["source"]}/{os.path.basename(r["audio_filepath"])}'
            rw, hw = normalize_v2(r["text"]).split(), normalize_v2(hyp[i]).split()
            e, n = wer_counts(rw, hw)
            mi = len(hw) >= a.ratio * max(len(rw), 1) and len(hw) - len(rw) >= a.extra
            ga = not mi and n >= a.min_words and e / max(n, 1) > a.max_wer
            w = round(100.0 * e / max(n, 1), 2)
            col[key] = (hyp[i], w, bool(mi), bool(ga))
            fl.write(f"{key}\t{int(mi)}\t{int(ga)}\t{w}\n")
    print(f"[hf_cols] {a.repo}: {len(col)} transcribed rows of {len(rows)} manifest rows; "
          f"missing_speech {sum(v[2] for v in col.values())}, garbage {sum(v[3] for v in col.values())}", flush=True)
    src = snapshot_download(a.repo, repo_type="dataset", allow_patterns=["data/**/*.parquet"], local_dir=os.path.join(wd, "in"))
    shards = sorted(glob.glob(os.path.join(src, "data", "**", "*.parquet"), recursive=True))
    nohit = 0
    for k, s in enumerate(shards):
        rel = os.path.relpath(s, src)
        t = pq.read_table(s)
        t = t.drop_columns([c for c in COLS if c in t.column_names])
        paths = [x["path"] for x in t.column("audio").combine_chunks().field("path").to_pylist()] \
            if hasattr(t.column("audio").combine_chunks(), "field") else [x["path"] for x in t.column("audio").to_pylist()]
        srcs = t.column("source").to_pylist() if "source" in t.column_names else [None] * len(paths)
        vals = []
        for p, so in zip(paths, srcs):
            v = col.get(p if "/" in p else f"{so}/{p}")
            nohit += v is None
            vals.append(v or (None, None, False, False))
        t = (t.append_column(COLS[0], pa.array([v[0] for v in vals], pa.string()))
              .append_column(COLS[1], pa.array([v[1] for v in vals], pa.float32()))
              .append_column(COLS[2], pa.array([v[2] for v in vals], pa.bool_()))
              .append_column(COLS[3], pa.array([v[3] for v in vals], pa.bool_())))
        out = os.path.join(up, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pq.write_table(t, out, compression="zstd", row_group_size=pq.ParquetFile(s).metadata.row_group(0).num_rows)
        os.remove(s)
        if k % 10 == 0:
            print(f"[hf_cols] shard {k + 1}/{len(shards)} {rel}", flush=True)
    print(f"[hf_cols] {len(shards)} shards written to {up}; {nohit} shard rows without a transcript (null columns)", flush=True)
    if a.upload:
        os.makedirs(os.path.join(up, "annotations"), exist_ok=True)   # the key flags as a small file too
        shutil.copy(os.path.join(wd, "flags.tsv"), os.path.join(up, "annotations", "parakeet_flags.tsv"))
        HfApi().upload_large_folder(repo_id=a.repo, folder_path=up, repo_type="dataset", num_workers=8)
        print(f"[hf_cols] uploaded {a.repo}", flush=True)
        shutil.rmtree(up, ignore_errors=True)
    print(f"[hf_cols] HF_COLS_DONE {a.repo}", flush=True)


if __name__ == "__main__":
    main()
