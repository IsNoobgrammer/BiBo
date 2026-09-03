"""Convert packed .arrow shards to parquet with a decoded-text column and push them to the Hub.

    python -m ablate.tools.push_corpus --src <packed_dir> --repo <user/name> [--no_raw]

Columns:
    label      list<int32>, exactly --seq tokens, no padding
    raw_label  str, `label` decoded back to text (omit with --no_raw)

There is no attention_mask column on purpose: the corpus is packed, so every position is a real
token and the mask would be all ones for every row -- a constant column costing storage and
inviting the same "it exists so it must be wired up" confusion that cost us a whole board.

`raw_label` is REDUNDANT -- it is recoverable by decoding `label` -- and it is several times larger
than the tokens it duplicates (Hindi runs ~9.9 bytes/token, so a 4096-token row is ~40 KB of text).
It exists for eyeballing the corpus without a tokenizer. Use --no_raw to skip it.

Auth comes from $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN in the environment. Do NOT pass a token on the
command line -- it lands in shell history and process listings.
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

# Xet high-performance transfer. Must be set BEFORE huggingface_hub is imported anywhere.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

TOKENIZER_JSON = "/tmp/qtk_patched.json"


def _convert(job):
    """Worker: one packed .arrow shard -> one zstd parquet. Returns (path, rows, bytes)."""
    src, out, tok_json, no_raw = job
    with pa.memory_map(src, "rb") as h:
        tbl = pa.ipc.open_stream(h).read_all()
    # Pass the ListArray STRAIGHT through -- it is already list<int32>, the output schema. The old
    # path did to_pylist() and rebuilt it with pa.array(), materializing 410M Python ints per shard
    # twice: 117s of the ~213s per shard, i.e. more than the upload it was blamed on.
    arr = tbl.column("input_ids").combine_chunks()
    cols = {"label": arr}
    if not no_raw:
        import gigatoken as gt
        dec = gt.Tokenizer(tok_json)
        # gigatoken.decode returns BYTES. errors="replace" because a row boundary cuts at a token
        # boundary, which for a byte-level BPE can land mid-UTF-8-character; one replacement char
        # at the seam is fine since raw_label is for inspection. `label` is exact regardless.
        cols["raw_label"] = pa.array(
            [(d.decode("utf-8", errors="replace") if isinstance(d, (bytes, bytearray)) else d)
             for d in (dec.decode(r) for r in arr.to_pylist())], type=pa.string())
    pq.write_table(pa.table(cols), out, compression="zstd")
    return out, len(arr), os.path.getsize(out)

CARD = """---
license: apache-2.0
language:
{langs}
size_categories:
- 1M<n<10M
---

# {repo}

Packed pretraining corpus, **{rows:,} rows x {seq} tokens = {btok:.3f}B tokens**, tokenized with
[fhai50032/QTK-81K](https://huggingface.co/fhai50032/QTK-81K).

## Format

| column | type | notes |
|---|---|---|
| `label` | `list<int32>` | exactly {seq} tokens, **no padding** |
| `raw_label` | `string` | `label` decoded back to text (redundant, for inspection) |

There is **no `attention_mask`** column: the corpus is packed, so every position is a real token and
the mask would be all ones on every row.

## How it was built

Documents are tokenized and appended to a running buffer with a single `<|im_end|>` (id **{sep}**)
between them. Every time the buffer reaches {seq} tokens a row is emitted and the remainder carries
into the next row. So no row is padded, no document is truncated, and a document longer than {seq}
tokens simply spans rows.

Source: `{src}`.

## Important: do not mask id 0

QTK-81K has `pad_token_id == eos_token_id == 0`. The separator here is `<|im_end|>` (81914), **not**
`<|endoftext|>`, so **id 0 never appears in this corpus**. Masking it is therefore a no-op and is
safe, but it is also unnecessary -- there is no padding to mask.

If you adapt this recipe to a corpus that separates documents with `<|endoftext|>`, masking id 0
would delete every document-boundary target and the model would never learn to stop -- a silent
failure that reads as degeneration rather than a bug.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--no_raw", action="store_true", help="skip the decoded-text column")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--tokenizer_json", default=TOKENIZER_JSON)
    ap.add_argument("--stage", default="/home/marimo/work/data/_stage")
    ap.add_argument("--langs", default="- hi")
    # capped by MEMORY, not cores: each worker holds a shard's decoded text (~4 GB for Hindi)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--upload_workers", type=int, default=8)
    a = ap.parse_args()

    tok_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert tok_env, "set HF_TOKEN in the environment; do not pass tokens on the command line"

    from huggingface_hub import HfApi
    api = HfApi(token=tok_env)
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)

    # the tokenizer is loaded per worker inside _convert, not here -- it does not pickle

    shards = sorted(glob.glob(os.path.join(a.src, "*.arrow")))
    assert shards, f"no .arrow shards under {a.src}"
    os.makedirs(a.stage, exist_ok=True)
    meta = {}
    mp = os.path.join(a.src, "pack_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    t0 = time.time()
    # PHASE 1 -- convert every shard to parquet, in parallel across cores. Workers are capped well
    # below nproc because each holds a whole shard's decoded text (~4 GB for Hindi), not because
    # of CPU.
    jobs = [(s, os.path.join(a.stage, f"train-{i:05d}.parquet"), a.tokenizer_json, a.no_raw)
            for i, s in enumerate(shards)]
    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_convert, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs)):
            out, rows, nbytes = fut.result()
            total += rows
            print(f"[push] converted {k+1}/{len(jobs)} {os.path.basename(out)} rows={rows} "
                  f"{nbytes/1e9:.2f}GB  {time.time()-t0:.0f}s", flush=True)
    t_conv = time.time() - t0
    print(f"[push] conversion done in {t_conv:.0f}s, {total:,} rows staged", flush=True)

    seq = meta.get("seq", 4096)
    card = CARD.format(repo=a.repo, rows=total, seq=seq, btok=total * seq / 1e9,
                       sep=meta.get("sep", 81914), src=meta.get("src", a.src), langs=a.langs)
    open(os.path.join(a.stage, "README.md"), "w", encoding="utf-8").write(card)

    # PHASE 2 -- one parallel, resumable upload of the whole folder. upload_file per shard was
    # strictly sequential AND never overlapped with the conversion above; upload_large_folder
    # runs many transfers at once and skips whatever already landed if it is re-run.
    t1 = time.time()
    try:
        api.upload_large_folder(repo_id=a.repo, folder_path=a.stage, repo_type="dataset",
                                num_workers=a.upload_workers)
    except AttributeError:                       # older hub: fall back to the folder uploader
        api.upload_folder(repo_id=a.repo, folder_path=a.stage, repo_type="dataset")
    print(f"[push] DONE {total:,} rows -> https://huggingface.co/datasets/{a.repo} "
          f"| convert {t_conv:.0f}s + upload {time.time()-t1:.0f}s = {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
