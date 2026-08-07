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

import pyarrow as pa
import pyarrow.parquet as pq

TOKENIZER_JSON = "/tmp/qtk_patched.json"

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
    a = ap.parse_args()

    tok_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert tok_env, "set HF_TOKEN in the environment; do not pass tokens on the command line"

    from huggingface_hub import HfApi
    api = HfApi(token=tok_env)
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)

    dec = None
    if not a.no_raw:
        import gigatoken as gt
        dec = gt.Tokenizer(a.tokenizer_json)

    shards = sorted(glob.glob(os.path.join(a.src, "*.arrow")))
    assert shards, f"no .arrow shards under {a.src}"
    os.makedirs(a.stage, exist_ok=True)
    meta = {}
    mp = os.path.join(a.src, "pack_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    t0, total = time.time(), 0
    for i, s in enumerate(shards):
        with pa.memory_map(s, "rb") as h:
            tbl = pa.ipc.open_stream(h).read_all()
        ids = tbl.column("input_ids").to_pylist()
        cols = {"label": pa.array(ids, type=pa.list_(pa.int32()))}
        if dec is not None:
            cols["raw_label"] = pa.array([dec.decode(r) for r in ids], type=pa.string())
        out = os.path.join(a.stage, f"train-{i:05d}.parquet")
        pq.write_table(pa.table(cols), out, compression="zstd")
        api.upload_file(path_or_fileobj=out, path_in_repo=f"data/train-{i:05d}.parquet",
                        repo_id=a.repo, repo_type="dataset")
        total += len(ids)
        print(f"[push] {i+1}/{len(shards)} rows={len(ids)} total={total} "
              f"{os.path.getsize(out)/1e9:.2f}GB {time.time()-t0:.0f}s", flush=True)
        os.remove(out)
        del tbl, ids, cols

    seq = meta.get("seq", 4096)
    card = CARD.format(repo=a.repo, rows=total, seq=seq, btok=total * seq / 1e9,
                       sep=meta.get("sep", 81914), src=meta.get("src", a.src), langs=a.langs)
    cp = os.path.join(a.stage, "README.md")
    open(cp, "w", encoding="utf-8").write(card)
    api.upload_file(path_or_fileobj=cp, path_in_repo="README.md",
                    repo_id=a.repo, repo_type="dataset")
    print(f"[push] DONE {total:,} rows -> https://huggingface.co/datasets/{a.repo} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
