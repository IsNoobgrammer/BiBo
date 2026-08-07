"""Pack a raw text corpus into fixed-length token rows with NO padding.

    python -m ablate.tools.pack_corpus --src <parquet_dir> --out <arrow_dir> [--seq 4096]

Documents are tokenized and appended to a running buffer with a separator between them; every time
the buffer reaches `--seq` tokens a row is emitted and the remainder carries into the next row. So
every row is exactly `--seq` real tokens, nothing is padded, nothing is discarded except the final
partial buffer, and a document longer than `--seq` simply spans rows.

SEPARATOR is `<|im_end|>` (81914), NOT `<|endoftext|>` (0). In QTK-81K pad_token_id == eos_token_id
== 0, and both the training `--pad_id` and the validation CE mask the pad id -- an EOS separator
would therefore be deleted from the loss wherever masking is on, silently removing every document
boundary. With 81914, id 0 never appears in the corpus at all, so masking it is a guaranteed no-op.

Tokenization is gigatoken (~10x the HF tokenizer, exact-parity checked). Output is Arrow IPC stream
shards with a single `input_ids` column, which is what ablate/common/data.py globs and reads.
Shards are written incrementally so memory stays flat regardless of corpus size.
"""
import argparse
import glob
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

SEP_ID = 81914          # <|im_end|>
TOKENIZER_JSON = "/tmp/qtk_patched.json"     # QTK-81K with byte_fallback cleared (see CLAUDE.md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of raw .parquet shards with a `text` column")
    ap.add_argument("--out", required=True, help="dir to write packed .arrow shards into")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--shard_rows", type=int, default=100_000)
    ap.add_argument("--sep", type=int, default=SEP_ID)
    ap.add_argument("--tokenizer_json", default=TOKENIZER_JSON)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = consume the whole source")
    ap.add_argument("--batch_rows", type=int, default=20_000, help="parquet read batch")
    a = ap.parse_args()

    import gigatoken as gt
    tk = gt.Tokenizer(a.tokenizer_json)

    srcs = sorted(glob.glob(os.path.join(a.src, "**", "*.parquet"), recursive=True))
    assert srcs, f"no parquet under {a.src}"
    os.makedirs(a.out, exist_ok=True)
    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    print(f"[pack] {len(srcs)} source shards -> {a.out} | seq={a.seq} sep={a.sep} "
          f"shard_rows={a.shard_rows}", flush=True)

    buf, rows, total_rows, shard_i, t0 = [], [], 0, 0, time.time()
    n_docs = 0
    dropped_tail = 0

    def flush_shard():
        nonlocal rows, shard_i
        if not rows:
            return
        path = os.path.join(a.out, f"data-{shard_i:05d}.arrow")
        tbl = pa.table({"input_ids": pa.array(rows, type=pa.list_(pa.int32()))}, schema=schema)
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_stream(sink, schema) as w:
                w.write_table(tbl)
        print(f"[pack] wrote {path}  rows={len(rows)}  total={total_rows}  "
              f"{time.time()-t0:.0f}s", flush=True)
        rows = []
        shard_i += 1

    stop = False
    for f in srcs:
        if stop:
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=a.batch_rows, columns=["text"]):
            texts = [t for t in batch.column("text").to_pylist() if t]
            if not texts:
                continue
            n_docs += len(texts)
            for ids in tk.encode_batch_list(texts):
                buf.extend(ids)
                buf.append(a.sep)
                while len(buf) >= a.seq:
                    rows.append(buf[:a.seq])
                    del buf[:a.seq]
                    total_rows += 1
                    if len(rows) >= a.shard_rows:
                        flush_shard()
                    if a.max_rows and total_rows >= a.max_rows:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
    dropped_tail = len(buf)
    flush_shard()

    meta = {"src": a.src, "seq": a.seq, "sep": a.sep, "rows": total_rows, "docs": n_docs,
            "tokens": total_rows * a.seq, "shards": shard_i,
            "dropped_tail_tokens": dropped_tail, "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "pack_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[pack] DONE rows={total_rows} ({total_rows*a.seq/1e9:.3f}B tokens) from {n_docs} docs "
          f"in {shard_i} shards | dropped final partial buffer of {dropped_tail} tokens "
          f"| {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
