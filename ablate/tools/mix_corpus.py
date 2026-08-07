"""Interleave two packed corpora at a fixed token ratio into one shuffled corpus.

    python -m ablate.tools.mix_corpus --src hi=<dir> en=<dir> --ratio hi=0.35 en=0.65 \
        --rows 2048000 --out <dir> [--holdout_rows 2000]

Rows are drawn from each source in order but placed in a DETERMINISTICALLY SHUFFLED sequence, so
the languages are interleaved row by row rather than arriving in blocks. Same seed => byte-identical
output, which is what makes two arms trained on this corpus comparable.

The last shard written is a HOLDOUT drawn from rows NOT used in training -- `data.split_shards`
reserves the final shard, so this becomes the in-distribution `val/loss` batch. It is deliberately
small: every row held out is a row training does not see.

Everything is vectorized through numpy and pa.ListArray.from_arrays; no token is ever a Python int
(that mistake cost 117s per shard elsewhere in this pipeline).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow as pa


class RowStream:
    """Sequential reader over a packed corpus's shards, yielding (n, seq) int32 blocks."""

    def __init__(self, src, seq):
        self.shards = sorted(glob.glob(os.path.join(src, "*.arrow")))
        assert self.shards, f"no .arrow under {src}"
        self.seq, self.i, self.buf, self.consumed = seq, 0, None, 0

    def _fill(self):
        while self.buf is None or len(self.buf) == 0:
            assert self.i < len(self.shards), "source exhausted -- not enough rows for --rows"
            with pa.memory_map(self.shards[self.i], "rb") as h:
                arr = pa.ipc.open_stream(h).read_all().column("input_ids").combine_chunks()
            self.buf = np.asarray(arr.values, dtype=np.int32).reshape(len(arr), self.seq)
            self.i += 1

    def take(self, n):
        out, need = [], n
        while need:
            self._fill()
            k = min(need, len(self.buf))
            out.append(self.buf[:k])
            self.buf = self.buf[k:]
            need -= k
        self.consumed += n
        return np.concatenate(out) if len(out) > 1 else out[0]


def write_shard(path, block, seq):
    n = len(block)
    values = pa.array(block.reshape(-1), type=pa.int32())
    offsets = pa.array(np.arange(n + 1, dtype=np.int32) * seq, type=pa.int32())
    arr = pa.ListArray.from_arrays(offsets, values)
    tbl = pa.table({"input_ids": arr})
    with pa.OSFile(path, "wb") as sink:
        with pa.ipc.new_stream(sink, tbl.schema) as w:
            w.write_table(tbl)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="name=dir, e.g. hi=/path en=/path")
    ap.add_argument("--ratio", nargs="+", required=True, help="name=frac, must sum to 1")
    ap.add_argument("--rows", type=int, required=True, help="TRAINING rows (holdout is extra)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--shard_rows", type=int, default=100_000)
    ap.add_argument("--holdout_rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42069)
    a = ap.parse_args()

    srcs = dict(s.split("=", 1) for s in a.src)
    ratio = {k: float(v) for k, v in (r.split("=", 1) for r in a.ratio)}
    assert set(srcs) == set(ratio), f"src {set(srcs)} != ratio {set(ratio)}"
    assert abs(sum(ratio.values()) - 1.0) < 1e-9, f"ratios sum to {sum(ratio.values())}"
    names = sorted(srcs)

    # exact per-source counts; the largest share absorbs the rounding remainder
    total = a.rows + a.holdout_rows
    want = {k: int(round(ratio[k] * total)) for k in names}
    big = max(names, key=lambda k: ratio[k])
    want[big] += total - sum(want.values())

    streams = {}
    for k in names:
        st = RowStream(srcs[k], a.seq)
        avail = 0
        for s in st.shards:
            with pa.memory_map(s, "rb") as h:
                avail += pa.ipc.open_stream(h).read_all().num_rows
        assert avail >= want[k], f"{k}: need {want[k]:,} rows, corpus holds {avail:,}"
        print(f"[mix] {k}: {want[k]:,} of {avail:,} rows ({100*ratio[k]:.0f}%)", flush=True)
        streams[k] = st

    rng = np.random.default_rng(a.seed)
    labels = np.concatenate([np.full(want[k], i, dtype=np.int8) for i, k in enumerate(names)])
    rng.shuffle(labels)                       # deterministic given --seed

    os.makedirs(a.out, exist_ok=True)
    t0, written, shard_i = time.time(), 0, 0
    # training shards first, holdout LAST so data.split_shards picks it up
    plan = [a.shard_rows] * (a.rows // a.shard_rows)
    if a.rows % a.shard_rows:
        plan.append(a.rows % a.shard_rows)
    plan.append(a.holdout_rows)

    for j, n in enumerate(plan):
        lab = labels[written:written + n]
        block = np.empty((n, a.seq), dtype=np.int32)
        for i, k in enumerate(names):
            pos = np.flatnonzero(lab == i)
            if len(pos):
                block[pos] = streams[k].take(len(pos))
        path = os.path.join(a.out, f"data-{shard_i:05d}.arrow")
        sz = write_shard(path, block, a.seq)
        written += n
        tag = "HOLDOUT" if j == len(plan) - 1 else ""
        print(f"[mix] wrote {os.path.basename(path)} rows={n} total={written} "
              f"{sz/1e9:.2f}GB {time.time()-t0:.0f}s {tag}", flush=True)
        shard_i += 1

    meta = {"sources": srcs, "ratio": ratio, "seq": a.seq, "seed": a.seed,
            "train_rows": a.rows, "holdout_rows": a.holdout_rows, "rows": written,
            "tokens": written * a.seq, "shards": shard_i,
            "consumed": {k: int(streams[k].consumed) for k in names},
            "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "pack_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[mix] DONE {written:,} rows ({written*a.seq/1e9:.3f}B tokens) in {shard_i} shards; "
          f"last shard ({a.holdout_rows:,} rows) is the validation HOLDOUT | "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
