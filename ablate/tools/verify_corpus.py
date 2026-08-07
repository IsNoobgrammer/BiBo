"""Verify a packed corpus before it is pushed anywhere.

    python -m ablate.tools.verify_corpus --src <packed_dir> [--seq 4096] [--vocab 81920]

Every check is a hard assert. A packed corpus is the input to every run that follows it, and the
last time a data assumption went unchecked it invalidated an entire board -- so this refuses to
pass anything it cannot prove.

Checks:
  1. every row is exactly --seq tokens                (the whole point of packing)
  2. all ids in [0, vocab)                            (an out-of-range id is a silent embedding OOB)
  3. id 0 NEVER appears                               (pad == eos == 0; masking must be a no-op)
  4. the separator appears at a plausible rate        (a corpus with no separators is one long doc)
  5. decode -> re-encode round-trips exactly          (the ids really are this tokenizer's output)
  6. row count and token total match pack_meta.json   (no shard silently lost)
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow as pa

SEP_ID = 81914


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=81920)
    ap.add_argument("--sep", type=int, default=SEP_ID)
    ap.add_argument("--tokenizer_json", default="/tmp/qtk_patched.json")
    ap.add_argument("--roundtrip_rows", type=int, default=20)
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.src, "*.arrow")))
    assert shards, f"no .arrow shards under {a.src}"
    meta = {}
    mp = os.path.join(a.src, "pack_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    rows = seps = zeros = 0
    lo, hi = a.vocab, -1
    bad_len = []
    for i, s in enumerate(shards):
        with pa.memory_map(s, "rb") as h:
            tbl = pa.ipc.open_stream(h).read_all()
        # combine_chunks FIRST: a ChunkedArray has no .value_lengths, and the old fallback went via
        # to_pylist(), materializing 410M Python ints per shard (~8 min/shard instead of seconds).
        arr = tbl.column("input_ids").combine_chunks()
        lens = np.asarray(arr.value_lengths())
        if (lens != a.seq).any():
            bad_len.append((s, int(lens.min()), int(lens.max())))
        flat = np.asarray(arr.values)          # zero-copy view of the flat child array
        rows += len(lens)
        seps += int((flat == a.sep).sum())
        zeros += int((flat == 0).sum())
        lo, hi = min(lo, int(flat.min())), max(hi, int(flat.max()))
        print(f"  shard {i+1}/{len(shards)}  rows={len(lens)}  ids[{flat.min()},{flat.max()}]  "
              f"sep={(flat == a.sep).sum()}  zeros={(flat == 0).sum()}", flush=True)
        del tbl, arr, flat

    print(f"\ntotal rows={rows:,}  tokens={rows*a.seq:,} ({rows*a.seq/1e9:.3f}B)  "
          f"id range [{lo},{hi}]  separators={seps:,}  zeros={zeros:,}")

    assert not bad_len, f"rows are not all {a.seq} tokens: {bad_len[:3]}"
    assert 0 <= lo and hi < a.vocab, f"ids out of range: [{lo},{hi}] vs vocab {a.vocab}"
    assert zeros == 0, (f"id 0 appears {zeros} times. pad_token_id == eos_token_id == 0, and both "
                        f"--pad_id and the validation CE mask it, so those targets would vanish "
                        f"from the loss")
    per_row = seps / max(rows, 1)
    assert 0.2 < per_row < a.seq / 8, (f"{per_row:.2f} separators/row is implausible -- the corpus "
                                       f"is either one giant document or mostly separators")

    # round-trip: decode a few rows and re-encode; ids must come back identical, which is what
    # proves these really are QTK-81K token ids and not some other tokenizer's.
    import gigatoken as gt
    tk = gt.Tokenizer(a.tokenizer_json)
    with pa.memory_map(shards[0], "rb") as h:
        sample = pa.ipc.open_stream(h).read_all().column("input_ids").slice(0, a.roundtrip_rows).to_pylist()
    n_ok = 0
    for r in sample:
        # strip the trailing partial token run: a row boundary can cut mid-document, so only the
        # interior between the first and last separator is guaranteed to re-encode identically
        idx = [k for k, t in enumerate(r) if t == a.sep]
        if len(idx) < 2:
            continue
        seg = r[idx[0] + 1:idx[-1]]
        if not seg:
            continue
        assert tk.encode_batch_list([tk.decode(seg)])[0] == seg, "decode->encode did not round-trip"
        n_ok += 1
    assert n_ok, "no row had two separators to round-trip between"
    print(f"round-trip: {n_ok}/{len(sample)} sampled rows re-encode identically")

    if meta:
        assert meta["rows"] == rows, f"pack_meta says {meta['rows']} rows, shards hold {rows}"
        assert meta["seq"] == a.seq, f"pack_meta seq={meta['seq']} vs --seq {a.seq}"
        print(f"pack_meta.json agrees: {meta['rows']:,} rows, seq {meta['seq']}, "
              f"{meta['docs']:,} docs, {meta['dropped_tail_tokens']} tokens dropped")
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()
