"""Training data: the pre-tokenized QTK-81K instruct corpus (their own multi-domain, multi-lingual incl
Hindi amalgamation). Emits (batch, seq_len+1) token blocks, ONE dataset row per sequence.

`synthetic=True` yields random tokens for local smoke tests (no download)."""
from . import _paths  # noqa: F401
import torch

# QTK-81K packed instruct (verified: decodes to ChatML + Hindi, ids < 81920). Override via train.py --dataset.
TRAIN_DATASET = "tinycompany/Better-Instruct-packed-2"


def _fit(ids, n, pad_id):
    """One row -> exactly n tokens: truncate if long, trailing-pad if short."""
    return ids[:n] if len(ids) >= n else list(ids) + [pad_id] * (n - len(ids))


def token_batches(batch, seq_len, device, dataset=TRAIN_DATASET, synthetic=False,
                  vocab=81000, seed=0, field="input_ids", pad_id=0):
    """Infinite generator of LongTensor (batch, seq_len+1) blocks (last token = next-token target).

    ONE dataset row -> ONE training sequence, TRUNCATED to seq_len+1. Rows are never concatenated.

    The previous version flattened every row into a running buffer and re-cut it at
    `batch*(seq_len+1)` strides, so a training sequence was an arbitrary window into the stream and
    routinely spanned two rows. On a corpus whose rows are content-then-trailing-pad that put the
    pad run in the MIDDLE of a sequence: measured 16.06% pad in the forwarded tensor, and 14.9% of
    real tokens carried 329 pads in their causal context. Truncation removes both -- the sequence
    ends where the row does, so nothing real ever attends across a boundary.

    Rows shorter than seq_len+1 are padded with `pad_id` (trailing, so still unattendable); pass
    --pad_id to drop them from the loss. On bip2 this never fires: every row holds >=1719 content
    tokens, so a 1025 cut is 0.00% padding.
    """
    n = seq_len + 1
    if synthetic:
        gen = torch.Generator(device=device).manual_seed(seed)
        while True:
            yield torch.randint(0, vocab, (batch, seq_len + 1), generator=gen, device=device)
        return
    from datasets import load_dataset
    import os, glob
    # if `dataset` is a local dir of pre-downloaded .arrow shards, stream from DISK (robust — no HTTP
    # range reads that time out; download them with hf_transfer/xet first). Else stream from the Hub.
    local_files = sorted(glob.glob(os.path.join(dataset, "**", "*.arrow"), recursive=True)) \
        if os.path.isdir(dataset) else None
    while True:                                    # loop the stream for multi-epoch token budgets
        ds = (load_dataset("arrow", data_files=local_files, split="train", streaming=True)
              if local_files else load_dataset(dataset, split="train", streaming=True))
        rows = []
        for ex in ds:
            ids = ex.get(field) or next(v for v in ex.values() if isinstance(v, list))
            rows.append(_fit(ids, n, pad_id))
            if len(rows) == batch:
                yield torch.tensor(rows, dtype=torch.long, device=device)
                rows = []


if __name__ == "__main__":
    assert _fit(list(range(1, 2049)), 1025, 0) == list(range(1, 1026))       # long row truncated
    assert _fit([7, 8, 9], 5, 0) == [7, 8, 9, 0, 0]                          # short row padded, trailing
    assert _fit([7, 8, 9, 10, 11], 5, 0) == [7, 8, 9, 10, 11]                # exact fit untouched
    # the invariant the old loader broke: a sequence never carries tokens from the next row
    assert _fit(list(range(1, 2049)), 1025, 0)[-1] == 1025
    print("data self-check OK")
