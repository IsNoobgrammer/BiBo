"""Training data. Emits (batch, seq_len+1) token blocks by SPLITTING each dataset row into as many
sequences as it holds -- never truncating it.

Default is the packed FineWeb mix (4096 tokens/row, hi 35 / en 65). At seq_len 1024 each row yields
4 sequences; truncating instead would discard 75% of every row AND make every sequence a document
head, which is the bias that invalidated the previous board.

`synthetic=True` yields random tokens for local smoke tests (no download)."""
from . import _paths  # noqa: F401
import torch

# QTK-81K packed instruct (verified: decodes to ChatML + Hindi, ids < 81920). Override via train.py --dataset.
TRAIN_DATASET = "tinycompany/Better-Instruct-packed-2"


HOLDOUT_SHARDS = 1     # reserved for validation; training never sees them


def _shards(dataset):
    """Sorted .arrow shards if `dataset` is a local dir, else None (Hub streaming)."""
    import os, glob
    return (sorted(glob.glob(os.path.join(dataset, "**", "*.arrow"), recursive=True))
            if os.path.isdir(dataset) else None)


def split_shards(dataset):
    """(train_shards, holdout_shards). The LAST shard is held out so validation is measured on text
    training provably never reached -- shard granularity rather than a row count because the loader
    streams and would otherwise have to know the corpus length up front. Returns (None, None) for a
    Hub-streamed dataset, where no such guarantee is possible."""
    fs = _shards(dataset)
    if not fs:
        return None, None
    if len(fs) <= HOLDOUT_SHARDS:
        raise ValueError(f"{dataset} has {len(fs)} shard(s); need > {HOLDOUT_SHARDS} to hold one out")
    return fs[:-HOLDOUT_SHARDS], fs[-HOLDOUT_SHARDS:]


def _split(ids, n, seq_len, pad_id):
    """One row -> LIST of n-token sequences, stride seq_len so targets tile without overlap.

    TRUNCATING here instead of splitting is what wasted 75% of every row on the 4096-token packed
    corpus and made every training sequence a document HEAD -- the head-bias that invalidated the
    previous board. A 4096-token row yields 3 full chunks at stride 1024 plus a final chunk anchored
    at the end, so all 4096 tokens are used and nothing is a head by construction.
    """
    L = len(ids)
    if L < n:
        return [list(ids) + [pad_id] * (n - L)]
    out, i = [], 0
    while i + n <= L:
        out.append(ids[i:i + n])
        i += seq_len
    if L - i >= seq_len // 2:          # tail worth keeping: anchor one last chunk at the end
        out.append(ids[L - n:])
    return out


def token_batches(batch, seq_len, device, dataset=TRAIN_DATASET, synthetic=False,
                  vocab=81000, seed=0, field="input_ids", pad_id=0):
    """Infinite generator of LongTensor (batch, seq_len+1) blocks (last token = next-token target).

    Each row is SPLIT into ceil-ish many seq_len+1 sequences at stride seq_len; rows are never
    concatenated, so a sequence never spans two rows. On the packed corpus (no padding, documents
    separated by <|im_end|>) every position is real text.

    History, so neither mistake is repeated: the original loader flattened rows into a buffer and
    re-cut at batch*(seq_len+1), which put bip2's 329-token pad runs mid-sequence -- 16.06% pad in
    the forwarded tensor, 14.9% of real tokens carrying 329 pads in their causal context. Replacing
    that with TRUNCATION fixed the padding but discarded 75% of each 4096-token row and made every
    sequence a document head; that regressed the board and made AttnRes go from -0.0147 to +0.082.
    Splitting fixes both.

    Rows shorter than seq_len+1 are padded with `pad_id` (trailing, so unattendable); pass --pad_id
    to drop them from the loss. On the packed corpora this never fires.
    """
    n = seq_len + 1
    if synthetic:
        gen = torch.Generator(device=device).manual_seed(seed)
        while True:
            yield torch.randint(0, vocab, (batch, seq_len + 1), generator=gen, device=device)
        return
    from datasets import load_dataset
    # if `dataset` is a local dir of pre-downloaded .arrow shards, stream from DISK (robust — no HTTP
    # range reads that time out; download them with hf_transfer/xet first). Else stream from the Hub.
    # The LAST shard is excluded: it is the validation holdout (see split_shards).
    local_files, _held = split_shards(dataset)
    while True:                                    # loop the stream for multi-epoch token budgets
        ds = (load_dataset("arrow", data_files=local_files, split="train", streaming=True)
              if local_files else load_dataset(dataset, split="train", streaming=True))
        rows = []
        for ex in ds:
            # the packed FineWeb corpora name the column `label`; bip2 names it `input_ids`
            ids = (ex.get(field) or ex.get("label")
                   or next(v for v in ex.values() if isinstance(v, list)))
            for chunk in _split(ids, n, seq_len, pad_id):
                rows.append(chunk)
                if len(rows) == batch:
                    yield torch.tensor(rows, dtype=torch.long, device=device)
                    rows = []


if __name__ == "__main__":
    # a 4096-token row at seq_len 1024 must yield 4 chunks covering ALL 4096 tokens, not 1
    c = _split(list(range(4096)), 1025, 1024, 0)
    assert len(c) == 4, f"expected 4 chunks from a 4096 row, got {len(c)}"
    assert all(len(x) == 1025 for x in c), [len(x) for x in c]
    assert c[0][0] == 0 and c[-1][-1] == 4095, "chunks must span the whole row"
    assert c[1][0] == 1024 and c[2][0] == 2048, "stride must be seq_len"
    # short row still pads, and a sequence never carries tokens from the next row
    assert _split([7, 8, 9], 5, 4, 0) == [[7, 8, 9, 0, 0]]
    assert _split(list(range(2048)), 1025, 1024, 0)[0][-1] == 1024
    print(f"data self-check OK -- 4096-token row -> {len(c)} sequences, all 4096 tokens used")
