"""Training data: the pre-tokenized QTK-81K instruct corpus (their own multi-domain, multi-lingual incl
Hindi amalgamation). Streams packed sequences and emits (batch, seq_len) token blocks.

`synthetic=True` yields random tokens for local smoke tests (no download)."""
from . import _paths  # noqa: F401
import torch

# QTK-81K packed instruct (verified: decodes to ChatML + Hindi, ids < 81920). Override via train.py --dataset.
TRAIN_DATASET = "tinycompany/Better-Instruct-packed-2"


def token_batches(batch, seq_len, device, dataset=TRAIN_DATASET, synthetic=False,
                  vocab=81000, seed=0, field="input_ids"):
    """Infinite generator of LongTensor (batch, seq_len+1) blocks (last token = next-token target)."""
    blk = batch * (seq_len + 1)
    if synthetic:
        gen = torch.Generator(device=device).manual_seed(seed)
        while True:
            yield torch.randint(0, vocab, (batch, seq_len + 1), generator=gen, device=device)
        return
    from datasets import load_dataset
    while True:                                    # loop the stream for multi-epoch token budgets
        ds = load_dataset(dataset, split="train", streaming=True)
        buf = []
        for ex in ds:
            ids = ex.get(field) or next(v for v in ex.values() if isinstance(v, list))
            buf.extend(ids)
            while len(buf) >= blk:
                chunk = torch.tensor(buf[:blk], dtype=torch.long, device=device).view(batch, seq_len + 1)
                buf = buf[blk:]
                yield chunk
