"""Per-step VALIDATION loss on FROZEN batches, built once at startup and reused unchanged.

This is not the deleted eval. It is one no-grad forward per source, logged on the normal training
log line. Because the batches never change, these numbers are comparable across steps AND arms --
which train loss is not, since train loss moves with whatever the loader happens to feed.

TWO TIERS, logged separately and never averaged together:

  `val/loss`  -- THE NUMBER. A held-out slice of the ACTUAL TRAINING CORPUS (the last shard, which
                 data.split_shards keeps training away from). In-distribution, so it moves for the
                 same reasons train loss does and is the only thing here that can catch a broken
                 data pipeline. Rank arms on this.

  `val/ext/*` -- four EXTERNAL instruction-following sources, watched but never folded into the
                 headline. Scored as WHOLE-SEQUENCE cross-entropy: not answer-only, not
                 multiple-choice. GSM8K and Belebele are deliberately excluded (single-answer).

                     no_robots        en   human-written instruction + response
                     winogrande       en   pronoun-resolution sentences, both options as text
                     alpaca_hi        hi   instruction + input + output, Devanagari
                     evol_instruct_hi hi   multi-turn conversations, Devanagari

Mixing the two tiers into one average would be the mistake that hid the loader regression: an
in-distribution number and an out-of-distribution number moving in OPPOSITE directions is the
signal, and averaging them destroys exactly that.

Cost: `--val_seqs` sequences per source (default 2), 10 sequences of seq_len+1 total, one forward
each, no backward. Against a 262k-token training step that is noise.

Run the self-check (downloads the four external sources, no GPU):
    python -m ablate.common.validation
"""
from . import _paths  # noqa: F401
import torch
from .data import split_shards

TOKENIZER = "fhai50032/QTK-81K"


# ─────────────────────────── source extractors ───────────────────────────
# Each returns list[str] of whole texts. These are per-source ON PURPOSE: a generic
# "join every string column" helper silently produced EMPTY text for the two sources whose content
# lives in a list column, which then reads as a working validation set scoring nothing. The
# build-time asserts below exist to catch exactly that.

def _no_robots(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    out = []
    for ex in ds:
        turns = ex.get("messages") or []
        t = "\n".join(m.get("content", "") for m in turns if isinstance(m, dict))
        if t.strip():
            out.append(t)
        if len(out) >= n:
            break
    return out


def _winogrande(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    out = []
    for ex in ds:
        t = f"{ex['sentence']}\n{ex['option1']}\n{ex['option2']}"
        if t.strip():
            out.append(t)
        if len(out) >= n:
            break
    return out


def _alpaca_hi(n):
    from datasets import load_dataset
    ds = load_dataset("iamshnoo/alpaca-cleaned-hindi", split="train")
    out = []
    for ex in ds:
        t = "\n".join(x for x in (ex.get("instruction"), ex.get("input"), ex.get("output")) if x)
        if t.strip():
            out.append(t)
        if len(out) >= n:
            break
    return out


def _evol_hi(n):
    from datasets import load_dataset
    ds = load_dataset("FreedomIntelligence/evol-instruct-hindi", split="train")
    out = []
    for ex in ds:
        turns = ex.get("conversations") or []
        t = "\n".join(m.get("value", "") for m in turns if isinstance(m, dict))
        if t.strip():
            out.append(t)
        if len(out) >= n:
            break
    return out


SOURCES = [("no_robots", "en", _no_robots), ("winogrande", "en", _winogrande),
           ("alpaca_hi", "hi", _alpaca_hi), ("evol_instruct_hi", "hi", _evol_hi)]


def build_holdout(dataset, seq_len, n_seqs, device, field="input_ids"):
    """THE headline batch: rows from the held-out shard, already tokenized by the corpus builder.

    Read straight from the shard's `input_ids` -- no tokenizer, no re-encoding -- so this is byte
    for byte the same kind of input the training loader produces, and any divergence in the number
    is the MODEL, not a preprocessing difference. Returns None when the dataset is Hub-streamed,
    since no disjointness guarantee is possible there.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    _train, held = split_shards(dataset)
    if not held:
        return None
    n = seq_len + 1
    rows = []
    for f in held:
        # BOTH layouts: bip2 ships arrow IPC, the packed FineWeb corpora ship parquet with the
        # column named `label`. Reading a parquet file as an IPC stream fails with a confusing
        # "expected to read N metadata bytes" rather than a format error.
        if f.endswith(".parquet"):
            t = pq.ParquetFile(f).read_row_group(0)
        else:
            with pa.memory_map(f, "rb") as h:
                t = pa.ipc.open_stream(h).read_all()
        col = t.column(field) if field in t.column_names else t.column("label")
        for r in col.slice(0, max(n_seqs * 4, 64)).to_pylist():
            if len(r) >= n:
                rows.append(r[:n])                # the holdout is a fixed probe, not training data
            if len(rows) >= n_seqs:
                break
        if len(rows) >= n_seqs:
            break
    assert len(rows) == n_seqs, (f"holdout shard {held} yielded {len(rows)} usable rows of >= {n} "
                                 f"tokens, need {n_seqs}")
    return torch.tensor(rows, dtype=torch.long, device=device)

_DEVANAGARI = ("ऀ", "ॿ")


def _devanagari_frac(text):
    if not text:
        return 0.0
    return sum(1 for ch in text if _DEVANAGARI[0] <= ch <= _DEVANAGARI[1]) / len(text)


SEP_ID = 81914          # <|im_end|>, NOT <|endoftext|> (0)


def build(tokenizer, seq_len, seqs_per_source=2, device="cuda", sep_id=SEP_ID):
    """Freeze the external validation batches. Returns [(name, lang, LongTensor[K, seq_len+1])].

    Documents are joined with `<|im_end|>` (81914) rather than `<|endoftext|>` (0) ON PURPOSE: the
    CE below masks the pad id, and in QTK-81K pad_token_id == eos_token_id == 0, so an EOS separator
    would be silently deleted from the loss along with the padding. 81914 is never masked, so every
    separator is scored and the boundary is real supervision.

    Deterministic: fixed sources, fixed order, no shuffle, so two runs build a byte-identical batch
    and their losses are comparable.
    """
    n_needed = seqs_per_source * seq_len
    out = []
    for name, lang, loader in SOURCES:
        # pull enough documents to fill the chunks; these sets average 29-155 tokens/doc
        texts = loader(max(64, n_needed // 8))
        assert texts, f"validation source {name} returned no text"
        joined = "\n".join(texts)
        if lang == "hi":
            frac = _devanagari_frac(joined)
            assert frac > 0.30, (
                f"validation source {name} is tagged hi but only {100*frac:.1f}% Devanagari -- the "
                f"extractor is probably reading the wrong column and scoring empty or English text")
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t, add_special_tokens=False))
            ids.append(sep_id)
            if len(ids) >= n_needed + seqs_per_source:
                break
        n = seq_len + 1
        chunks = [ids[i * seq_len:i * seq_len + n] for i in range(seqs_per_source)]
        chunks = [c for c in chunks if len(c) == n]
        assert chunks, (f"validation source {name}: only {len(ids)} tokens, need "
                        f"{n_needed + seqs_per_source} for {seqs_per_source}x{n}")
        out.append((name, lang, torch.tensor(chunks, dtype=torch.long, device=device)))
    return out


@torch.no_grad()
def losses(model, holdout, batches, ce_fn, amp, pad_id=0):
    """Returns (headline, flat). `headline` is the held-out-corpus CE and is the ONLY ranking number;
    the external sources land under val/ext/* and are never averaged into it. Restores train mode.

    `pad_id` is ALWAYS masked, independently of the training --pad_id flag. Padding is trivially
    predictable, so scoring it would drag val/loss down in proportion to how much padding a corpus
    happens to carry -- an arm on a 16%-padded corpus would look better than an identical arm on a
    packed one. Fixed here so the number means one thing across every corpus. The external batches
    use <|im_end|> as their separator precisely so this mask cannot eat their document boundaries.
    """
    was_training = model.training
    model.eval()                      # so the MoE balancer bias update stays off (gated on .training)

    def _ce(ids):
        inp, tgt = ids[:, :-1], ids[:, 1:].reshape(-1)
        with amp:
            h = model.model(input_ids=inp, use_cache=False)
            h = h.last_hidden_state if hasattr(h, "last_hidden_state") else h[0]
            return float(ce_fn(h.reshape(-1, h.shape[-1]), model.lm_head.weight, tgt,
                               ignore_index=int(pad_id)))

    flat, by_lang, headline = {}, {}, None
    try:
        if holdout is not None:
            headline = _ce(holdout)
            flat["val/loss"] = headline
            flat["val/ppl"] = float(torch.exp(torch.tensor(headline)))
        for name, lang, ids in (batches or []):
            loss = _ce(ids)
            flat[f"val/ext/{name}"] = loss
            by_lang.setdefault(lang, []).append(loss)
    finally:
        if was_training:
            model.train()
    flat.update({f"val/ext/{lg}": sum(v) / len(v) for lg, v in by_lang.items()})
    return headline, flat


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    b = build(tok, seq_len=1024, seqs_per_source=2, device="cpu")
    assert len(b) == len(SOURCES), f"expected {len(SOURCES)} sources, built {len(b)}"
    assert {lang for _, lang, _ in b} == {"en", "hi"}, "both languages must be present"
    for name, lang, ids in b:
        assert ids.shape == (2, 1025), f"{name}: got {tuple(ids.shape)}, expected (2, 1025)"
        print(f"  val/ext/{name:16s} {lang}  {tuple(ids.shape)}  first ids {ids[0, :6].tolist()}")
    # determinism: a second build must be bit-identical, else val loss is not comparable step to step
    b2 = build(tok, seq_len=1024, seqs_per_source=2, device="cpu")
    assert all(torch.equal(a[2], c[2]) for a, c in zip(b, b2)), "external batch is NOT deterministic"
    # the masked pad id must not appear in the external batches, or the mask would delete real
    # supervision rather than padding -- this is the check that makes SEP_ID=81914 load-bearing
    for name, _lang, ids in b:
        assert (ids == 0).sum() == 0, (f"{name} contains pad id 0; the val CE masks it, so those "
                                       f"targets would vanish from the loss")

    # holdout tier, if a corpus path was given: python -m ablate.common.validation <dataset_dir>
    if len(sys.argv) > 1:
        tr, held = split_shards(sys.argv[1])
        h = build_holdout(sys.argv[1], 1024, 2, "cpu")
        assert h.shape == (2, 1025), f"holdout: got {tuple(h.shape)}"
        assert set(tr).isdisjoint(held), "holdout shard is also a training shard -- LEAKAGE"
        print(f"  val/loss (holdout)      {tuple(h.shape)}  from {len(held)} shard(s), "
              f"{len(tr)} train shards, disjoint")
        assert torch.equal(h, build_holdout(sys.argv[1], 1024, 2, "cpu")), "holdout NOT deterministic"
    else:
        print("  val/loss (holdout)      skipped -- pass a dataset dir to check it")
    print("[validation self-check] 4 external sources (2 en + 2 hi), deterministic, "
          "Devanagari asserted; holdout disjoint from train  OK")
