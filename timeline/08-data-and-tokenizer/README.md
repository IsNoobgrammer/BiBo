# 08 - Data and tokenizer

**Span** Aug 2026.

## Verdicts

| Work | Verdict | Number |
|---|---|---|
| Gigatoken tokenizer | **SHIPPED** | 10.1x faster than HF tokenizers, parity-gated 0/3200 |
| FineWeb 4096-packed hi/en/mixed corpus | **SHIPPED** on HF | sep 81914, int32 required |
| bip2 corpus | **trap found** | PADDED, not packed - 15.93% of tokens |

## What we learned

**A tokenizer ships on exact token-id equality, not on speed.** Gigatoken is 10.1x faster, but the
number that licensed the swap is **0 mismatches over 3200 documents**, Hindi and English, sampled
from the start, quarter, middle and late of the corpus. Same rule as kernels: bit identity or it
does not ship. QTK-81K needs `byte_fallback` cleared first, which is safe only because the vocab
has no byte tokens to fall back to - and again, the parity check is what licenses it, not the
argument.

**We were scoring padding.** bip2 is padded, not packed: 15.93% of its tokens are padding, and the
loader ignored `attention_mask`. Padding is trivially predictable, so scoring it drags val loss
down in proportion to how much padding a corpus happens to carry - an arm on a 16%-padded corpus
looks better than an identical arm on a packed one. **Always pass `--pad_id 0`**. `validation.losses`
now masks pad independently of the training flag, for exactly this reason.

**The loader must SPLIT rows, not truncate them.** A 4096-packed row feeding a 1024-context model
must become four sequences. Truncating silently discards three quarters of the corpus.

**vocab_size is 81920, not 81000.** The real `len(tokenizer)`; 81000 overflows.

## Sources

Memory: `gigatoken`, `fineweb-corpus-build`, `padded-corpus-pad-id`.
