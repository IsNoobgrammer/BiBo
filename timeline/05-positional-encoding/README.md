# 05 - Positional encoding

**Span** Aug 13-14 2026 &middot; `bibo-baseline-2k`, 4 arms x 2000 steps.

## Verdict: SHIPPED and then made unconfigurable

Full-attention layers get **NoPE**. Sliding-window layers get **full RoPE** over the whole
head_dim. There is no partial-rotary fraction, no per-layer-type width, no second base, and no NTK
scaling. Every one of those was a knob whose value the measurement settled, so it is now spelled in
the code instead of exposed as a flag (BiBo `55143cb`).

## The four arms

| Arm | Global layers (0,3,6,9) | Local layers | ctx4095 |
|---|---|---|---|
| `rope-g334-l334` | partial, 42/128 dims | partial, 42/128 | 3.4613 |
| `rope-g1-l1` | full, 128/128 | full, 128/128 | 3.3805 |
| `rope-gnope-l334` | **NoPE** | partial | 3.3245 |
| `rope-gnope-l1` | **NoPE** | **full** | **3.3107** |

## What we learned

**The global layers govern extrapolation; the local rotary is irrelevant to it.** With global NoPE,
moving the local width from 42 to 128 dims changed ctx4095 by 0.0037 -- inside the floor. With
local full, moving global from full to NoPE changed it by **0.0698**. That is mechanically right: a
window-128 layer only ever resolves relative distances <= 128, which is always in distribution, so
its rotary width cannot affect length generalisation.

**Rank on the absolute number, not the delta.** A uniformly worse model degrades less and wins on
`delta_ctx4095`. Our own data showed them disagreeing.

**A dead branch silently rescaled the base.** Dynamic NTK keyed off `max_position_embeddings` and
rescaled the base at eval: at seq 4095 the base became `10000 * 2^(42/40) ~= 20700`, while
ctx1024 and ctx2048 ran at 10000. Every pre-Aug-13 extrapolation number therefore compared two
different models. Removing NTK was a prerequisite for the round meaning anything.

**Deleting the knobs broke the model, and the tests did not catch it for a day.** `55143cb` removed
`_compute_inv_freq` but left `_init_weights` calling it, so every `build_arm` on the src path died
with AttributeError. Grep found the definitions; only the test suite would have found the callers.
Fixed in `f62731b`, with `test_rope.py` rewritten to assert the architecture -- windowed layers must
react to a position offset, full-attention layers must not.

## Caveat kept on the record

Winogrande favoured the control by 0.082 (8x its floor). It was not pre-registered so it did not
re-rank the result, but it is real and unexplained.

## Sources

Memory: `bibo-noise-floor`, `local-venv-testing`. Code: `src/configuration_bibo.py`,
`src/modeling/embed.py`, `src/modeling/attn/base.py`, `tests/test_rope.py`.
