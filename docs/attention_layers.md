# Attention Layer Design Verdict — SWA vs Global (SSMax × Sink × Value-Scale)

> **Status:** forward-looking design norm (2026-07-01). BiBo is currently full-attention only
> (sliding-window was removed — see AGENTS.md design decision #1). This file is the **binding spec
> for when hybrid SWA lands** and for how SSMax/sink/value-scale combine on each layer type.
> Reference implementation + grad-checks + all supporting experiments: `src/.autoresearch/ssmax_sink_ref.py`.

---

## 1. The verdict (norm going forward)

| Layer type | SSMax | Attention sink | Value scaling |
|---|---|---|---|
| **Sliding-window (SWA)** | ❌ OFF (always) | ✅ ON — plain per-head bias, **unscaled** | ❌ OFF |
| **Global (full)** | configurable | configurable | ❌ OFF (always) |

**Global layers are one of three sanctioned configs** (ablation switch):

| Config | SSMax | Sink | Note |
|---|---|---|---|
| **G1 — SSMax + sink** | ✅ | ✅ (scaled by `C`, see §4) | full recipe |
| **G2 — SSMax only** | ✅ | ❌ | current BiBo behavior |
| **G3 — sink only** | ❌ | ✅ (unscaled) | isolates the sink |

The sink is a **single learnable scalar per head** (`shape [H]`, `requires_grad` per the training recipe),
appended as one value-less softmax column and dropped before the value matmul (GPT-OSS / MiMo-V2.5 style).

---

## 2. Config surface (to add when SWA is implemented — not yet in code)

Mirror MiMo-V2.5's split so sink is toggled **per attention type**:

- `hybrid_layer_pattern` / `layer_types` — which layers are `sliding_attention` vs `full_attention`.
- `add_swa_attention_sink_bias` — **forced True** for SWA (the norm; §3).
- `add_full_attention_sink_bias` — the sink toggle for global layers (G1/G3 = True, G2 = False).
- `use_ssmax` — already exists; **forced False on SWA layers**; selects G1/G2 (True) vs G3 (False) on global.
- Value scaling: **no config** — we do not add it (§5).

SWA layers are not free to opt out of the norm: sink ON, SSMax OFF, value-scale OFF.

---

## 3. Why SWA layers: sink only, unscaled, no SSMax, no value-scale

**No SSMax on SWA.** SSMax multiplies logits by `C = s·log(n)`, where `n` is the query's causal
context length. A fixed window caps `n` at `W` (e.g. 128), so past position `W` **`log(n)` is constant**
→ SSMax collapses to a per-head *constant temperature* (plus a tiny ramp over the first `W` tokens). It
provides **zero length-adaptivity** in a windowed layer — the thing SSMax exists for cannot happen when
`n` is pinned. If you ever want a per-head temperature on SWA, add it as an explicit constant, not as
SSMax; but QK-norm already controls logit scale, so we don't. **Cost of ignoring this:** per-position
`log(n)` compute + a learnable scale that trains toward a redundant constant. ⚠️ If SSMax *is* ever
enabled on a windowed layer, `n` **must be capped at `W`** (`n_j = min((kv_len−q_len)+j+1, W)`) — using
the absolute position would sharpen on tokens the layer cannot even see (a bug).

**Sink is exactly what SWA needs.** In a global layer the first token (BOS) is always visible, so the
model has a natural, always-available "dump bucket" (attention sink). In a **windowed** layer the window
slides *past* the first token — the natural sink **falls out of view** exactly when the window has moved
on, and every windowed query is then forced to fully distribute attention over its `W` local tokens even
when nothing is relevant. A learnable sink is **position-independent** — it can never slide out of the
window or be evicted — so every windowed query keeps its escape valve. This is the canonical reason
StreamingLLM / windowed models adopt sinks.

**Unscaled sink on SWA.** No SSMax means no `C`, so the sink is just `softmax([z_1…z_W, β])` — a plain
per-head bias. Nothing to scale.

**No value-scale.** See §5 (skipped everywhere).

---

## 4. Why global layers: three configs, and the SSMax×sink coupling

SSMax and the sink attack **complementary failure modes** on a global layer (verified in
`ssmax_sink_ref.py`, 100-step toy):

- **Something relevant exists** → logits have spread → SSMax sharpens onto the target (retrieval).
  Without SSMax (G3), bounded QK-norm cosines can't be sharpened → blurry retrieval.
- **Nothing relevant** → logits ≈ flat → SSMax has nothing to sharpen → softmax fades to `1/n`.
  The **sink** lets the head opt out instead of smearing a noise floor (G2 cannot).

That's why we keep all three: **G1** = both (best on the toy: retrieval 0.005, opt-out 0.001), **G2** =
SSMax-only (sharp retrieval, no opt-out), **G3** = sink-only (opt-out, blurry retrieval). Ablations pick
the winner on the real LM.

**The coupling rule for G1 (mandatory): scale the sink by the same `C`.** SSMax scales the query, so a
real logit is `C·z`. If the sink bias `β` is left unscaled, then as context grows `C` amplifies the real
logits while `exp(β)` stays fixed → **the sink's mass shrinks and the valve closes at long context**
(backwards — long context is when you most want to opt out). Fix (Option A): append `C·β`, i.e.

```
softmax over [ C·z_1, …, C·z_n , C·β ]   =   softmax( C · [z_1,…,z_n, β] )
```

so the whole row (sink included) shares one temperature `C`. `β` then behaves as a **learned per-head
threshold in unscaled logit units** — "opt out unless some token clears `β`" — and it stays meaningful
at every length. Verified in the script's valve demo: unscaled `p_sink` collapses to ~0 by `n=64`,
scaled stays usable to `n=1024+`. **G3 has no `C`, so its sink is unscaled by definition** — no coupling.

---

## 5. What we deliberately skip, and why

**Value scaling (fixed gain on the attention output) — skipped everywhere.**
It's a constant LayerScale on the residual write. (a) It's **redundant** with BiBo's learnable RMSNorm +
`o_proj`, which absorb any constant output gain. (b) It **double-counts** with the sink, which already
shrinks the output by `(1−p_sink)`. (c) Its only real use is fixed init/variance control, which BiBo
doesn't need. Net: no benefit, one more knob to tune. Skip.

**Per-head-*per-dimension* (virtual-key) sink — skipped.**
A per-dim sink is a learned key `k_sink ∈ ℝ^d` giving a *query-dependent* bar. It was floated as a fix
for the "fixed `β` drifts out of place at long context" worry. **It does not fix that.** The worry is:
the max of `n` random junk cosines grows ~`σ·√(2 ln n)` and eventually crosses `β` (extreme-value). A
virtual key is still *one fixed direction* the query must beat, facing the same rising tide. It changes
*where the bar sits per query*, not the tail. Measured (`escape_analysis()`):

- Escapes are pure q/k geometry — **SSMax does not change who escapes** (monotone temperature), only how
  loud they get. Escape **fraction is fixed by `(head_dim, β)` and identical at every `n`**; only the
  count scales ∝ n.
- Escapes are a **head_dim problem**. Junk cosine spread is `1/√d`; worst junk over 1M tokens is
  `~5.26/√d`. At **`head_dim=128` (BiBo default): worst ≈ 0.47**, so any `β∈[0.5, 0.85]` gives **~0
  escapes at 1M** while staying well below real matches (~0.9). At `β=0.5`: escape rate `7.7e-9`
  (`~0` tokens at 1M). (`head_dim=16` can't be saved — worst > 1 — but that's a toy artifact.)

Verdict: at our head_dim, a **scalar per-head `β` is sufficient** and the escape problem is bought off by
dimension + sane `β` placement, at **zero extra params/FLOPs**. Build a virtual-key sink **only** if a
genuinely different need appears — *query-adaptive opt-out sensitivity* (some queries should be pickier
than others within a head). We have no evidence BiBo needs that. YAGNI until we do.

---

## 6. Implementation notes

- **SSMax is SDPA-compatible** — it pre-scales the query, so `F.scaled_dot_product_attention` still
  fires. G2 keeps the fast path.
- **Sinks break the SDPA fast path** — an extra unmasked softmax column doesn't fit SDPA. Sink-bearing
  configs (G1, G3, all SWA) need **eager attention** (materialize scores → cat sink → softmax → drop) or
  a **sink-capable flash kernel** (lands in the separate kernels repo). Factor this into cost: sink
  layers are more expensive than SSMax-only layers.
- **The causal/sliding mask is applied to real keys only; the sink column is appended *after* the mask**
  (it must stay finite/visible to every query — that's the whole point on SWA).
- **Reference:** `src/.autoresearch/ssmax_sink_ref.py` — eager forward (grad-checked bit-for-bit vs SDPA
  with sink off, fwd + grads, float64; NaN-free with sink on), the 100-step complementary-regimes demo,
  the valve demo, the extrapolation sweep to 1M, and `escape_analysis()`. Grad-check any kernel against
  this before trusting it (Rules 1–2).

---

## 7. What to actually monitor (real runs, not synthetic)

The escape analysis assumes irrelevant keys behave like random unit vectors. Real learned keys may
cluster and have a fatter upper tail than `1/√d` predicts. So the metric to log on long-context runs is
**max-irrelevant-similarity vs `β` as sequence length grows** (not `p_sink`). If it tracks the
`σ·√(2 ln n)` curve, `β` stays placed and this is a closed issue. If the tail fattens, revisit — and even
then the fix is representation/normalization (or bumping head_dim), **not** a bigger sink.
