# SSMax (Scalable-Softmax) — REFUTED at 524M, removed from BiBo

> **Status: REFUTED and disabled.** Measured on real BiBo pretraining at 524M tokens, August 2 2026.
> SSMax is **worse than the XSA baseline on training loss in every window from step 500 onward**
> (+0.0097 by `[1500, 2000)`) and worse on held-out bpb (+0.00263). It is off in the ablation
> (`use_ssmax=False`, pinned in `ablate/common/configs.py`) and slated for removal from `src/`.
>
> The mechanism is not broken — it does exactly what the paper says, and the long-context numbers
> below confirm that. It is simply **not what limits BiBo at our operating length**, and it charges
> for a benefit we cannot cash. The theory section is retained at the bottom because it remains
> correct and is the reason this was worth testing.

## What was measured

Five arms, all `seed 42069`, 64 experts, `top_k 6`, `64 x 4 x 1024`, 2000 steps = 524M tokens,
radial NormSiLU, cosine LR. **All five share an identical LR schedule** (`lr@0 = 5.00e-05`, warmup
peak at step 200) — verified before comparing, because a mismatched warmup invalidates everything.

### Training loss, mean per 500-step window

| arm | `[0,500)` | `[500,1000)` | `[1000,1500)` | `[1500,2000)` |
|---|---|---|---|---|
| **xsa + learnable alpha** (baseline) | 4.0052 | **2.0952** | **1.9048** | **1.7442** |
| control (no XSA, no SSMax) | 4.0032 | 2.1046 | 1.9076 | 1.7521 |
| **xsa + SSMax** | 4.0090 | 2.1071 | 1.9107 | 1.7539 |
| ptanh | 3.9961 | 2.1095 | 1.9121 | 1.7551 |
| xsa (fixed alpha) | 3.9874 | 2.1179 | 1.9152 | 1.7565 |

SSMax is **third of five** and loses to the baseline in every window after warmup. The ordering is
stable across all three late windows, so this is not window-selection noise.

### Held-out bpb

| arm | bpb | vs baseline |
|---|---|---|
| xsa + learnable alpha | **0.67505** | — |
| xsa (fixed alpha) | 0.67444 | −0.00061 |
| control | 0.67633 | +0.00128 |
| **xsa + SSMax** | **0.67768** | **+0.00263** |
| ptanh | 0.68096 | +0.00591 |

Training loss and bpb **agree in direction**, which is what makes this conclusive rather than a
single-metric artifact.

## Why it fails here — the hypothesis, and the evidence for it

**SSMax solves attention fading. At 1024 tokens BiBo does not have attention fading. We pay the
premium and collect nothing.**

`s` was learned enthusiastically — it is not inert, and this is not a plumbing failure:

| step | `s` (mean) | `s` range across heads | `C = s·log(1024)` |
|---|---|---|---|
| 0 | 0.1443 | [0.1443, 0.1443] | 1.000 |
| 250 | 0.1904 | [0.1401, 0.2497] | 1.320 |
| 500 | 0.2070 | [0.1533, 0.2740] | 1.435 |
| 950 | 0.2192 | [0.1579, 0.2765] | 1.519 |

`s` rose 52% off its neutral init and the heads spread 1.75× apart, so the model actively used the
knob. The cost is what it bought with it: `C` climbing to 1.52 means the model **re-tuned itself into
a sharper-attention regime**, and that regime is worse for next-token prediction at this length.

Two independent measurements support "the mechanism works, the problem is absent":

**1. Long-context degradation collapses — SSMax does its job.**

| | bpb @1024 | @2048 | @4096 | degradation 1024→4096 |
|---|---|---|---|---|
| xsa (baseline) | 1.30236 | 1.30943 | 1.35880 | **1.0433** |
| xsa + SSMax | 1.30822 | 1.30747 | **1.31474** | **1.0050** |

An **8.7× reduction** in length degradation, and bpb@4096 improves by 0.044. This is precisely the
advertised behaviour. It is also **worthless to us right now**: we train at 1024 and score at 1024.

**2. The regression concentrates where SSMax is weakest — short sequences.** The bpb eval is short
documents (belebele passages, gsm8k problems, a few hundred tokens). At `n ≈ 300`, `C = 0.219·log(300)
≈ 1.25`; at `n = 1024`, `C = 1.52`. The model optimized for the sharp end of the range and is scored
at the flat end. Per-source deltas are all positive-or-zero (worse) on the short-text sources.

### The honest caveat on significance

Per-source, the bpb regression is **not individually significant**:

| source | xsa | xsa+SSMax | delta | sigma |
|---|---|---|---|---|
| belebele_en | 1.27440 | 1.27890 | +0.00450 | 0.4 |
| belebele_hi | 0.55920 | 0.56130 | +0.00210 | 0.5 |
| gsm8k_en | 0.85340 | 0.86200 | +0.00860 | 0.9 |
| gsm8k_hi | 0.48130 | 0.48110 | −0.00020 | 0.0 |

No source clears 1σ, and our **between-seed** sigma is 0.007 (n=1 arms resolve little under 0.02).
**The bpb result alone would not carry this decision.** What carries it is the training-loss
comparison: monotone, ordered identically across three consecutive windows, on a metric with no
sampling error, against a matched LR schedule. bpb agrees in sign, which is corroboration rather than
proof.

Also worth recording: SSMax **improved ICL** (0-shot 0.20→0.30, 8-shot 0.73→0.78, lower NLL at every
shot count) and MCQ belebele_en (0.284→0.292). Those are real and they still did not save it — the
pretraining loss is the axis that decides.

## What would change this verdict

SSMax is refuted **for BiBo at 1024-token training**, not in general. Reopen it if:

1. **Training context grows to 8K+.** Attention fading is an `n → ∞` pathology. The 4096 numbers
   above say the mechanism is already paying off outside the training window; at a long training
   length the ledger could invert.
2. **Long-context retrieval becomes a target metric.** If needle-in-haystack or 32K+ serving enters
   the eval suite, the 8.7× degradation improvement stops being free money we throw away.
3. **`s` is re-anchored to the eval length rather than the training length.** The diagnosed cause is
   a train/eval temperature mismatch. Pinning `C ≈ 1` at eval-typical `n` instead of at `n = 1024`
   is a one-line change and directly targets the mechanism. **Untested** — it was the recommended
   next step when the axis was closed.

Anything reopening this needs a **second seed**, given the between-seed sigma above.

## Implementation notes, if it is ever restored

Two throughput bugs were found and fixed while this was live; both would recur in a naive
reimplementation.

**1. Two passes over `q` instead of one.** Written the obvious way, `q * ssmax_scale * log_n` walks
the full `(B, H, q_len, D)` tensor twice. The combined scale is `(1, H, q_len, 1)` — 8k elements — so
folding it first is free: `q * (ssmax_scale * log_n)`.

**2. Silent dtype promotion — the expensive one.** `ssmax_scale` is an fp32 Parameter and `q` is bf16
under autocast, so the product is **fp32**: a full-size fp32 `q` gets materialized, handed to SDPA
against bf16 `k`/`v` (measured `sdpa_in = (float32, bfloat16, bfloat16)`), then cast straight back
down. Fix is `.to(query_states.dtype)` on the combined scale.

Measured at `64 x 4 x 1024`:

| | tok/s | tax vs base |
|---|---|---|
| base | 174.7k | — |
| SSMax, naive | 168.1k | 3.8% |
| SSMax, folded scale | 170.0k | 2.7% |
| SSMax, folded + dtype fix | **172.6k** | **1.2%** |

Casting the scale is **not** gradient-neutral — `ds/dL` shifts by rel 1.9e-3 (worst head 8.5e-3),
because a bf16 output rounds each `grad_out·q` product before the fp32 reduction. That was accepted:
`grad_out` is bf16-valued regardless (SDPA emits bf16), so it matches every other gradient in the
model. `d(q·s)/ds = q`, so the scale's own rounding never enters the gradient.

**3. `n` must be per causal position.** Query `j` attends to `n_j = (kv_len − q_len) + j + 1` keys.
Using one global `log(kv_len)` collapses SSMax to a constant temperature during fixed-length
training — absorbable into the q/k weight norms, i.e. the mechanism does nothing at all. Under a
padding mask, `n` must come from `mask.cumsum(-1)`, not the grid position.

**4. SSMax must be applied AFTER QK-norm.** RMSNorm is scale-invariant, so scaling `q` beforehand is
erased exactly (measured: max abs difference 2.4e-06, i.e. fp32 rounding — a true no-op). There is no
design choice here; only one order is meaningful. Order versus RoPE does not matter — RoPE is a
rotation and scaling commutes with it exactly.

**5. Disable on sliding-window layers.** A window caps `n ≤ W`, so after the first `W` positions
`s·log(n)` is a fixed scalar and SSMax degenerates to a constant temperature — dead weight, not harm.

---

# Retained: the original theory

The analysis below is unchanged and remains correct. It is why the axis was worth testing. Nothing
here is contradicted by the refutation — the theory says SSMax fixes fading as `n → ∞`, and our
measurement says we do not have fading at `n = 1024`.

## The problem: attention fading

In standard attention, `attention_score_i = exp(z_i) / Σ exp(z_j)`. As `n` grows the denominator
grows while the numerator does not, so the maximum attention score approaches zero and the
distribution flattens. Formally:

```
max_output <= exp(z_max) / [(n-1)·exp(z_min) + exp(z_max)]
```

which tends to 0 as `n → ∞` unless `z_max − z_min` grows with `n`.

## The mechanism

```
SSMax(z_i) = n^(s·z_i) / Σ n^(s·z_j) = exp((s·log n)·z_i) / Σ exp((s·log n)·z_j)
```

with `s` learnable per head. Implemented by scaling queries — the softmax itself is untouched:

```python
log_n = log(n_per_position)          # (q_len,), broadcast as (1,1,q_len,1)
scaled_Q = Q * ssmax_scale * log_n
```

**Paper**: [Scalable-Softmax Is Superior for Attention](https://arxiv.org/abs/2501.19399), Nakanishi 2025.

## `log n` is the *critical* scaling (MIT, 2025)

[Critical attention scaling in long-context transformers](https://arxiv.org/abs/2510.05554) (Chen,
Lin, Polyanskiy, Rigollet) analyzes attention scaled as `β_n = γ·log n` — exactly SSMax with `γ ↔ s`
— and proves a phase transition governed by `γ* = 1/(1−ρ)`, where `ρ` is the background token–token
inner product:

| regime | condition | behavior |
|---|---|---|
| subcritical | `γ < γ*` | rank collapse — attention → uniform, vanishing gradients |
| critical | `γ = γ*` | sparse, content-adaptive attention (the target) |
| supercritical | `γ > γ*` | attention → identity-like, token interaction suppressed |

So `log n` is a knife-edge, not a heuristic, and `s` is a **critical threshold rather than a
bigger-is-sharper knob**. Both sides fail qualitatively. This also independently confirms the
no-SSMax-on-windowed-layers rule: rank collapse is an `n → ∞` phenomenon and a window caps `n ≤ W`.

Note this theory is about behaviour as `n → ∞`. Our refutation is a measurement at `n = 1024`. They
do not conflict; they describe different regimes, and the 4096 result is the theory being visible.

## QK-norm × SSMax (synthetic probe, June 2026)

A 2×2 on a synthetic passkey length-generalization probe (tiny model, train @128, eval to 32×, 3
seeds), extrapolation accuracy averaged over 256–4096:

| | no SSMax | SSMax |
|---|---|---|
| no QK-norm | 0.86 | 0.97 |
| QK-norm | 0.57 | 0.96 |

This was the strongest argument for SSMax in BiBo: QK-norm alone hurts length generalization
(0.86→0.57) because bounding the logits removes the model's ability to sharpen by growing Q/K
magnitude, and SSMax restores it (0.57→0.96).

**How to read this now.** It measured *length extrapolation on a synthetic passkey task*, and the
524M run reproduces that finding — degradation 1.0433→1.0050. It never measured pretraining loss at
the training length, which is the axis that decided the outcome. A synthetic probe on the metric a
mechanism is designed to improve will confirm the mechanism; it cannot tell you what the mechanism
costs on the objective you actually train.

## Overhead

1 parameter per head, 1 log, 1 multiply per query. For 12 layers × 12 heads: 144 parameters.
The real cost was never the parameters — see the dtype-promotion note above.
