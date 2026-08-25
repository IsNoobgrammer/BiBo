# 10 - Per-layer interp: base-allmoe-s2026

The first run instrumented per layer end to end. One 2000-step all-MoE baseline, read through
expert-load histograms, the HSV load map, per-head XSA, per-expert radial p, and the AttnRes carry.

| | |
|---|---|
| Run | `base-allmoe-s2026` (`o35njyt7`), project `bibo-dense-vs-moe-2k` |
| Config | 805.14M total / 120.94M active, 10 layers, 64 experts top-6, all-MoE |
| Attention | `swa_pattern block3` = **G S S G S S G S S G**, window 128, global NoPE / local RoPE 1e4 |
| Tokens | 524M (2000 x 262,144), seed 2026, bf16, Muon |
| Result | **val 3.5952**, train 3.5947, 167.4k tps median |

---

## The headline: all-MoE reproduces

Three all-MoE runs now exist at this config:

| run | carry | seed | val |
|---|---|---|---|
| `ffn-C-allmoe` | raw | 42069 | 3.5829 |
| `ffn-C2-allmoe-sig` | sigmoid | 42069 | 3.5903 |
| `base-allmoe-s2026` | sigmoid | **2026** | 3.5952 |

Range **0.0123**, under the board's val noise floor of **0.0217**. The dense-end-layer arms sit at
3.6489 (`dense09`) and 3.6249 (`dense0`), so the all-MoE win of ~0.066 is now measured against a
spread rather than a single point. It survives.

The bounded carry costs nothing: 3.5903 and 3.5952 straddle 3.5829 within the floor.

---

## 1. Attention type is the organising variable

Every per-layer quantity that shows structure splits on **global vs sliding**, not on depth alone.
Layers 0, 3, 6, 9 are the global-attention layers, and they are also the AttnRes block boundaries
and the NoPE layers - a triple confound this run cannot separate. What follows is what the model
does, not which of the three causes it.

### Carry: global suppresses, sliding amplifies

`c = 2*sigmoid(theta)`, init 1.0, per dim, 512 dims per layer.

| layer | type | c (mean) | std |
|---|---|---|---|
| L0 | G | **0.757** | 0.079 |
| L1 | S | 0.966 | 0.104 |
| L2 | S | 1.062 | 0.077 |
| L3 | G | 1.046 | 0.115 |
| L4 | S | 1.198 | 0.081 |
| L5 | S | 1.220 | 0.107 |
| L6 | G | 0.921 | 0.109 |
| L7 | S | 1.297 | 0.110 |
| L8 | S | **1.498** | 0.102 |
| L9 | G | **0.651** | 0.131 |

Global mean **0.844**, sliding mean **1.207**. And the sliding layers are a clean monotone depth
ramp: 0.966, 1.062, 1.198, 1.220, 1.297, 1.498. Deeper windowed layers want *more* carried
attention residual; the global layers want less, and the two endpoints want least of all.

**The L0 collapse was the parameterisation.** Under a raw (unbounded) carry, `ffn-C-allmoe` drove
L0 to 0.009 by step 250 and left it there. Under the bounded form L0 settles at 0.757 and stops -
flat from step 250 to step 2000. So the earlier "layer 0 switches the carry off" was an unbounded
coefficient finding a free direction, not a preference of the model. Reading the raw run's L0 as a
statement about attention would have been wrong.

Note the std: 0.08-0.13 on every layer, on a mean near 1. The carry is a *narrow* per-dim
distribution, which is consistent with the standing result that the whole 512-dim vector is
replaceable by the scalar 1 at a cost under the noise floor. Per-dim freedom is not being used.

### XSA: one layer runs it backwards

`tanh(alpha)` per head, init 0 so the model switches it on itself.

| layer | type | h0 | h1 | h2 | h3 | spread |
|---|---|---|---|---|---|---|
| L0 | G | +0.726 | +0.438 | +0.150 | +0.473 | 0.576 |
| **L1** | **S** | **-0.530** | **-0.938** | **-0.628** | **-0.942** | 0.412 |
| L2 | S | +0.517 | -0.032 | +0.342 | +0.815 | **0.847** |
| L3 | G | +0.715 | +0.670 | +0.832 | +0.624 | 0.208 |
| L4 | S | +0.804 | +0.531 | +0.793 | +0.526 | 0.278 |
| L5 | S | +0.649 | +0.662 | +0.648 | +0.557 | 0.105 |
| L6 | G | +0.935 | +0.372 | +0.847 | +0.459 | **0.564** |
| L7 | S | +0.617 | +0.624 | +0.683 | +0.647 | 0.066 |
| L8 | S | +0.796 | +0.676 | +0.904 | +0.817 | 0.228 |
| L9 | G | +0.396 | +0.491 | +0.298 | +0.526 | 0.228 |

Two structures, both of which were previously recorded only in the **1B** XSA tables and are here
at 137M active:

- **L1 sign flip.** All four heads negative, three of them near saturation. This is not an
  initialisation artifact: L1 starts *positive* (+0.071 at step 100), crosses zero around step 200,
  and saturates at -0.78 by step 1000. The layer actively reverses the operation.
- **L6 head split.** Heads 0 and 2 at +0.94/+0.85, heads 1 and 3 at +0.37/+0.46. Two heads run XSA
  hard, two run it half. L2 splits even harder (+0.815 to -0.032, spread 0.847) with one head
  effectively off.

Every other layer is a tight bundle (L5, L7 spread under 0.11). Where per-head freedom matters, it
matters a lot; where it does not, the four heads agree to three decimals. That is the argument for
keeping alpha per head rather than per layer, and it only shows up in the per-head log.

### Radial p: a depth axis with two outliers

`p = sigmoid(theta)` per GLU expert, 64 per layer. p -> 0 is pure NormSiLU, p -> 1 is full
magnitude.

| layer | type | mean | std | range | experts p<0.1 | experts p>0.7 |
|---|---|---|---|---|---|---|
| L0 | G | 0.161 | 0.096 | 0.043-0.468 | 23 | 0 |
| **L1** | S | **0.077** | 0.037 | 0.018-0.320 | **52** | 0 |
| L2 | S | 0.356 | 0.142 | 0.067-0.686 | 1 | 0 |
| L3 | G | 0.155 | 0.115 | 0.041-0.557 | 25 | 0 |
| L4 | S | 0.289 | 0.112 | 0.084-0.745 | 2 | 1 |
| L5 | S | 0.537 | 0.087 | 0.219-0.740 | 0 | 1 |
| L6 | G | 0.203 | 0.160 | 0.019-0.632 | 22 | 0 |
| L7 | S | 0.284 | 0.158 | 0.012-0.725 | 7 | 1 |
| L8 | S | 0.335 | 0.179 | 0.024-0.921 | 3 | 4 |
| **L9** | G | **0.589** | 0.138 | 0.264-0.863 | 0 | **15** |

- Every layer leaves 0.5 and every layer is **done by step 500**, then flat for 1500 steps.
- The IQR widens with depth (0.12 at L0 to 0.22 at L8): late layers *differentiate* their experts,
  early layers keep them alike.
- **L1 collapses to normalisation.** 52 of 64 experts below p=0.1, IQR 0.037 - the layer is
  essentially pure NormSiLU with no expert-to-expert variation. L1 is also the XSA sign-flip layer.
  Two independent mechanisms both single out L1, which is the strongest per-layer signal in the run.
- **L9 goes the other way.** 15 experts above 0.7, none below 0.1, the highest mean in the model.
  The last layer wants magnitude preserved.
- Cross-layer p correlation at matched expert index is **-0.008** (L0 vs L8). Expert *i* in one
  layer has nothing to do with expert *i* in another, as it should - a nonzero value here would
  have meant a plumbing bug.

---

## 2. The router: balanced on the surface, saturated underneath

### Balance is genuinely solved

| layer | balance (uniform=1.0) | balance entropy | max load (x uniform) | dead |
|---|---|---|---|---|
| L0 | 0.937 | **0.905** | **6.17** | 1 |
| L1 | 0.992 | 0.948 | 3.48 | 2 |
| L2 | 0.971 | 0.987 | 3.19 | 0 |
| L3 | 0.990 | 0.973 | 3.70 | 0 |
| L4 | 1.000 | 0.991 | 2.45 | 0 |
| L5 | 1.000 | 0.992 | 2.00 | 0 |
| L6 | 0.996 | 0.980 | 3.48 | 1 |
| L7 | 1.042 | 0.979 | 2.64 | 0 |
| L8 | 1.027 | 0.981 | 2.38 | 0 |
| L9 | 0.994 | 0.979 | 3.33 | 1 |

No aux loss, bias-update balancing only, and the load-balancing quantity sits within 6% of perfect
on every layer. **This is why the per-layer histogram was worth building:** balance entropy reads
0.98 on a layer whose busiest expert is still taking 3.5x its share. The two summary statistics
that used to be the whole story are not sensitive enough to see it.

### The routers saturate

`router_z_loss` is `logsumexp(logits)^2`. At init, all logits are ~0 and logsumexp is exactly
`log(64) = 4.16` on every layer - the metric self-checks.

| layer | logsumexp final | growth |
|---|---|---|
| L0 | **31.0** | 7x |
| L1 | **198.9** | 48x |
| L2 | 137.0 | 33x |
| L3 | 80.9 | 19x |
| L4 | 71.5 | 17x |
| L5 | 138.6 | 33x |
| L6 | 156.0 | 37x |
| L7 | 85.5 | 21x |
| L8 | 101.0 | 24x |
| L9 | 118.4 | 28x |

With sigmoid gating and `router_z_loss_coef = 0`, the top logit reaches ~200. `sigmoid(200)` is
1.0 to machine precision: the leading experts are **saturated**, and their scores carry no gradient
and no ordering information. What separates rank 6 from rank 7 is the unsaturated tail plus the
balancing bias - which is what `boundary_gap` measures, and it sits at 0.02-0.10 in sigmoid units,
well inside the linear region.

So the picture is: a saturated head, a live tail, and a bias doing the tie-breaking. It works - the
loss is fine and balance is fine - but "the router learned to pick experts" is not what the logits
say. This is the one thing in the run that argues for testing a small `router_z_loss` coefficient,
and it was invisible before per-layer z was logged.

### L0 is a different animal

Every router metric singles out layer 0:

- **lowest logit scale** (31 vs 71-199) yet the **most concentrated load** (6.2x uniform vs 2.0-3.7)
- **lowest balance entropy** (0.905 vs 0.95-0.99)
- **smallest boundary gap** (0.0026 - the 6th and 7th choice are effectively tied)
- **one expert dead for 48 consecutive logged steps = 1200 training steps**, 60% of the run. Every
  other layer's longest dead run is 26 logged steps, and most are under 10.

A weak router with tied scores and a concentrated realised load means **the bias, not the score, is
choosing** at L0. And L0 is simultaneously the layer that suppresses its carry hardest (0.757) and
sits second-lowest on radial p. The entry layer behaves unlike the rest of the stack on every axis
measured, which is consistent with the earlier knockout result that removing L0's carry costs
+0.4256 - by far the largest of any layer.

### Expert identity churns; it does not specialise

Rank correlation of the per-expert load vector between step 500 and step 2000:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---|---|---|---|---|---|---|---|---|---|
| -0.08 | +0.25 | -0.01 | **+0.52** | +0.04 | +0.29 | +0.24 | +0.18 | +0.29 | +0.22 |

Adjacent logged steps (25 apart) correlate only +0.09 to +0.35. The busiest expert changes
identity between step 500, 1000, 1500 and 2000 in nine layers out of ten (L3 keeps expert 4).

This is not sampling noise: each traced step counts **1.63M assignments** over 64 experts, so the
per-expert standard error is ~1.3% while the observed spread is 200-600%. The load pattern is real
and it moves. At 2000 steps and 524M tokens, experts are not settling into stable roles - which
means any analysis that names "expert 37 is the code expert" at this scale is reading a snapshot of
a rotating assignment.

- **Dead experts are transient**: 54 expert-layers hit zero at some point, but almost all recover.
  The exception is L0's 1200-step outage.
- **Correlation between an expert's p and its load: |r| <= 0.21 on every layer.** How much traffic
  an expert gets and how it shapes its activation are independent axes. Two knobs, not one.

---

## 3. The load map

Rows are layers, columns are experts, colour is load relative to uniform: blue starved, white
uniform, red hot, black dead.

- **Step 0** is the most colourful frame in the run. With near-tied random scores, top-6 selection
  amplifies tiny logit differences, so the initial routing is *more* unequal than the trained one.
- **Step 500** shows the only structured row: L0 carries a contiguous blue block of starved experts
  with reds beside it. The rest of the stack is already pale.
- **Step 1975** is near-white with scattered blue and orange and a few black cells, and L0 is still
  the reddest row.

The map whitens over training. That is the bias update working, and it is legible in one image in a
way that ten 64-bin histograms are not.

---

## 4. Two bugs the instrumentation found

**Validation tokens were being counted into a train routing metric.** Each traced step recorded
1,634,304 assignments where `batch x seq x top_k x grad_accum` predicts 1,572,864. The 61,440
extra are 10 sequences of 1024 - the validation and extrapolation forwards, which run after the
flush with the hooks still armed, so their tokens land in the next interval's expert counts. 3.9%
of the histogram was held-out text. Fixed by disarming the hooks immediately after flush; the
numbers in this report carry the contamination and none of the conclusions turn on 3.9%.

**Router metrics were keyed by discovery order, not layer index.** Caught before this run. With
dense end-layers the first router lives in layer 1, so `layer_0` would have meant a different layer
in the router charts than in every other per-layer series in the same run.

---

## 5. What this says to do next

| | |
|---|---|
| **Test a small router z-loss** | Top logits at ~200 with a sigmoid gate means the leading experts are saturated and gradient-free. `router_z_loss_coef` is 0; marin runs 0 too, but their gate is a softmax over 256 where the scale means something different. |
| **L1 deserves its own arm** | Two unrelated mechanisms (XSA sign, radial p collapse) both single it out. Nothing else in the stack does that. |
| **Do not name experts at this scale** | Load rank churns between logged steps. Any per-expert story needs a stability check first. |
| **L0 is not a normal layer** | Weakest router, tied boundary, a 1200-step dead expert, hardest carry suppression, largest knockout cost. Whatever the entry layer is doing, it is not what layers 1-9 are doing. |
| **The carry may still be deletable** | std 0.08-0.13 on a mean near 1, and the earlier ablation put c=1 within 0.0033. The per-dim freedom is not being used. |

---

## Caveats

- **n = 1 for every per-layer number here.** The val loss has three seeds behind it; the carry,
  XSA, p and router tables have one. The six architecturally identical windowed layers vary by up
  to 0.33 on carry across earlier arms, so treat any per-layer difference smaller than that as
  unmeasured.
- **G/S, NoPE and AttnRes boundaries coincide.** Layers 0/3/6/9 are global, NoPE, and block edges
  at once. Nothing here can say which of the three drives the split.
- **`load_balancing_loss` is not comparable to marin's.** Their router is a softmax over 256
  experts; ours is a sigmoid over 64 with a bias. Compare our layers to each other only.
- **Histograms are one micro-batch**, not the whole interval. `dead_experts` means "took no token
  in the traced batch", which is why it flickers.
