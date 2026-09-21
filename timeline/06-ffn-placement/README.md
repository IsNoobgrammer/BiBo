# 06 - FFN placement: dense vs MoE end-layers

**Span** Aug 15-25 2026 &middot; `bibo-dense-vs-moe-2k`, 7 runs.

## Verdict: SUPERSEDED Sep 17 2026 -- layer 0 now runs an all-active ensemble

**Both headline claims in this page failed their seed controls.** Read the update at the bottom
first; the val ladder and the extrapolation separation are each contradicted by a control run
added after the page was written.

**Read the update at the bottom before quoting the 0.066.** A second seed reversed the sign of the
val gap. The result survives, on a different and much stronger metric: long-context extrapolation.

The board config ran a dense FFN at layers 0 and 9 and a 64-expert top-6 MoE everywhere else. The
question was whether those two layers want expert capacity too.

| Arm | Dense layers | Total | Active | val | ctx4095 | tps |
|---|---|---|---|---|---|---|
| A | 0, 9 | 668.24M | 120.88M | 3.6489 | 3.4613 | 179.6k |
| B | 0 | 736.69M | 120.91M | 3.6249 | 3.4219 | 175.1k |
| **C** | none | 805.14M | 120.94M | **3.5829** | **3.3098** | 170.9k |

Monotone on every metric in dose order: val -0.0240 then -0.0420.

## Matching active params is what made this an ablation

At the board default `intermediate_size=1024` a dense layer costs `3*h*1024` = 1.57M active while
the MoE block beside it costs `top_k*3*h*768 + h*E` = 7.11M -- **4.5x more**. Swapping one for the
other moved active params +5% per layer, so the naive arm would have been a capacity arm wearing a
dense-vs-MoE name. Parity needs no tuning: `dense_inter = top_k * moe_inter = 4608`, which is the
FLOP-parity rule `configs.py` already documented and the board config had never satisfied.

New flags: `--mlp_only_layers` (comma list or `none`) and `--dense_inter`.

## Replication

Both arms were re-run under a bounded carry (`sigmoid`): 3.5903 vs 3.5829 for all-MoE, 3.6268 vs
3.6249 for dense-0. The ordering and the effect size do not depend on the carry parameterisation.

## Cost, measured properly

The tps numbers above come from a **paired interleaved probe**, not from the training runs. The box
those trained on drifted ~20% over one afternoon with the GPU reading perfectly healthy, so their
own tps is unusable. Interleaved A-B-C twice: pass1 == pass2 to 0.1k, power-cap counter unchanged.
Each dense->MoE swap costs a consistent ~2.5%.

At that exchange rate 4.9% is cheap: spending it on A instead buys ~98 extra steps, worth ~0.010
val at A's end-of-run slope, against the 0.066 all-MoE delivers.

## The caveat that must travel with this result

Under `swa_pattern block3`, layers 0 and 9 are **also** the full-attention layers, **also** the
AttnRes block boundaries, and **also** the NoPE layers. "Dense FFN" is confounded with all three.
This is "MoE-everywhere helps GIVEN this stack", never "dense end-layers are bad".

## UPDATE, Aug 25 2026: a second seed, and val was the wrong metric

`base-allmoe-s2026` and `dense0-s2026` repeat arms C and B at **seed 2026**, matched active params,
bounded carry, full per-layer instrumentation.

| arm | seed | val | train | ctx1024 | ctx4095 | delta_ctx4095 |
|---|---|---|---|---|---|---|
| allmoe (C) | 42069 | 3.5829 | 3.6053 | 3.2687 | **3.3098** | **0.0411** |
| allmoe (C2, sigmoid) | 42069 | 3.5903 | 3.6022 | 3.2702 | **3.3223** | **0.0521** |
| allmoe (s2026) | 2026 | 3.5952 | 3.5947 | 3.2619 | **3.3156** | **0.0537** |
| dense0 (B) | 42069 | 3.6249 | 3.6117 | 3.2807 | 3.4219 | 0.1412 |
| dense0 (B2, sigmoid) | 42069 | 3.6268 | 3.6111 | 3.2720 | 3.4084 | 0.1364 |
| dense0 (s2026) | 2026 | **3.5813** | 3.6004 | 3.2569 | 3.4082 | 0.1513 |
| dense09 (A) | 42069 | 3.6489 | 3.6397 | 3.3051 | 3.4613 | 0.1562 |

**val flipped sign.** At seed 42069 all-MoE beat dense-0 by 0.037; at seed 2026 dense-0 beat
all-MoE by 0.014. Two runs of the same dense-0 config differ by **0.0455** across seeds -- twice
the 0.0217 floor the round was reading against. `val/loss` is scored on `--val_seqs 2`, a
two-sequence holdout. It is too small to carry a 0.03 effect and it never was.

**Every other metric held, and one separated cleanly.** Train loss (524M tokens) puts all-MoE ahead
at both seeds, by 0.0089 and 0.0057 -- consistent, and an order of magnitude smaller than the val
gap that was being quoted. And `delta_ctx4095`, the extrapolation penalty, separates the arms
**3 against 4 with no overlap**:

    all-MoE   0.0411  0.0521  0.0537
    dense     0.1364  0.1412  0.1513  0.1562

A 2.7x penalty, three runs each, both seeds, both carry parameterisations. At the trained length
(ctx1024) the arms are within 0.024 of each other. **The whole effect is at 4x extrapolation.**

**And it is layer 0, not capacity.** dense09 (0.1562) and dense0 (0.1364-0.1513) are the same
within their own spread, so the second dense layer adds nothing to the damage. Under `block3`,
layer 0 is a NoPE full-attention layer, and the rope round already established that the global
layers govern extrapolation. Putting a dense FFN on the entry NoPE layer is what costs the
long-context behaviour.

**Restated verdict:** all-MoE wins on long-context extrapolation by 2.7x on the degradation metric,
consistently across seeds; it wins on train loss by ~0.007, consistently; and the val gap is noise
that happened to point the right way at the first seed. The 4.9% throughput price now buys
extrapolation, not 0.066 val.

## Sources

Memory: `dense-moe-round`, `box-tps-drift`, `carry-is-flat`. Results:
`ablate/certified_results/tps_probe_20260815.md`.


---

## UPDATE 2, Sep 10-17 2026: the extrapolation claim failed too, and L0 changed

`base-allmoe-s23` -- the control this round never had -- is all-MoE at a second seed:

    all-MoE   delta_ctx4095  0.0411  0.0521  0.0537  |  **0.1932**  (seed 23)
    dense                    0.1364  0.1412  0.1513
    ensemble  (8x576 all-on) 0.1516  0.1629  0.1691

**all-MoE at seed 23 extrapolates worse than every dense and ensemble run.** Its seed range on
this metric, 0.041 to 0.193, is larger than the 0.098 effect Update 1 reported as "3 against 4
with no overlap". That sentence is wrong as written. What is true: sparse L0 has a better MEAN
(0.085 vs 0.161) and a much worse tail. `train` and `ctx1024` are seed-stable; only
`delta_ctx4095` is unstable, and only for the sparse arm.

### The adopted stack

Layer 0 now runs **8 experts at width 576, all active** -- the router produces mixing weights and
makes no selection. `8 x 576 = 4608 = top_k * moe_inter`, so L0 costs the same per token as the
sparse layer it replaces and 68.4M less in total. Layers 1-9 are unchanged 64-expert top-6 MoE.
Encoded in `SHARED` and pinned by a test.

| | all-MoE L0 | ensemble L0 |
|---|---|---|
| train (2 seeds) | 3.5940 | 3.5923 |
| ctx1024 (2 seeds) | 3.2586 | **3.2505** |
| delta_ctx4095 mean | **0.0850** | 0.1612 |
| delta_ctx4095 range | 0.152 | **0.0175** |
| L0 params | 75.5M | **7.08M** |

Consistency, size and a simpler entry layer, bought with worse expected long-context. A deliberate
trade, not a free win -- revisit if long-context serving becomes the target.

### What the checkpoints say about why

`ablate/tools/length_probe.py` runs a frozen checkpoint at two context lengths using ctxabl's own
construction (same targets, varying history):

- **L0 is the most length-sensitive router in the model** -- weight TV distance 0.184 between
  ctx1024 and ctx4095, against 0.016-0.046 everywhere else.
- **The ensemble's L0 does not move at all** (weight TV 0.0030, load TV exactly 0). It is
  functionally dense, which is why it matched dense-L0's extrapolation.
- **Zeroing L0's balancing bias changes nothing** (0.1817 -> 0.1799), killing the stale-bias
  hypothesis.

And the mechanism guess in Update 1 was backwards: seed 23 has a *healthy* L0 router (boundary gap
0.112 vs seed 2026's 0.0026), the highest length sensitivity, and the worst extrapolation. Routing
that SHIFTS under length change looks like the problem, not the cure. Untested.

**Capacity explains none of it.** L0 holds 75.5M params in both all-MoE (64x768 top-6) and coarse
(32x1536 top-3), and those sit at opposite extremes: 0.054 and 0.757.

### Two method lessons, both expensive

1. **Run the control before the conclusion.** Three of this round's claims were written from arms
   compared against a single-seed reference. Two did not survive the second seed.
2. **A nondeterminism floor is not a seed floor.** Two accidental same-seed repeats gave a 0.0004
   train spread; the seed spread is 0.0043. Scoring a 0.0038 margin against the smaller one made
   noise look like a result.
