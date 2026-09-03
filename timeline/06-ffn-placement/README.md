# 06 - FFN placement: dense vs MoE end-layers

**Span** Aug 15-25 2026 &middot; `bibo-dense-vs-moe-2k`, 7 runs.

## Verdict: all-MoE SHIPPED-pending -- but the reason changed on the second seed

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
