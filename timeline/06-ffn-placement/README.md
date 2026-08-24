# 06 - FFN placement: dense vs MoE end-layers

**Span** Aug 15-16 2026 &middot; `bibo-dense-vs-moe-2k`, 5 runs.

## Verdict: all-MoE SHIPPED-pending, wins 0.066 val for 4.9% throughput

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

## Sources

Memory: `dense-moe-round`, `box-tps-drift`, `carry-is-flat`. Results:
`ablate/certified_results/tps_probe_20260815.md`.
