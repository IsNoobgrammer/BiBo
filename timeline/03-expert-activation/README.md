# 03 - Expert activation

**Span** Jul 22 - Aug 7 2026 &middot; `polyglu-ablations` / `-large` / `-XL` (38 runs),
`bibo-act-1b`, `bibo-baseline-2k`.

## Verdict: radial NormSiLU SHIPPED, then the axis saturated

`--act radial --radial_p sigmoid` is the default. `p = sigmoid(radial_theta)`, one value per GLU
expert, learned.

| Finding | Number |
|---|---|
| Normed activations beat plain silu | 3 configs + 500M priors agree |
| radial NormSiLU is the winner | beats silu in every case, transfers across systems |
| Activation axis **SATURATED** | three different mechanisms all land ~0.6767 |
| At 1B, normsilu still wins | but the gap **shrank 3.6x** vs 524M |
| Bounded parameters win, free ones break | the recurring shape of this axis |
| Mixing activations across experts | never wins |
| silu at 1B | kills 41 expert slots via a scale death-spiral |

## What we learned

**Three unrelated mechanisms converge on the same loss.** Once we found ~0.6767, a conditioning
tweak, a normalisation tweak and a radial exponent all reached it. That is the signature of a
saturated axis: stop spending runs on it.

**The win shrinks with scale.** normsilu beats silu by 3.3x the floor at 1B -- but the gap was 3.6x
larger at 524M. An activation advantage measured at small scale is an upper bound on what you get
at large scale, not a preview.

**p is a depth ramp, and a phase switch.** Radial p rises with depth, with a phase switch around
layers 4-5, and the outlier ratio anti-correlates with p. In the Aug 16 interp round the ramp was
confirmed from layer 3 onward in two independent models -- but the early-layer profile (L0-L2) was
**not** stable, so only the L3+ ramp should be quoted.

**A flag that does not reach the eager path is a silently different model.** `ACT_CYCLE` steered
only the patched forward while the eager expert loop indexed a hardcoded
`("silu","relu2","normsilu")[e % 3]`, so 2/3 of experts ran the wrong activation and corrupted
downstream routing. Now both paths read the same `radial_theta` parameter, and `interp_deep.py`
carries a selfcheck that recomputes one layer from raw weights and aborts on disagreement.

**p needs its own learning rate** (0.01, via `--act_scale_lr`); at the shared LR it barely moves.

## Sources

Memory: `expert-activation-round`, `radial-adoption`, `scale-freedom-round`, `act-1b-round`,
`activation-conditioning-round`, `radial-repro-interp`, `bibo-src-debloat`.
