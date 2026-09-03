# 02 - MoE routing

**Span** Jul - Aug 2026 &middot; `BiBo-Router` (13), `bibo-specials-conv` (176 -- the largest
project we have run), `bibo-baseline-2k`.

## Verdicts

| Axis | Verdict | Number |
|---|---|---|
| Load balance | **solved, free** | balance entropy 0.9996 with bias-update alone; no aux loss |
| Router gate shape | **CLOSED** | no variant separated |
| Router input norm | **CLOSED NEGATIVE** | per-token norm killed by icv = 0.0004 |
| Router temperature | **CLOSED** | T=2 best, T=0.5 worst, interior optimum |
| Signed +/- Identity experts | **CLOSED NEGATIVE twice** | 5.9-11.1% worse under AttnRes, 4-6 sigma |
| Zero expert | **RETIRED** | superseded by signed Identity, which then also lost |
| top-k 8 -> 6 at inference | **CLOSED NEGATIVE** | +0.00264 bpb, 7.1x the floor -- not free |
| top-k depth (dense/6/4) | **CLOSED NEGATIVE** at 10 layers | monotone: .6646 / .6727 / .6828 |
| MoE output RMS-norm | **CLOSED NEGATIVE** | ~0.010 bpb worse in all 3 variants |
| LatentMoE (d=256) | **CLOSED NEGATIVE** at 137M | costs 0.07-0.14 train loss |

## What we learned

**Flattening the router has an interior optimum.** Temperature 2 beat both 1 and 0.5. The intuition
that sharper routing means better specialisation is wrong in both directions, and the
router-boundary gap is *not* a sharpness proxy -- it moved independently of the thing it was
supposed to measure.

**Identity experts lost twice, on different stacks.** The second time under AttnRes at 4-6 sigma.
The code was deliberately retained (and extended to support asymmetric +/- counts) because the
mechanism is cheap to re-test, but the default is off and should stay off.

**"Effective k" reasoning does not survive measurement.** A prior estimate that only ~5.4 of 8
experts were doing useful work predicted that dropping to k=6 at inference would be free. Measured,
it costs 0.00264 bpb -- seven times the floor.

**Reducing top-k is a monotone dose, not a threshold.** dense 0.6646, top-6 0.6727, top-4 0.6828,
with bpb and generation agreeing. Kept on file because the conclusion is depth-dependent and this
was measured at 10 layers.

**The expert count semantics flipped and it is a live trap.** `num_routed_experts` is now the
TOTAL, with GLU experts derived as total minus specials. The same numbers therefore build a
*different* model before and after Aug 1 2026: at `num_experts=6, special_pairs=1` you now get
4 GLU + 2 specials, where the old code gave 6 GLU + 2 specials = 8 routed.

## Sources

Memory: `router-gate-axis`, `router-temperature`, `special-experts-round`,
`identity-experts-closed`, `topk-reduction-round`, `topk-depth-round`, `moe-output-norm`,
`latent-moe-round`, `bibo-src-debloat`.
