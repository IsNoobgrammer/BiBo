# 04 - Attention

**Span** Aug 1-13 2026 &middot; `bibo-xsa-524m` (17), `bibo-attnres` (30), `-v2` (23), `-1b` (4),
`bibo-baseline-2k`.

## Verdicts

| Axis | Verdict | Number |
|---|---|---|
| XSA with learnable per-head alpha | **SHIPPED** | beat fixed full-strength XSA by 20x the floor |
| XSA alpha init | **SHIPPED at 0** (tanh(0)=0, starts off) | the model switches it on itself |
| XSA fused kernel | **SHIPPED** | real cost 8.9% -> 1.3%, 87% recovered |
| AttnRes (Kimi K3 attention residuals) | **PARKED** | nothing clears the seed floor |
| AttnRes default | b3s1 + per-dim + raw | chosen on **tps (1.5%)**, not quality |
| AttnRes carry scale | **flat direction** | c=1 everywhere costs <= 0.0033 |
| AttnRes per-dim carry | **flat too** | whole 512-dim vector replaceable by the scalar 1 |
| AttnRes routing via (1,H) pseudo-queries | **unproven** | the win was the LR, not the geometry |

## What we learned

**XSA specialises in both directions.** Mean rejection strength climbs to ~0.55, but the
interesting behaviour is the spread: some heads drive strongly *negative* (tanh(alpha) ~ -0.72),
the opposite of rejection. From checkpoints we localised every negative head in the all-MoE model
to **layer 1** -- which matches a phenomenon already on record here, and is why the off-simplex
embedding term exists (to test whether L1's negative alpha is a workaround for a missing
token-identity channel).

**Per-head alpha depends on FFN type, not attention type.** Measured tables for base1b and pdc1b,
with an L6 head-split and an L1 sign-flip.

**A microbenchmark mispredicted the real cost 13x, in both directions.** XSA's cost from a
microbench bore no relation to its cost in the model. Only end-to-end tps settles a kernel's price.

**AttnRes is parked, honestly.** Across 57 runs nothing cleared the noise floor. The default was
picked on a 1.5% throughput edge, and that should be stated whenever the default is quoted. Reviving
it means scaling **depth**, since the block structure is what it exploits.

**The carry's learned value says nothing about its importance** (Aug 16, two models):

| Layer | learned c | cost of removing it |
|---|---|---|
| L0 (all-MoE) | 0.781, near-lowest | **+0.4256**, the largest in the model |
| L0 (dense-0) | 0.843 | **+2.7384** -- destroys the model |
| L6 (all-MoE) | 0.502, the lowest | -0.0005, i.e. free |

Full write-up: the **Carry Paradox** report. Expert capacity at layer 0 cuts its carry dependence
6.4x, which is the first mechanism offered for why all-MoE wins.

**A fused two-stream res_add is broken.** carry + embedding in ONE fused call diverges from eager
by 1.8e-01 on the hidden state. One stream is exact. Embedding-term arms must run eager.

## Sources

Memory: `xsa-round`, `xsa-alpha-1b-values`, `attnres-carry-round`, `attnres-routing-round`,
`carry-is-flat`, `two-stream-res-add-broken`, `parity-vs-plumbing`.
