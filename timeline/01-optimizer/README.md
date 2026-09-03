# 01 - Optimizer

**Span** May - Aug 2026 &middot; `bibo-qwen-ablate-optimizer`, `bibo-manas-*`, `bibo-lr-sweep`,
plus the kernel work in `triton-kernel-fused`.

## Verdicts

| Axis | Verdict | Number |
|---|---|---|
| Muon vs AdamW | **SHIPPED** | Muon ~2x AdamW at matched steps |
| Muon weight decay | **SHIPPED**, dominant knob | 7x the effect of any coefficient change; optimum 2-4 |
| Muon LR | **SHIPPED** at 3e-3 | 4.3 sigma over the 1e-2 incumbent, plateau to 5e-3 |
| Newton-Schulz coefficients | **CLOSED** | ns6 == k2 at matched wd, at every scale |
| Symmetric-matmul NS | **SHIPPED** (sm120) | 1.4-1.5x, parity-clean |
| Gram-space NS, restart@3 | **SHIPPED** above symmul | 1.22x step win |
| Aurora / EMA variants | **CLOSED**, 4-way tie | after fixing an sm120 dispatch bug |
| SinkGD / LEO / sinkhorn-as-preconditioner | **CLOSED NEGATIVE**, refuted 3x | family closed |
| Manas | **CLOSED**, champion found | +0.0255 held-out vs Muon at g.08/rho.98/r8 |
| fp32 stabiliser in NS | **REFUTED** | |
| kappa at r=1 | **CLOSED**, loss-neutral | at 137M |
| Dither | **DEMOTED** | harmful on a real LM, dose-response 0.035-0.097 |

## What we learned

**Weight decay dominates the optimizer axis.** Once wd was matched, the entire Newton-Schulz
coefficient family collapsed to a tie -- `ns6` and `k2` are indistinguishable at every scale we
tried, including through grokking. Months of coefficient tuning were measuring an unmatched wd.
The lesson generalises: **before sweeping a subtle axis, check that the blunt one is matched.**

**The Manas result stands but its theory does not.** The champion (g=0.08, rho=0.98, r=8) beats
Muon by 0.0255 on held-out. The "Nexus" explanation was refuted; the mechanism that survived
scrutiny is long-memory momentum-free lookahead.

**A bench can fail its own reference.** The Muon parity FAIL at 6.7e-02 was `bench.py`'s reference
implementation being wrong -- its scale formula and cautious-decay handling -- not the kernel.
Corrected, the kernel matched at 9.5e-07. **When a parity test fails, suspect the test.**

**Read the launch script, not the argparse default.** The Muon LR sweep found 3e-3 optimal, and in
the process found that the `--muon_lr` argparse default of 3e-4 was dead code -- every run had been
overridden by the launcher.

## Open

- `ns8` free compression, pending a 121M A/B.
- Whether the Manas champion transfers to BiBo at 2k scale.

## Sources

Memory: `perf-per-flop-round`, `muon-symmul`, `muon-gram`, `aurora-ema`, `manas-round`,
`sinkgd-leo-research`, `kappa-pareto-round`, `r1-kappa-root-cause`, `muon-lr-sweep`,
`muon-parity-bench-trap`, `olm-seeds8`.
