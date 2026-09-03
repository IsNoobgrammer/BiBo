# 07 - Kernels and speed

**Span** May - Aug 2026 &middot; `triton-kernel-fused` repo, measured through BiBo runs.

## Throughput, end to end

| Milestone | tps |
|---|---|
| 64-expert MoE, before the speed round | 128.9k |
| after the fused Triton GEMM+GLU | 148.9k |
| current board, all-MoE (Blackwell) | 170.9k |
| current board, dense end-layers | 179.6k |

## Verdicts

| Work | Verdict | Number |
|---|---|---|
| Fused Triton MoE GEMM+GLU | **SHIPPED** | 128.9k to 148.9k; 6 other approaches measured and rejected |
| MoE megakernel (fused norm + router + experts) | **SHIPPED**, default | supersedes the plain `moe` patch on MoE layers |
| XSA fused kernel | **SHIPPED** | 8.9% to 1.3% of step time |
| Fused CE, fp16 gw addmm_ + in-kernel 1/n | **SHIPPED** (T4) | 0.96x compiled at 3.75x less peak memory |
| Symmetric-matmul and gram-space Muon NS | **SHIPPED** (sm120) | 1.4-1.5x, then 1.22x on top |
| Two-stream fused res_add | **BROKEN, do not use** | diverges 1.8e-01 from eager |

## The rules that came out of it

**Bit identity is the ship gate.** Max absolute difference between kernel and eager must be exactly
zero, forward AND backward, on the model's dtype layout. A kernel that was *more accurate* than
eager cost real bpb, because the model had trained against eager's rounding. "Close enough" is a
different model.

**Parity passing does not mean the feature works.** XSA's parity test passed while its alpha was
dead. Assert that the feature changes an **end-to-end result**, not just that two tensors match.

**Never key an autotune config on a grid-size dimension.** Keying on sequence length caused a 4.1x
eval stall as every new length re-tuned. Outputs written with atomic_add need `reset_to_zero`.

**Microbenchmarks lie about real cost, in both directions** - XSA's was off by 13x each way.

**The stale-cache trap.** A temp script left over from an earlier call was silently re-run,
producing a table that looked plausible and described the wrong thing. Send code inline.

**Throughput is only measurable paired and interleaved.** One box drifted 20% in an afternoon: SW
power capping for 21 minutes (1182s on the counter, against a 1279s window), then full speed, then
slow again, with clocks, memory clocks, temperature, ECC and host load all clean and the startup
GEMM benchmark reading 420-434 TFLOPS throughout. The card was fine; the step was stalling. Two
throughput claims had to be retracted before `ablate/tools/tps_probe.sh` existed.

## Sources

Memory: `moe-kernel-speed-round`, `ce-memory-pareto`, `xsa-round`, `kernel-contract-bit-identity`,
`parity-vs-plumbing`, `triton-autotune-traps`, `two-stream-res-add-broken`, `box-tps-drift`,
`muon-symmul`, `muon-gram`, `kernels-layout`.
