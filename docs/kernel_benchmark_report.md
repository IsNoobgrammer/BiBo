# BiBo Kernel Optimization — Benchmark Report

> Generated: June 2026 | Device: RTX 3050 Laptop (4GB, sm_86) | PyTorch 2.6.0+cu124 | Triton 3.7.0

## ⚠️ CORRECTION (June 24, 2026) — read this before trusting numbers below

The fp32/speedup tables in the rest of this doc were measured with a **broken harness**
and should not be trusted. Three bugs:
1. Timed in **fp32**, but the T4 training target runs **fp16** (Turing has no bf16).
   cuBLAS-vs-Triton tradeoffs invert between fp32 and fp16 (tensor cores).
2. **3-sample median** — far too few for a stable measurement.
3. A **profiler-wrapped iteration was included in the timed samples**, polluting them.
These produce the 0.17x / 15.91x / 6.35x anomalies — they are noise, not signal.

Re-measured in **fp16** with `triton.testing.do_bench` (L2 flush + warmup + median),
harness in `src/kernels/.autoresearch/`:

| Component | vs PyTorch eager (fp16) | Notes |
|-----------|-------------------------|-------|
| **MoE experts** | **1.42x fwd / 1.40x fwd+bwd** (7–8 of 9 cases) | was a **0.84x regression** before the fix |
| Dense MLP (Liger SwiGLU) | 1.19x fwd / 1.23x fwd+bwd (9/9 fwd) | already shipping; it *is* Liger |

**MoE fix (the real change):** the live dispatch wrapper compared CUDA *scalar* tensors
every expert iteration (`start = boundaries[i]; if end - start == 0`), forcing an implicit
GPU→CPU sync per expert that serialized the pipeline. Pulling boundaries to CPU ints once
(loop with Python ints) removed it — ~15 lines, no new kernel. Grad-equivalent to eager
(fp32 ~1e-7, fp16 in-tolerance, NaN-free).

**E2E (full model):** unmeasurable on the RTX 3050 — 50M-model step times bounce 3–4x
within a single run (thermal/clock throttle), and the model is CPU-launch-bound at this
scale. Validate E2E on the T4. See `src/kernels/.autoresearch/FINDINGS.md`.

---

## (June 25, 2026) Custom fused-linear cross-entropy + full-model E2E matrix

**Fused-linear-CE** (`src/kernels/fused_ce.py`) — cut-cross-entropy style: online-softmax forward
(never materializes the (N,V) logits, saves only `lse`) + chunked-cuBLAS backward with a fused
in-place grad-logits kernel. Replaces Liger's chunked CE (removed). Default in `apply_triton_kernels`
(`use_fused_ce=True`); also opt-in via `config.use_fused_linear_ce=True`. Measured vs standard
`F.cross_entropy` (H=320, V=81000, fp16, locked clock 1200 MHz):

| N (tok) | forward | fwd+bwd | peak mem (ours/std) |
|---------|---------|---------|----------------------|
| 1024  | 2.11x | 1.05–1.16x | 0.42x (465 vs 1118 MB) |
| 2048  | 2.16x | 1.06–1.26x | 0.22x (469 vs 2115 MB) |
| 4096  | 2.18x | 5.71x      | 0.11x (472 vs 4110 MB) |
| 8192  | 18.8x | 20.4x      | 0.06x (480 vs 8097 MB) |
| 16384 | std OOMs | std OOMs (ours 547 ms) | **enabling** (489 MB; std needs ~10.6 GB) |

- Memory is **flat ~0.5 GB regardless of N**; standard CE grows linearly → OOMs at the 16k step.
- Adaptive backward CHUNK (mem-budgeted, ~2485 rows) = **1.18–1.25x backward** (reconfirmed by
  repeated interleaved `do_bench`). Forward tile `(32,128,64)` is the reliable optimum — a sweep
  flagged `(64,128,64)` as 1.17x but 7 reconfirmation rounds showed it 0.94x (sweep noise).
- Grad-exact: fp16 gH ~2e-7, gW ~1e-7. Wired into **both BiBo and Qwen** (Qwen patch now uses our
  kernel, not Liger's).

**Full-model E2E matrix** (BiBo, subprocess-isolated per cell, fwd+bwd step ms; `bench/e2e_ce_matrix.py`).
Clean ≤2048-tok on the 3050 (≥4096 swaps on 4 GB → run on T4):

| tok | base/std | liger/std | all/std | all/fusedCE | all/std vs base |
|-----|----------|-----------|---------|-------------|------------------|
| 1024 | 254 | 232 | **197** | 253 | **1.29x** |
| 2048 | 394 | 349 | **310** | 374 | **1.27x** |

- **Kernels (`all` vs baseline) = ~1.27–1.29x** full-step speedup (Liger-only ~1.1x; custom MoE/conv
  adds the rest). The biggest small-scale win is killing the eager-MoE per-expert `.item()` syncs
  (profiler: 53.5 ms → 9 ms at 1024 tok).
- **Fused-CE is ~break-even on time at small N** (the loss is a small slice and its 3-GEMM backward
  recomputes logits) but cuts step memory to **0.66–0.74x**. The fused-CE *time* win and its
  OOM-enabling appear at ≥4096 tok (standard CE OOMs) — the regime the 4 GB card can't measure.
- **Recommended training config: `all + fused-CE`** — kernels give ~1.3x and fused-CE is what lets
  the real 16k step run at all.

---

## (June 25, 2026) Gradient checkpointing — exactness + time/memory tradeoff

Full BiBo model (all kernels + fused CE), `use_reentrant=True`, checkpointing ON vs OFF
(`bench/ckpt_compare.py`, subprocess-isolated, RTX 3050).

**Gradient equivalence: BIT-IDENTICAL.** `loss off=11.36270 on=11.36270` (Δloss 0), **max|Δgrad| =
0.00e+00** across 146 param tensors. Our kernels are `autograd.Function`s, so the per-layer recompute
participates normally; with router noise commented out + `attention_dropout=0` the forward is
deterministic → the recompute is bit-exact (not just within-tolerance). **Checkpointing is safe with
the full kernel stack.**

| tokens (B×S) | off ms | on ms | time × | off MB | on MB | mem × |
|---|---|---|---|---|---|---|
| 1024 (2×512)  | 253   | 350  | 1.38  | 1328 | 866  | 0.65 |
| 2048 (2×1024) | 376   | 462  | 1.23  | 2249 | 1218 | 0.54 |
| 4096 (2×2048) | 6100* | 1142 | swap* | 4776 | 1823 | 0.38 |
| 4096 (4×1024) | 1657* | 839  | swap* | 3935 | 1807 | 0.46 |
| 8192 (8×1024) | 24845*| 1612 | swap* | 6392 | 2160 | 0.34 |

- **Memory: real monotonic saving 0.65× → 0.34×** (more at larger seq/batch — more layer activations
  to drop). The genuine benefit.
- **Time: +23–38% at clean shapes** (those that fit without ckpt) = the recompute tax, one extra
  forward per layer. **The `*` ≥4096 rows where ckpt looks "faster" are a 4 GB swap artifact** —
  without ckpt those balloon to 4.8–6.4 GB → host-memory swap (6 s / 25 s steps). On a 16 GB T4 where
  the no-ckpt case fits, expect the ~1.2–1.4× tax everywhere. (`expandable_segments` is unsupported on
  this Windows build, sharpening the swap cliff.)
- **Use checkpointing when memory-bound** (large effective batch/seq); off when you're not.

---


## (June 25, 2026) Fused XSA (Exclusive Self Attention) kernel

Fused Triton kernel (`src/kernels/xsa_fused.py`, `patch_xsa_with_triton`) for the XSA rejection
`z = y − (y·v̂)v̂`. Collapses the eager path (repeat_kv + normalize + dot + scale + subtract = 7
launches over `(B,H,S,D)` tensors) into one forward + one backward kernel, broadcasting V across the
GQA group **in-kernel** (like SDPA `enable_gqa` — the `(B,H,S,D)` `repeat_kv` copy and full-size
normalized-V are never materialized), with fp32 in-register reductions. Grad-exact vs eager (fp32
~2e-7; fp16 *more* accurate than fp16-eager). Backward identity: `grad_Y = reject(grad_z, v̂)` (the
forward operator is symmetric-idempotent).

**Standalone module** (RTX 3050, fp16, `do_bench` median; vs the old eager `repeat_kv` path). Exact ms / peak MB:

| shape (BiBo H5 Hkv1 D64) | repeat_kv fwd | broadcast fwd | **fused fwd** | repeat_kv fwd+bwd | broadcast fwd+bwd | **fused fwd+bwd** |
|---|---|---|---|---|---|---|
| 1024 tok  | 0.109 ms | 0.090 ms | **0.042 ms** | 0.495 ms | 0.366 ms | **0.153 ms** |
| 4096 tok  | 0.334 ms | 0.246 ms | **0.142 ms** | 1.430 ms | 0.957 ms | **0.559 ms** |
| 16384 tok | 1.207 ms | 0.874 ms | **0.547 ms** | 5.180 ms | 3.266 ms | **2.074 ms** |
| big H32 Hkv8 (4096 tok) | 2.067 ms | 1.401 ms | **0.900 ms** | 8.286 ms | 5.266 ms | **3.365 ms** |

Peak MB, fwd+bwd: 1024 tok 5.3→**4.0**; 4096 tok 21.1→**16.0**; 16384 tok 84.3→**64.0**; big-H 152.5→**104.0**.
(eager-broadcast alone, no kernel, already removes the repeat_kv copy — the middle column.)

**Whole `BiBoAttention`** (Liger RMSNorm on, fp16; the XSA tax in context). Exact ms / peak MB, fwd+bwd:

| shape | no-XSA | eager-XSA | **fused-XSA** | no-XSA MB | fused-XSA MB |
|---|---|---|---|---|---|
| 2048 tok (B2 S1024) | 9.738 ms | 10.124 ms | **9.915 ms** | 190.9 | 191.1 |
| 4096 tok (B2 S2048) | 36.285 ms | 37.077 ms | **36.651 ms** | 683.3 | 683.8 |
| 4096 tok (B4 S1024) | 18.840 ms | 19.932 ms | **19.182 ms** | 363.0 | 363.5 |

Using fused XSA costs ~0.2–0.3 ms fwd+bwd on the whole attention block (vs eager XSA's ~0.4–1.1 ms)
and **~0 extra peak memory** (SDPA's quadratic transient sets the high-water mark). >4096 tok OOMs the
4 GB 3050 (quadratic SDPA) → measure on T4. The 1024-tok whole-attention row is below the 3050's
timing-noise floor and is omitted. Benches: `src/kernels/.autoresearch/bench_xsa_compare.py`,
`bench_xsa_attn.py`.

---

## Current kernel results (fp16, `triton.testing.do_bench`) — quick reference

| Kernel | vs eager (fp16) | Notes |
|--------|-----------------|-------|
| MoE experts (per-expert, `patch_moe_auto`) | 1.42× fwd / 1.40× fwd+bwd | sync-removal fix; grouped path ~2–2.5× fwd at 4k–8k |
| Dense MLP (Liger SwiGLU) | 1.19× fwd / 1.23× fwd+bwd | it *is* Liger |
| Conv **router** (`patch_conv_router_with_triton`) | ~2.5× fwd+bwd (large batch) | conv **expert** kernel intentionally NOT used (0.41× slower) |
| Fused-linear CE | 2.1× fwd; fwd+bwd 1.05–1.26× ≤2048, 5.7× @4096, enabling @16384 | mem flat ~0.5 GB; only path that runs the 16k step |
| RMSNorm (Liger) | dim320: 7.8–8.5× fwd / 2.7× fwd+bwd · QK dim64: 2.3× fwd / 1.0–1.13× fwd+bwd | QK-norm break-even on bwd; not worth a custom kernel |
| **XSA (fused, ours)** | 2.2–2.6× fwd / 2.5–3.2× fwd+bwd; mem 0.68–0.76× | see table above |
| E2E all-kernels + fused-CE | ~1.27–1.29× full step (≤2048 tok, 3050); step mem 0.66–0.74× | recommended training config |

**Reproduce:** kernel benches live in `src/kernels/bench/` (`bench_moe.py`, `bench_dense_mlp.py`,
`bench_conv.py`, `verify_grads.py`) and `src/kernels/.autoresearch/` (CE, RMSNorm, XSA, E2E matrix).
Always fp16 + `triton.testing.do_bench`. E2E matrix: `bench/e2e_ce_matrix.py` (subprocess-per-cell).

---

## Research References

- **Liger-Kernel** (linkedin/Liger-Kernel): Production-grade fused RMSNorm, RoPE, SwiGLU
- **TritonMoE** (arXiv:2605.23911): Block-scheduled grouped GEMM for MoE dispatch
- **FlashMoE** (arXiv:2506.04667): Single persistent kernel for distributed MoE
- **SonicMoE** (arXiv:2512.14080): IO-aware fused MoE on Blackwell/Hopper

---

## TODO / Future Work

1. **Persistent MoE dispatch kernel**: Eliminate Python for-loop entirely (15x paper claim)
2. **torch.compile on attention + norms**: Auto-fusion without manual kernels
3. **Larger GPU testing**: RTX 3090/4090 to test at 70B+ scale
4. **Fused QKV projection**: 1 GEMM instead of 3 for attention
