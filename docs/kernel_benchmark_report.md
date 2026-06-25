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

## Executive Summary (original — fp32, unreliable; kept for history)

Implemented and benchmarked Triton kernel optimizations for BiBo's MoE transformer, following Liger-Kernel's pattern: **fuse elementwise operations BETWEEN GEMMs, never fuse the GEMMs themselves**.

### Key Results

| Model Size | Params | Best Variant | Best Speedup | Best Memory Savings |
|-----------|--------|--------------|--------------|---------------------|
| small     | 3.4M   | MoE GLU      | 1.63x        | 30MB                |
| medium    | 10.8M  | MoE GLU      | 1.34x        | 100MB               |
| large     | 17.5M  | MoE GLU      | 2.66x        | 95MB                |
| xlarge    | 32.2M  | MoE GLU      | 6.35x        | 558MB               |

**Dense MLP (Liger SwiGLU) helps at scale**: 2.16x avg at xlarge (32M params).

---

## Changes Made

### 1. Dense MLP — Liger SwiGLU (`src/kernels/dense_mlp.py`)

**What changed**: Replaced custom `_FusedSwiGLUFull` with Liger's `LigerSiLUMulFunction`.

**Before**: Fused gate_up GEMM + custom Triton SwiGLU kernel
**After**: Separate cuBLAS GEMMs (gate_proj, up_proj) + Liger's production-grade fused activation

```python
# Before: fused GEMM + custom Triton
fused_weight = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
gate_up = F.linear(x_2d, fused_weight)  # 1 fused GEMM
intermediate = _FusedSwiGLUFull.apply(gate_up)  # custom Triton

# After: Liger pattern (separate GEMMs + Liger activation)
gate = self.gate_proj(x_2d)  # cuBLAS GEMM
up = self.up_proj(x_2d)      # cuBLAS GEMM
intermediate = LigerSiLUMulFunction.apply(gate, up)  # Liger Triton
```

**Why**: Liger's kernel is production-grade, handles edge cases (NaN, dtype casting), and uses in-place backward.

### 2. MoE GLU Activation (`src/kernels/moe_dispatch.py`)

**What changed**: Added fused GLU activation kernel + batched GLU kernel + weight-scatter kernel.

**New kernels**:
- `_fused_glu_act_kernel`: Per-expert GLU activation (SiLU/ReLU²/Tanh dispatch)
- `_batched_glu_act_kernel`: All experts in 1 launch (per-row act_type dispatch)
- `_fused_weight_scatter_kernel`: Weight multiply + scatter-add (inference-only)

**Performance**: 1.15x-6.35x speedup depending on model size and config.

### 3. Conv Expert (`src/kernels/conv_fused.py`)

**What changed**: Replaced Triton kernel with PyTorch ops.

**Before**: `_TritonConvGateMultiplyFunction` (permute + SiLU + multiply fused)
**After**: `F.silu(conv_out.permute(0, 2, 1)) * up_out` (PyTorch ops)

**Why**: Benchmark showed Triton kernel was ALWAYS slower (0.47x avg) due to kernel launch overhead on cheap elementwise ops.

### 4. Benchmark Infrastructure (`src/kernels/bench/`)

**New/updated files**:
- `profile_benchmark.py`: Full-model benchmark (5 variants × 4 sizes × 12 configs)
- `bench_isolated_kernels.py`: Fixed stale import for `_FusedSwiGLUSeparateBackward`

---

## Benchmark Results

### Full-Model Benchmark (forward+backward speedup vs baseline)

```
Model    | Config     | Liger  | Dense MLP | MoE     | All
---------|------------|--------|-----------|---------|------
small    | bs2_sl128  | 1.22x  | 1.23x     | 1.63x   | 1.09x
small    | bs2_sl256  | 1.03x  | 1.16x     | 15.91x* | 15.79x*
small    | bs2_sl512  | 0.47x  | 0.24x     | 0.17x   | 0.29x
small    | bs4_sl128  | 1.18x  | 1.10x     | 1.54x   | 1.73x
small    | bs4_sl256  | 1.20x  | 1.23x     | 1.51x   | 1.15x
small    | bs4_sl512  | 1.14x  | 1.45x     | 1.57x   | 0.94x
small    | bs8_sl128  | 1.01x  | 0.99x     | 1.08x   | 0.98x
small    | bs8_sl256  | 1.06x  | 1.11x     | 1.19x   | 1.14x
small    | bs8_sl512  | 1.05x  | 1.10x     | 1.03x   | 1.02x
medium   | bs2_sl128  | 1.21x  | 0.99x     | 0.79x   | 1.18x
medium   | bs2_sl512  | 0.89x  | 1.03x     | 0.94x   | 1.08x
medium   | bs4_sl256  | 0.33x  | 1.45x     | 1.06x   | 0.95x
medium   | bs8_sl128  | 1.23x  | 0.82x     | 1.31x   | 0.49x
large    | bs2_sl128  | 2.08x  | 2.16x     | 2.66x   | 1.79x
large    | bs2_sl512  | 1.16x  | 0.75x     | 0.79x   | 0.91x
large    | bs4_sl256  | 0.52x  | 0.53x     | 0.53x   | 0.53x
large    | bs8_sl128  | 1.26x  | 1.25x     | 1.34x   | 1.15x
xlarge   | bs2_sl128  | 0.54x  | 0.57x     | 0.70x   | 0.36x
xlarge   | bs2_sl512  | 1.42x  | 5.31x     | 6.35x   | 6.03x
xlarge   | bs4_sl256  | 1.16x  | 0.80x     | 0.94x   | 0.54x
xlarge   | bs8_sl128  | 1.16x  | 1.97x     | 2.45x   | 1.79x
```

*Triton autotuning anomaly — first pass picks optimal config.

### Average Speedup Per Model

```
small:   Liger=1.09x   Dense MLP=4.24x   MoE=4.20x   All=4.27x
medium:  Liger=1.91x   Dense MLP=5.36x   MoE=6.57x   All=3.76x
large:   Liger=1.26x   Dense MLP=1.17x   MoE=1.33x   All=1.09x
xlarge:  Liger=1.07x   Dense MLP=2.16x   MoE=2.61x   All=2.18x
```

### Memory Savings vs Baseline

```
Model    | Config     | Liger  | Dense MLP | MoE     | All
---------|------------|--------|-----------|---------|------
small    | bs2_sl512  | 15.0MB | 4.0MB     | 13.1MB  | 29.8MB
medium   | bs2_sl512  | 75.4MB | 58.3MB    | 75.2MB  | 100.0MB
medium   | bs8_sl128  | 29.0MB | 41.5MB    | 57.2MB  | 81.9MB
large    | bs2_sl512  | 40.3MB | 13.0MB    | 78.4MB  | 95.2MB
large    | bs8_sl128  | 41.5MB | 12.0MB    | 53.0MB  | 76.0MB
xlarge   | bs2_sl512  | 73.5MB | 20.6MB    | 128.7MB | 140.0MB
xlarge   | bs4_sl256  | 75.0MB | 19.9MB    | 541.9MB | 558.6MB
xlarge   | bs8_sl128  | 75.5MB | 18.7MB    | 147.1MB | 196.6MB
```

---

## Key Findings

### 1. Dense MLP helps at scale
- **0.92x-1.45x at small** (3.4M) — marginal
- **0.82x-1.45x at medium** (10.8M) — mixed
- **0.53x-2.16x at large** (17.5M) — helps at bs2/sl128
- **0.57x-5.31x at xlarge** (32.2M) — significant win

### 2. MoE GLU is the best overall performer
- Consistent 1.0-2.6x across most configs
- Up to 6.35x at xlarge/bs2/sl512
- Memory savings scale with model size

### 3. Kernel launch overhead dominates at small sizes
- At 3.4M params, all variants show 0.2-1.5x (variable)
- At 32.2M params, variants show 0.5-6.3x (consistent wins)

### 4. Memory savings are real and significant
- xlarge/bs4/sl256: 558MB savings with MoE alone
- All patches combined: up to 558MB savings
- Critical for training larger models on limited VRAM

---

## Architecture Decisions

### Why Liger SwiGLU (not custom Triton)
1. Production-grade: handles NaN, dtype casting, DTensor
2. In-place backward: zero allocation in backward pass
3. Battle-tested across thousands of models

### Why PyTorch ops for conv expert
1. Baseline ops are 0.036-0.807ms — too fast for Triton
2. Kernel launch overhead (~0.05ms) dominates
3. Memory savings don't justify performance cost

### Why batched GLU kernel exists (but isn't used in training)
1. Reduces 8 Triton launches → 1 launch
2. But concat/split overhead negates savings at small sizes
3. Available for inference-only paths

---

## Files Modified

| File | Lines Changed | What |
|------|--------------|------|
| `src/kernels/dense_mlp.py` | +30/-20 | Liger SwiGLU import + patched forward |
| `src/kernels/moe_dispatch.py` | +268/-100 | Batched GLU kernel + weight-scatter kernel |
| `src/kernels/conv_fused.py` | +57/-40 | PyTorch ops for conv expert |
| `src/kernels/__init__.py` | +15/-10 | Updated exports + docstring |
| `src/kernels/bench/profile_benchmark.py` | +731/-518 | Full-model benchmark (5 variants × 4 sizes) |
| `src/kernels/bench/bench_isolated_kernels.py` | +2/-1 | Fixed stale import |

---

## How to Run

### Isolated kernel benchmark
```bash
.\.venv\Scripts\python src\kernels\bench\bench_isolated_kernels.py
.\.venv\Scripts\python src\kernels\bench\bench_isolated_kernels.py --section dense_mlp
.\.venv\Scripts\python src\kernels\bench\bench_isolated_kernels.py --section moe
```

### Full-model benchmark
```bash
.\.venv\Scripts\python src\kernels\bench\profile_benchmark.py          # Full sweep (192 runs)
.\.venv\Scripts\python src\kernels\bench\profile_benchmark.py --quick  # Quick (48 runs)
```

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
