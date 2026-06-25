# BiBo Triton Kernels

Custom + Liger GPU kernels for BiBo, applied by **monkey-patching** (never by editing modeling
code). Every kernel is wrapped in a `torch.autograd.Function`, is **grad-exact vs PyTorch eager**
(atol 1e-3 fp16), and is benchmarked in **fp16** with `triton.testing.do_bench`.

> **Target hardware:** Tesla **T4** (sm_75, training — fp16 tensor cores, no bf16) and **RTX 3050**
> (sm_86, local bench). All numbers below are RTX 3050 fp16 unless noted; a few large-token shapes
> are marked *T4-cert pending* (the 4 GB card can't measure them).

## Design philosophy

1. **Fuse *between* GEMMs, not the GEMMs themselves.** cuBLAS already wins large aligned fp16 GEMMs;
   the leverage is killing the elementwise/normalize/scatter **HBM round-trips** around them.
2. **Account HBM traffic.** Any intermediate written to HBM and read back is a fusion target —
   collapse it into its producer's epilogue so it never leaves on-chip.
3. **Library = baseline to beat.** Liger is used where it already wins (RMSNorm, RoPE, SwiGLU);
   custom kernels are written only where a measured win exists (MoE dispatch, fused-CE, XSA).
4. **Grad-exact, NaN-free, profiled.** See "Correctness rules" below.

---

## Quick start — apply the whole stack

```python
from src.kernels import (
    patch_bibo_with_liger,         # Liger RMSNorm + RoPE (alias: patch_bibo_with_triton)
    patch_moe_auto,                # MoE experts (per-expert / grouped auto-dispatch)
    patch_dense_mlp_with_triton,   # dense MLP (Liger SwiGLU)
    patch_xsa_with_triton,         # XSA rejection
    patch_conv_router_with_triton, # conv router (only if router_type="conv")
)

model = BiBoForCausalLM(config).cuda()
patch_bibo_with_triton(model)
patch_moe_auto(model)
patch_dense_mlp_with_triton(model)
patch_xsa_with_triton()
model.config.use_fused_linear_ce = True   # fused-linear cross-entropy
# patch_conv_router_with_triton(model)     # only when config.router_type == "conv"
```

The bench suite does this in one call: `apply_triton_kernels(model, config, use_fused_ce=True)`
(`bench/models.py`) — and applies the fused CE to **both** BiBo and Qwen for a fair comparison
(Qwen routes through `patch_qwen3_fused_ce`). Pass `no_triton=True` for the eager baseline.

---

## Kernels

### 1. RMSNorm — Liger (`patch_bibo_with_triton`)
- **What:** every `BiBoRMSNorm` — the layer norms (hidden dim) and the attention QK-norms (head dim).
- **How it saves:** fused normalize with an fp32 upcast and in-place backward; the normalized output
  never round-trips through a separate intermediate.
- **Results (fp16):** layer-norm (dim 320) **7.8–8.5× fwd / 2.7× fwd+bwd**; QK-norm (dim 64)
  **2.3× fwd / 1.0–1.13× fwd+bwd** (break-even on the tiny dim-64 backward — a custom kernel there
  isn't worth it; see `docs/kernel_benchmark_report.md`).
- **Apply:** `patch_bibo_with_triton(model)`.

### 2. RoPE — Liger (`patch_bibo_with_triton`)
- **What:** rotary position embedding on Q/K.
- **How it saves:** fused rotation, eliminates the rotated-copy intermediate.
- **Apply:** `patch_bibo_with_triton(model)` (same call as RMSNorm).

### 3. MoE experts — custom (`patch_moe_auto`)
- **What:** the routed-expert FFN dispatch. Per-call auto-dispatch: a fixed **per-expert** path for
  small token counts, a **grouped-GEMM** path for `n_tokens ≥ 4096`.
- **How it saves:** three things — (a) the live dispatch wrapper used to compare CUDA *scalar*
  tensors per expert, forcing an implicit GPU→CPU **sync every expert**; pulling boundaries to CPU
  ints once removed it (the biggest small-scale win); (b) a **fused GLU activation** kernel
  (SiLU / ReLU² / Tanh dispatch) avoids writing the `gate_up` intermediate; (c) the grouped path
  avoids per-expert kernel-launch overhead at scale.
- **Results (fp16):** per-expert **1.42× fwd / 1.40× fwd+bwd** (was a 0.84× *regression* before the
  sync fix); grouped path **~2–2.5× fwd** at 4k–8k tok (*16k T4-cert pending*). Profiler: eager
  per-expert `.item()` syncs 53.5 ms → 9 ms at 1024 tok.
- **Apply:** `patch_moe_auto(model)`. (`patch_moe_with_triton` = per-expert only; `patch_moe_grouped`
  = grouped only.)

### 4. Dense MLP — Liger SwiGLU (`patch_dense_mlp_with_triton`)
- **What:** the dense MLP layers (layers 0 and N-1).
- **How it saves:** fused `SiLU(gate) * up` activation between the two cuBLAS GEMMs.
- **Results (fp16):** **1.19× fwd / 1.23× fwd+bwd**.
- **Apply:** `patch_dense_mlp_with_triton(model)` (Qwen: `patch_qwen_dense_mlp_with_triton`).

### 5. Conv router — custom (`patch_conv_router_with_triton`)
- **What:** the causal-conv1d router (only when `config.router_type == "conv"`).
- **How it saves:** reads hidden states natively in `(B, S, H)` and does a **transpose-free**
  backward — kills the permute round-trips the eager conv router pays.
- **Results (fp16):** full router **~2.5× fwd+bwd** at large batch; projection ~5× fwd.
- **Apply:** `patch_conv_router_with_triton(model)`.
- **⚠ Not this:** the conv **shared-expert** kernel (`patch_conv_expert_with_triton`) exists but is
  **intentionally not used** — 0.41× (slower than PyTorch ops; launch overhead dominates the cheap
  elementwise op).

### 6. Fused-linear cross-entropy — custom (`config.use_fused_linear_ce`)
- **What:** the final `lm_head` projection + cross-entropy, fused (cut-cross-entropy style).
- **How it saves:** **online-softmax forward never materializes the `(N, V)` logits** — it saves
  only the per-row `lse`. So activation memory for the loss is **flat (~0.5 GB) regardless of N**,
  while standard CE grows linearly and OOMs at the 16k-token step. Backward is chunked-cuBLAS with a
  fused in-place grad-logits kernel.
- **Results (fp16, H=320 V=81000):** forward **2.1×** at all N; fwd+bwd **1.05–1.26× at N≤2048,
  5.7× @4096, ~20× @8192**, and **enabling @16384** (standard CE OOMs needing ~10.6 GB; ours 547 ms
  / 489 MB). Peak memory **0.06–0.49×** standard. Grad-exact (fp16 gH ~2e-7).
- **Apply:** `model.config.use_fused_linear_ce = True` (BiBo); `patch_qwen3_fused_ce(model)` (Qwen).
  `apply_triton_kernels(..., use_fused_ce=True)` does both.

### 7. XSA (Exclusive Self Attention) — custom (`patch_xsa_with_triton`)
- **What:** the XSA rejection `z = y − (y·v̂)v̂` on the attention output (`docs/xsa.md`).
- **How it saves:** one fused forward + one fused backward kernel that broadcast V across the GQA
  group **in-kernel** (like SDPA `enable_gqa` — the `(B,H,S,D)` `repeat_kv` copy and the full-size
  normalized-V are never materialized), reducing in **fp32 in-register**. Backward uses the identity
  `grad_Y = reject(grad_z, v̂)` (the rejection operator is symmetric-idempotent); `grad_V` is
  analytic, accumulated over the group.
- **Results (fp16):** **2.2–2.6× fwd / 2.5–3.2× fwd+bwd**; peak memory **0.68–0.76×**. Grad-exact
  (fp32 ~2e-7; fp16 *more* accurate than fp16-eager since the eager reductions run in fp16). In the
  whole (SDPA-dominated) attention block it costs ~0.2–0.3 ms fwd+bwd and ~0 extra peak memory.
- **Apply:** `patch_xsa_with_triton()` (module-level; `unpatch_xsa()` to revert). The eager
  `apply_xsa(..., enable_gqa=True)` already drops the `repeat_kv` copy via broadcasting even without
  the kernel.

---

## End-to-end (full model)

| config | full step (≤2048 tok, RTX 3050) | step memory |
|--------|----------------------------------|-------------|
| baseline (eager) | 1.00× | 1.00× |
| Liger only | ~1.1× | — |
| **all kernels** | **~1.27–1.29×** | — |
| **all + fused-CE** | ~1.27–1.29× time | **0.66–0.74×** |

**Recommended training config: all kernels + fused-CE.** The kernels give ~1.3×; fused-CE flattens
step memory and is the **only path that runs the real 16k-token step** (standard CE OOMs). E2E above
2048 tok swaps on the 4 GB card → validate on the T4. Gradient checkpointing is **bit-exact** with
the full stack (`use_reentrant=True`) and trades ~1.2–1.4× recompute time for 0.34–0.65× memory.

---

## Correctness rules (enforced for every kernel)

1. **Gradient equivalence vs eager** (the *unpatched* PyTorch model, not the Liger-patched one):
   atol/rtol 1e-3 fp16, 1e-5 fp32. `src/kernels/bench/verify_grads.py`.
2. **NaN-free over ≥2 fwd+bwd passes.**
3. **Three-phase benchmark** — forward / backward / forward+backward, ≥3 warmup + ≥3 timed, report
   median ms + peak MB.
4. **`torch.profiler`** (CPU+CUDA activities) for the kernel breakdown, plus `triton.testing.do_bench`
   for wall-clock — never `time.time()`.

## Where the benches live

- `src/kernels/bench/` — `bench_moe.py`, `bench_dense_mlp.py`, `bench_conv.py`, `verify_grads.py`,
  `verify_e2e.py`, `profile_benchmark.py`.
- `src/kernels/.autoresearch/` — fused-CE, RMSNorm, **XSA** (`bench_xsa_compare.py`,
  `bench_xsa_attn.py`), grouped-MoE cert harness, E2E matrix scratch.
- `bench/e2e_ce_matrix.py` — full-model matrix, subprocess-isolated per cell.

Full measured tables + methodology: **`docs/kernel_benchmark_report.md`**.
