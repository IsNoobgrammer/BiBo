# BiBo-JAX — Agent Onboarding

> System prompt for any AI agent working inside `src_jax/`.
> Read this AND the root `../AGENTS.md` (all root behavior rules still apply here).
> Read `PLAN.md` next for the build roadmap and extension-point map.

## What this dir is

The **JAX/Flax port of BiBo** for benching on **Kaggle TPU v5e-8** (8 cores, 128 GB HBM).
The rest of the repo (`src/`) is the PyTorch/CUDA model and is unaffected — this is a parallel
implementation, not a migration of `src/`. Both will coexist.

**Why it exists:** T4's 16 GB×2 caps the param count we can bench; v5e-8's 128 GB HBM lets us bench
bigger BiBo-vs-Qwen configs. Bigger models is the driver; MFU/throughput is a bonus.

## Stack

- **JAX + Flax NNX** (not Linen).
- **[EasyDeL](https://github.com/erfanzar/EasyDeL)** — we **EXTEND** it (subclass/compose), do NOT
  fork it and do NOT rebuild its trainer/sharding/optimizers. Pin a known-good version.
- **[ejkernel](https://github.com/erfanzar/ejkernel)** — EasyDeL's kernels (Triton GPU / Pallas TPU).
  Mirror its registry + `custom_vjp` pattern for our own kernels.
- **[eformer](https://github.com/erfanzar/eformer)** — optimizer factory (Muon lives here).

## The golden rule: reuse, don't rebuild

EasyDeL already gives us trainer, SPMD sharding (DP/FSDP/TP/EP/SP), gradient checkpointing,
HF-compat, RMSNorm, RoPE, and Muon. **Default to import-or-subclass. Write custom code ONLY where
BiBo's architecture is structurally incompatible** (see PLAN.md map). The custom surface is small:

- **OWN (write it):** PolyGLU MoE expert engine (heterogeneous activations + param-free experts),
  conv router, and the body-override bits of attention (XSA, NoPE head-split).
- **REUSE/SUBCLASS (everything else):** registration decorators, `UnifiedAttention`, RMSNorm, RoPE,
  Muon, the GMM kernel (`ejkernel.modules.grouped_matmul`).

If you're about to reimplement something EasyDeL provides, stop — find their version first.

## Environment

- **TPU** (v5e-8): Kaggle only. No local TPU.
- **Local (RTX 3050, sm_86)**: GPU verification only. `jax[cuda]`. ⚠️ Pallas's GPU path (Mosaic-GPU)
  is Hopper-leaning and the Triton backend is best-effort on Ampere — don't trust local Pallas for
  TPU-numerics parity; use it for logic/grad checks only.
- **Venv**: use a JAX-specific venv (do NOT pollute the PyTorch `../.venv`). Root rule #15 still holds
  — never the global interpreter.

## Kernel / correctness discipline (carries over from root)

1. **Grad correctness is non-negotiable.** Every custom kernel/module's gradient must match a
   reference. For a custom Pallas/Triton kernel: verify its hand-written VJP against `jax.grad` of a
   **pure-`jnp` reference** on CPU (backend-independent, runs locally). Then a bf16 grad-check on
   Kaggle TPU before trusting it.
2. **No autograd through Pallas** — backward is always hand-written (`jax.custom_vjp` `_fwd`/`_bwd`).
   Same discipline as the CUDA Triton kernels.
3. **Parity target = CUDA-BiBo.** The PyTorch model in `src/` is ground truth. Expect tolerance
   (bf16 TPU vs fp16 CUDA), not bit-equality.
4. **Both models on one stack.** Qwen baseline runs on EasyDeL too — never compare TPU-BiBo loss vs
   GPU-Qwen loss.

## Pointers

- `PLAN.md` — extension-point map + build order (read this).
- `../AGENTS.md` — root onboarding; the architecture (SSMax, XSA, PolyGLU, conv router, NoPE split,
  Muon recipe) is defined there. This dir re-expresses that architecture in JAX, it does not redesign it.
- EasyDeL ref for BiBo: their **Qwen3MoE** (`easydel/modules/qwen3_moe/`) — BiBo is architecturally
  closest to it; use it as the subclassing template.

## Status

Scoping. Nothing implemented beyond `PLAN.md` and this file.
