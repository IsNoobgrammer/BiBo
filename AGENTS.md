# BiBo — Agent Onboarding

> This file is the system prompt for any AI agent working on this repo.
> Read this FIRST before doing anything.

## ⛔ No Commits Without Asking

**NEVER run `git commit` or `git push` unless the user explicitly says "commit" or "push".**

The user decides when the code is stable enough to commit. Work freely, stage files, but NEVER commit without being told to.

---

## What Is BiBo

BiBo is a **Mixture-of-Experts (MoE) Transformer** for causal language modeling. It's a research model — not a product. The goal is to explore diverse expert architectures and SSMax attention for long-context performance.

**Key differentiators from vanilla MoE (like Qwen3MoE):**
1. **SSMax** — learnable per-head query scaling (`scale * log(n)`, where `n` is each query's **causal context length** `(kv_len − q_len) + t + 1`, not a global `kv_len`) that prevents attention fading at long sequences
2. **Diverse experts** — PolyGLU layout: groups of 3 GLU experts with different activations (SiLU, ReLU², Tanh) + Identity/Zero special experts
3. **Shared Conv1D expert** — always-active causal convolution (gated, SwiGLU-style)
4. **Router logit normalization** — `router_lambda` scales normalized logits + threshold-based bias heuristics for load balancing
5. **Flash Attention (SDPA)** — uses `F.scaled_dot_product_attention` when `output_attentions=False`
6. **Conv router option** — `router_type="conv"` gives router local context awareness
7. **XSA (Exclusive Self Attention)** — parameter-free rejection of each token's attention output from its own value vector (`z = y − (y·v)v/‖v‖²`); applied after value-aggregation, before o_proj. See `docs/xsa.md`

---

## Project Structure

```
src/
├── configuration_bibo.py          # BiBoConfig (all hyperparams)
├── modeling_bibo.py               # Flat re-export for backward compat
└── modeling/
    ├── norm.py                    # BiBoRMSNorm
    ├── embed.py                   # BiBoRotaryEmbedding (Qwen3-compatible RoPE)
    ├── attn/
    │   ├── base.py                # BiBoAttention (SDPA + SSMax, fallback manual)
    │   ├── ssmax.py               # apply_ssmax_query_scaling
    │   ├── xsa.py                  # apply_xsa (Exclusive Self Attention rejection)
    │   └── utils.py               # repeat_kv
    ├── ffn/
    │   ├── mlp.py                 # BiBoMLP (SwiGLU)
    │   ├── experts.py             # Identity, ReLU², Zero, CausalConv1D
    │   ├── router.py              # BiBoMoERouter (MLP or Conv, logit norm)
    │   └── moe.py                 # BiBoMoELayer (routing + dispatch + bias update)
    ├── layers.py                  # BiBoDecoderLayer
    └── models.py                  # BiBoModel, BiBoForCausalLM

baseline/                          # Reference implementations for comparison
├── qwen3/                         # Qwen3 dense model
└── qwen3moe/                      # Qwen3MoE (our primary baseline)

docs/                              # Technical documentation
├── ssmax.md                       # SSMax paper notes + implementation details
├── moe_shared_scaling.md          # Monte Carlo scaling derivation
├── configuration_guide.md         # Full config parameter reference
└── deprecated.md                  # Removed components (NoiseExpert) + reasoning

kaggle_ablations/                  # Single-GPU ablation (gitignored, local only)

misc/kaggle/multi_gpu/             # 2×T4 parallel ablation
├── config.yaml                    # Configs (bibo→cuda:0, qwen→cuda:1)
├── data.py                        # Generate sorting task data (3 buckets: 64/128/256)
├── train.py                       # Parallel training (multiprocessing)
├── analyze_router.py              # Per-token router analysis + plots
├── metrics/                       # JSON metrics
├── plots/                         # Generated visualizations
└── report/                        # Next.js report (deployed via GitHub Pages)

research_on_activations/           # Activation function research (gitignored)
old_kernels/                        # Retired kernel variants (kept for re-benchmarking)
├── dense_mlp.py                   # Forward-only variant (Triton fwd, PyTorch bwd)
└── dense_mlp_fused.py             # Fully fused fwd+bwd single kernel

legacy/                            # Old monolithic code (DO NOT USE for new work)
```

---

## Environment

- **OS**: Windows (local dev), Linux (Kaggle)
- **Python**: Use `.\.venv\Scripts\python` ALWAYS (never system python)
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **GPU (local)**: RTX 3050 Laptop 4GB — compute capability 8.6 (supports Flash Attention)
- **GPU (Kaggle)**: 2×T4 16GB
- **Transformers**: v4.50+
- **Run commands from repo root**: `c:\Users\shaur\OneDrive\Documents\BiBo`

### Quick smoke test
```bash
.\.venv\Scripts\python -c "from src.modeling_bibo import BiBoForCausalLM; print('OK')"
```

---

## Architecture At A Glance

```
BiBoForCausalLM
├── BiBoModel
│   ├── Embedding (vocab → hidden)
│   ├── BiBoRotaryEmbedding (RoPE)
│   ├── BiBoDecoderLayer × N
│   │   ├── RMSNorm → BiBoAttention (GQA + SSMax + SDPA)
│   │   └── RMSNorm → BiBoMoELayer (or dense BiBoMLP for first/last layers)
│   │       ├── BiBoMoERouter (conv or mlp, logit norm)
│   │       ├── Routed: PolyGLU groups (SiLU + ReLU² + Tanh GLU) + Identity + Zero
│   │       └── Shared: 1 MLP-SwiGLU or CausalConv1D (always active, scaled by moe_shared_scaling)
│   └── Final RMSNorm
└── LM Head
```

**Attention**: SDPA (Flash Attention) by default. Falls back to manual matmul when `output_attentions=True`. GQA (fewer KV heads). QK-norm. SSMax query scaling. XSA rejection on the output (`use_xsa`).

**MoE**: First and last layers are dense MLP (layers 0 and N-1; `mlp_only_layers=[0, N-1]`). All remaining layers are MoE. Router uses logit normalization. Bias heuristics for load balancing. Router bias is `requires_grad=False` (not optimizer-managed, updated heuristically).

**Expert layout (PolyGLU)**: `polyglu_expert_multiplier` groups of 3 (SiLU-GLU, ReLU²-GLU, Tanh-GLU) + `special_expert_pairs` × (Identity, Zero). Default: 2×3 + 1×2 = 8 routed experts.

---

## Key Config Parameters

| Param | Default | What it does |
|-------|---------|-------------|
| `use_ssmax` | True | Enable SSMax query scaling |
| `use_xsa` | True | Exclusive Self Attention: reject attn output from its own value vector |
| `polyglu_expert_multiplier` | 2 | Groups of 3 GLU experts (SiLU, ReLU², Tanh) |
| `special_expert_pairs` | 1 | Pairs of (Identity, Zero) special experts |
| `num_experts_per_tok` | 6 | Top-K routing |
| `router_type` | "mlp" | Router architecture ("mlp" or "conv") |
| `router_lambda` | 1.0 | Logit norm scaling (higher = more decisive routing) |
| `router_noise` | 0.5 | Exploration noise during training |
| `bias_update_factor` | 0.01 | Load balancing step size |
| `bias_update_threshold` | 100K | Tokens between bias updates |
| `shared_expert_type` | "mlp" | Shared expert type: `"mlp"` (SwiGLU, like Qwen) or `"conv"` (CausalConv1D) |
| `moe_shared_scaling` | auto | Shared expert output scaling (auto-computed via Monte Carlo, accounts for router_lambda) |
| `mlp_only_layers` | [0, N-1] | Which layers use dense MLP instead of MoE (first + last) |
| `rope_nope_ratio` | 0.5 | Fraction of attention heads that are NoPE (no positional encoding). 0.5 = 2:2 (6 RoPE+NTK heads, 6 NoPE content heads at the 12h/2kv default). 0.0 = all RoPE (original). Must align with KV group boundaries. |

---

## Changes Made This Session (May 15-16, 2026)

### Code Changes

1. **Removed BiBoNoiseExpert** — no academic backing. Slot converted to extra MLP expert. See `docs/deprecated.md`.

2. **Fixed MC simulation for `moe_shared_scaling`** — now accounts for `router_lambda` (logit normalization). Also fixed numerically unstable softmax (`exp(x-max)`).

3. **Router bias `requires_grad=False`** — prevents FSDP conflicts. Bias is heuristically updated, not optimizer-managed.

4. **Config validation**: `num_routed_experts >= 4` (was >= 5 before noise expert removal).

5. **Attention uses SDPA (Flash Attention)** — `F.scaled_dot_product_attention` when `output_attentions=False`. Manual fallback for attention weight extraction.

6. **MoE dispatch**: Tried multiple approaches (batched BMM, Qwen-style fused gate_up, sorted dispatch). **Reverted to original sorted-dispatch loop** — fastest on single GPU with 8 experts. The loop is only a bottleneck at 64+ experts with EP across nodes.

### Research & Documentation

7. **Activation function research** — `research_on_activations/` (gitignored): docs on SwiGLU, ReLU², dReLU, xIELU, PolyCom, HeLU, PolyGLU, NormSiLU/DECO. Ablation study on MNIST comparing 7 activations.

8. **DECO paper analysis** (arXiv:2605.10933, ICML 2026) — key findings:
   - NormSiLU: dual-stage normalization before SiLU in experts
   - Non-gated MLPs better than SwiGLU with ReLU-based routing
   - Per-expert learnable scaling (not single scalar)
   - Shared expert should be 1-2× routed expert size (validates BiBo)

9. **Convolutions in LLMs research** — BiBo's CausalConv1D shared expert is novel (no prior work does conv as shared MoE expert).

10. **Engram research** (arXiv:2601.07372) — hash-based N-gram lookup. Different from Constant Expert (MoE++). Engram is a parallel memory system, not a routed expert.

### Scalability Analysis

11. **Identified P0 issues for distributed training:**
    - Sequential expert loop (code issue, not architectural)
    - Unsynchronized bias buffers across DDP ranks
    - CausalConv1D incompatible with sequence parallelism (needs halo exchange)
    - SSMax uses local kv_len (needs global seq_len under SP)
    - MC simulation non-deterministic (needs seed)

12. **Router noise across ranks is NOT an issue** — router is replicated, only experts are parallelized in EP.

### Ablation Results (Kaggle)

13. **BiBo vs Qwen3MoE on sorting task** — BiBo converges faster and reaches lower final loss (~0.1 vs ~0.25) despite fewer params (8.3M vs 11.2M).

14. **Router confidence insight**: BiBo's logit normalization gives moderate confidence (0.4-0.7) across top-k experts = better expert utilization. Qwen's raw softmax gives 0.9+ top-1 = wastes the other k-1 experts. High top-1 confidence is BAD when top_k > 1.

---

## Important Design Decisions

1. **No sliding window / recurrent attention** — removed. Only standard softmax + SSMax. **If SWA is ever added: do NOT use SSMax on the windowed layers** — a fixed window caps `n`, so SSMax degenerates to a redundant constant temperature there. Keep SSMax only on full/global layers (set windowed layers' `s=0`). See `docs/ssmax.md` § "Do NOT use SSMax on sliding-window-attention layers".
2. **SSMax init**: `1.0 / log(max_pos_emb / 2)` — ensures attention starts ~neutral, not 6× sharper than standard.
3. **Shared expert is NOT routed** — it's always active. Only the Conv1D is shared.
4. **`output_attentions=True`** works (falls back to manual attention).
5. **Router bias is non-trainable** — `requires_grad=False`, updated via heuristic `.add_()`.
6. **Noise expert was removed** — no evidence it helps. Identity covers the "dump bucket" use case. See `docs/deprecated.md`.
7. **Conv router** — gives router local context (sees previous `kernel_size-1` tokens). Novel — no other MoE paper uses convolutional routing.
8. **Logit norm prevents expert waste** — when top_k > 1, normalization ensures all selected experts contribute meaningfully (not just top-1 dominating).

### Kernel Design Decisions (June 2026)

9. **Triton kernels have custom backward** — all MoE Triton kernels use custom Triton backward kernels, not PyTorch fallback. Gold standard: `_FusedSwiGLUFull` in dense_mlp.py (matched forward+backward Triton kernels with autotune).

10. **GQA via repeat_kv, NOT `enable_gqa` (reversed Jun 25 2026)** — the SDPA attention call uses
    `repeat_kv` to full MHA and does **NOT** pass `enable_gqa=True`. Measured: `enable_gqa=True`
    silently forces SDPA onto the **MATH backend** (materializes the O(n²) scores → ~10–25× slower,
    O(n²) memory) because the mem-efficient/flash backends don't accept the GQA broadcast on our
    torch/HW. **T4 (sm_75) has no flash at all → mem-efficient is the production path.** `repeat_kv`
    (cost O(B·H·S·D), not O(n²)) lets the dispatcher pick mem-efficient (T4) / flash (Ampere+).
    Verified bit-equivalent (max|Δ| 3.7e-4 fp16) + efficient backend confirmed (no n² materialization).
    Bench: `src/.autoresearch/bench_topk_attn.py`. **XSA keeps its own `enable_gqa` broadcast (in-module,
    not SDPA) — that path is unaffected.**

11. **No autotuning for MoE kernels** — M varies per expert per step, causing recompilation overhead. Use fixed block sizes with `triton.next_power_of_2()` clamped to ranges. Dense MLP, attention, conv kernels CAN use autotuning.

12. **Float32 in backward** — all backward computations use float32 intermediates to prevent overflow in GLU derivatives (`sig * (1 + g * (1-sig))` can overflow float16).

13. **Scatter → Gather pattern** — Forward uses `atomic_add` for scatter (non-differentiable). Backward uses `tl.load` with gathered indices (differentiable). This is the standard pattern for non-overlapping scatter-add.

14. **Recompute vs save for backward** — For small M (MoE experts), saving gate_up is preferred (low memory cost). For large M, recomputation saves memory. Decision: save for MoE, recompute for dense.

---

## Common Tasks

### Run a forward pass
```python
import torch
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

cfg = BiBoConfig(vocab_size=5000, hidden_size=512, num_hidden_layers=4,
                 num_attention_heads=8, num_key_value_heads=2,
                 polyglu_expert_multiplier=2, special_expert_pairs=1,
                 num_experts_per_tok=2,
                 moe_intermediate_size=256, intermediate_size=1024,
                 moe_shared_scaling=2.0)
model = BiBoForCausalLM(cfg)
x = torch.randint(0, 5000, (2, 128))
out = model(x, labels=x)
print(out.loss)
```

### Run Kaggle ablation (2×T4)
```python
!git clone https://github.com/IsNoobgrammer/BiBo.git
%cd BiBo
!pip install -qU transformers einops wandb bitsandbytes pyyaml seaborn
!python misc/kaggle/multi_gpu/data.py
!python misc/kaggle/multi_gpu/train.py
!python misc/kaggle/multi_gpu/analyze_router.py
```

### Compare with Qwen3MoE baseline
```python
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
```

---

## Known Quirks / TODOs

- `router_temperature` param exists in config but is never used (legacy, kept for compat)
- `moe_shared_scaling=1.0` triggers a 10K-iteration Monte Carlo in config init — pass explicit value to skip
- The `legacy/` folder has the old monolithic code — don't touch it, it's reference only
- No tests currently exist — they were removed during cleanup. New tests needed.
- Qwen baseline requires real `transformers` package (not stubs)
- CausalConv1D needs halo exchange for sequence parallelism (future work)
- MoE dispatch loop is fine for ≤16 experts on single GPU; needs grouped GEMM for 64+ experts with EP
- All MoE Triton kernels now have custom Triton backward kernels (not PyTorch fallback)
- GQA uses `repeat_kv` + plain SDPA (NOT `enable_gqa` — that hit the slow MATH backend; see design decision #10, reversed Jun 25 2026)
- Conv kernels (conv_fused.py) intentionally not used — 0.41x slower due to kernel launch overhead on small tensors
- **(June 24 2026) Benchmark in fp16, not fp32** — T4 (training GPU) is fp16-only. The old fp32 numbers in `docs/kernel_benchmark_report.md` are unreliable (see correction header there). Use `triton.testing.do_bench`, never the hand-rolled 3-sample timer.
- **(June 24 2026) MoE dispatch sync fix** — `triton_moe_experts_forward` now builds expert boundaries as CPU ints (was comparing CUDA scalar tensors per expert → implicit GPU sync per iter). Took the per-expert path from 0.84x regression → 1.40x vs eager (fp16).
- **(June 24 2026) Grouped-GEMM MoE** (`moe_grouped.py`, opt-in via `patch_moe_grouped`) — forward ~2–2.5x, fwd+bwd ~2x at 4k–8k tokens vs eager (fp16). **Certify the 16384-tok training shape on T4** — the 4GB RTX 3050 can't measure it (thermal + allocator pressure). Cert harness: `src/kernels/.autoresearch/bench_real.py`. Stays opt-in until T4-verified; per-expert path remains default.
- **(June 24 2026) CE default = standard non-chunked fused CE** — `models.py` now defaults to
  `logits = lm_head(x)` + `F.cross_entropy` (aten-fused, non-chunked). Liger's chunked CE was REMOVED.
- **(June 25 2026) Opt-in CE = BiBo's OWN fused-linear-CE Triton kernel** (`src/kernels/fused_ce.py`),
  via `config.use_fused_linear_ce=True`. Cut-cross-entropy style: online-softmax forward (never
  materializes (N,V) logits, saves only lse) + chunked-cuBLAS backward.
  ⚠️ **(June 26 2026) CORRECTION — the "fwd 2.1× / fwd+bwd faster" claims were measured vs EAGER
  `F.cross_entropy` (torch.compile was broken locally, so the real baseline was never tested).
  Against the REAL pipeline (torch.compile'd standard CE = cuBLAS GEMM + inductor-fused log_softmax),
  the fused kernel is ~2.36× SLOWER:** at H512/V81920/B16 on T4, compiled-std-CE = 688ms / 14.7% MFU
  vs fused-CE = 1625ms / 6.2% MFU; the `_flce_fwd_kernel` alone is 59% of the step (~951ms). Root
  cause: the tl.dot streaming forward GEMM runs at ~2% SoL vs cuBLAS ~50% (matches the prior
  "Triton GEMM can't beat cuBLAS" finding). **FIX (June 26 2026): `fused_linear_cross_entropy` now
  dispatches to a cuBLAS-CHUNKED implementation** (`_CECublasChunked` in `fused_ce.py`) — forward
  GEMM via cuBLAS in row-chunks, never materializes (N,V), keeps only lse(N,). Backward unchanged
  (already cuBLAS-chunked). Grad-exact vs `F.cross_entropy` (loss Δ3.8e-6, grad Δ~4e-9). Goal:
  cuBLAS speed (≈ compiled std CE) at bounded memory. The old tl.dot kernel is preserved as
  `fused_linear_cross_entropy_tldot`. Default `use_fused_ce: true` (= cuBLAS-chunked, NOT tl.dot).
  **✅ T4-VALIDATED (June 26 2026, H512/V81920/B16, --compile):** cuBLAS-chunked **727ms / 13.9% MFU
  / 13.0 GB** vs compiled-std-CE **725ms / 13.9% / 14.5 GB** vs old tl.dot **1625ms / 6.2% / 8.7 GB**.
  Ties std CE on speed (both cuBLAS), saves 1.5 GB (only (chunk,V) transients, never full (N,V)) →
  the one CE path at B16. Memory knob = `_BWD_LOGITS_BUDGET` (1 GiB → chunk ~6710; lower for less
  peak at a few more launches). tl.dot path is dead, reference only.
- **(June 25 2026) CE wired into model patching** — `apply_triton_kernels(model, config,
  use_fused_ce=True)` now enables the fused CE by default: BiBo sets `config.use_fused_linear_ce`;
  `patch_qwen3_fused_ce` swaps Qwen's loss to OUR kernel (was Liger's chunked CE). Pass
  `use_fused_ce=False` for the standard-CE baseline. E2E matrix bench: `bench/e2e_ce_matrix.py`.
  Verified: BiBo loss/gradnorm match std CE under autocast; Qwen routes through our kernel.
  ⚠️ **SUPERSEDED (June 26 2026): both models now default to compiled STANDARD CE (`use_fused_ce:
  false`)** — see the correction above; fused-CE is 2.36× slower than compiled std CE when logits fit.
- **(June 25 2026) fused-linear-CE measured results** ⚠️ **MEASURED VS EAGER CE — NOT THE REAL
  BASELINE. SUPERSEDED, see June 26 correction above.** (vs *eager* `F.cross_entropy`, H=320 V=81000
  fp16): forward 2.1–2.2×, fwd+bwd 1.05–1.26× at N≤2048, etc. These numbers stand ONLY against
  un-compiled eager CE; against torch.compile'd standard CE (the real pipeline) the kernel loses
  ~2.36× because its tl.dot forward is ~2% SoL vs cuBLAS ~50%. Kept for history; do not act on them.
- **(June 25 2026) full-model E2E matrix** (BiBo, `bench/e2e_ce_matrix.py`, subprocess-isolated per
  cell — single-process runs accumulate allocator fragmentation that poisons later cells). Clean
  ≤2048-tok on the 3050 (≥4096 swaps on 4 GB → run on T4): all-kernels = **~1.27–1.29× full-step**
  over baseline (Liger-only ~1.1×); biggest small-scale win is removing eager-MoE per-expert
  `.item()` syncs (53.5 ms→9 ms at 1024 tok). Fused-CE is ~break-even on time at small N (loss is a
  small slice; its 3-GEMM backward recomputes logits) but cuts step memory to **0.66–0.74×**; its
  time win + OOM-enabling are at ≥4096 tok. **Recommended training config: all-kernels + fused-CE.**
- **(June 25 2026) BiBo-vs-Qwen bench framing: LOSS is the primary metric** (MFU/tps are secondary
  supporting evidence). Qwen is **deliberately handicapped larger** so a BiBo loss win is conservative:
  experts kept equal at 8 each (but 2 of BiBo's 8 are Identity/Zero specials → Qwen's 8 full GLU
  experts are bigger by construction: 6.88M vs 4.42M routed/layer), and Qwen widened
  (`moe_intermediate_size` 768→896, `intermediate_size` 1024→1280) to **85.9M total / 44.6M active**
  vs BiBo **71.7M / 39.2M** (+20% total, +14% active, active capped ≤45M). Clean conclusion only in
  the BiBo-wins direction (a Qwen win is size-confounded); the larger Qwen must get enough steps to
  converge or it's just undertrained. torch.compile is used **on T4 only** (broken locally) — verify
  both runs log "torch.compile OK". Both models get the fused CE.
- **(June 25 2026) Gradient checkpointing verified safe with the full kernel stack** — ckpt ON vs OFF
  (all kernels + fused CE, `use_reentrant=True`): gradients **BIT-IDENTICAL** (max|Δgrad| 0.00e+00,
  Δloss 0, 146 tensors). Kernels are autograd.Functions so the layer recompute participates normally;
  deterministic forward (router noise commented out, dropout=0) makes recompute bit-exact. Tradeoff
  (`bench/ckpt_compare.py`): memory **0.65×→0.34×** (more saving at larger seq/batch), time **+23–38%**
  recompute tax at shapes that fit without ckpt (≥4096-tok "speedups" on the 3050 are 4 GB swap
  artifacts; expect the ~1.2–1.4× tax everywhere on T4). Router noise injection in `router.py` is
  **commented out (DEPRECATED, do not remove)** — we run `router_noise=0`; forward-time randomness
  would break checkpointing without RNG preservation.
- **(June 24 2026) Conv router fused** (`conv_fused.py`, `patch_conv_router_with_triton`) — real
  Triton kernel now (was fake PyTorch): native-(B,S,H)-read forward + transpose-free Triton backward.
  Full router fwd+bwd ~2.5x vs eager at large batch; projection ~5x fwd. fp16 grad-correct.
- **(June 24 2026) `patch_moe_auto` (recommended MoE patch)** — per-call dispatch: grouped path for `n_tokens >= GROUPED_MIN_TOKENS` (default 4096), else the fixed per-expert path. Beats PyTorch eager across all token regimes by construction (both branches grad-verified). Tune `GROUPED_MIN_TOKENS` after T4 cert.
- **(June 26 2026) Partial RoPE head-split (`rope_nope_ratio=0.5` = 2:2, DEFAULT)** — first `round(num_heads*(1-ratio))` query heads + corresponding KV heads get full RoPE+NTK; the remaining heads are NoPE (no positional encoding). At the **12h/2kv default**: 6 RoPE+NTK heads (1 RoPE KV group) + 6 NoPE content heads (1 NoPE KV group) — clean KV-group split, no GQA geometry change needed (group size 6; 6%6=0). Empirical (synthetic passkey + MQAR length-gen, train@128, eval to 32×, 3 seeds): more NoPE monotonically improves extrapolation — pure NoPE hits XG=1.000 on both tasks vs full-RoPE+NTK 0.93 (passkey) / 0.33 (MQAR); 2:2 gets 0.99 (passkey) / 0.50 (MQAR). **2:2 chosen as the default** (not higher NoPE): our retrieval evals treat position as *noise* (RoPE's worst case), so they overstate NoPE's value for a general LM where position is signal — 2:2 keeps half the heads positional, and is the only partial ratio that's KV-aligned at 12h/2kv without changing GQA. Mechanism: RoPE position-encodes key tokens by *where* they're planted → breaks key-matching; NoPE heads match position-independently; SSMax sharpens; causal mask gives implicit scale-free position. **The boundary must align with KV groups** (config validation enforced; only {0.0, 0.5} valid at 12h/2kv). `rope_nope_ratio=0.0` restores original all-RoPE. **IID/downstream cost of NoPE is UNMEASURED** (our synthetic evals saturate to 1.00 at train length) — Kaggle ablation on real-LM perplexity needed before pushing the ratio higher (would require changing GQA geometry to unlock 0.25/0.75). See `src/.autoresearch/FINDINGS.md` (2026-06-26 V10 + MQAR).

---

## Triton Kernels (`src/kernels/`)

**All custom GPU kernels MUST be written in `src/kernels/`. Read the `tritonify` skill before implementing any kernel.**

### Architecture

```
src/kernels/
├── __init__.py              # Exports all patching functions
├── patch.py                 # Liger-Kernel patches (RMSNorm, RoPE)
├── moe_dispatch.py          # Triton MoE kernels (fused GLU activation, router)
├── dense_mlp.py             # Triton fused SwiGLU for dense MLP layers
├── conv_fused.py            # Triton fused conv permute + activation + gate
└── bench/                   # All kernel benchmarks
    ├── __init__.py
    ├── bench_utils.py       # Shared: benchmark_phase, gradient check, NaN check
    ├── profile_benchmark.py # torch.profiler 4-way benchmark (Baseline/Liger/Triton/Triton+AT)
    ├── verify_grads.py      # Gradient equivalence verification (6 tests)
    ├── bench_moe.py         # MoE benchmark (correctness + 3-phase perf)
    ├── bench_dense_mlp.py   # Dense MLP 3-variant head-to-head
    ├── bench_conv.py        # Conv fusion benchmark
    ├── bench_moe_fwdbwd.py  # MoE full fwd+bwd training step benchmark
    └── verify_e2e.py        # Full model E2E verification

old_kernels/                  # Retired variants (kept for re-benchmarking)
├── dense_mlp.py             # Forward-only (Triton fwd, PyTorch bwd)
└── dense_mlp_fused.py       # Fully fused fwd+bwd single kernel
```

### Kernel Inventory (June 2026)

| Kernel | File | Forward | Backward | Status |
|--------|------|---------|----------|--------|
| `_fused_glu_act_kernel` | moe_dispatch.py | Triton | **Triton** (`_TritonMoEGLUFunction`) | Production |
| `_fused_linear_glu_kernel` | moe_dispatch.py | Triton | **Triton** (recomputes gate_up) | Production |
| `_fused_down_weight_kernel` | moe_dispatch.py | Triton | **Triton** (integrated) | Production |
| `_fused_weight_scatter_kernel` | moe_dispatch.py | Triton | **Triton** (scatter→gather) | Production |
| `_fused_router_kernel` | moe_dispatch.py | Triton | N/A (detached) | Production |
| `_batched_glu_act_kernel` | moe_dispatch.py | Triton | **Triton** (`_TritonFusedGLUFunction`) | Production |
| `_grouped_mm_kernel` | moe_grouped.py | Triton | **Triton** (`_GroupedMoE`, transposed via strides) | Opt-in (large-seq) |
| `_grouped_wgrad_kernel` | moe_grouped.py | N/A | **Triton** (per-expert weight grads) | Opt-in (large-seq) |
| `_fused_swiglu_*_kernel` | dense_mlp.py | Triton | **Triton** (`_FusedSwiGLUFull`) | Gold standard |
| `_fused_permute_act_gate_kernel` | conv_fused.py | Triton | **Triton** (`_TritonConvGateMultiplyFunction`) | Production |
| `_fused_permute_act_kernel` | conv_fused.py | Triton | **Triton** (`_TritonPermuteActFunction`) | Production |
| Liger RMSNorm | patch.py | Triton | Triton (Liger) | Production |
| Liger RoPE | patch.py | Triton | Triton (Liger) | Production |
| Liger SiLUMul | patch.py | Triton | Triton (Liger) | Production |
| Liger Fused CE | models.py | Triton | Triton (Liger) | Production |

### Backward Kernel Pattern (from dense_mlp.py gold standard)

```python
class _FusedXxxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, act_type):
        ctx.save_for_backward(x, weight)
        ctx.act_type = act_type
        # Launch Triton forward kernel
        out = torch.empty(...)
        _fused_xxx_forward_kernel[grid](x, weight, out, ...)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # Launch Triton backward kernel for activation derivatives
        grad_gate_up = torch.empty(...)
        _fused_xxx_backward_kernel[grid](grad_output, x, weight, grad_gate_up, ...)
        # GEMM backward via cuBLAS (optimal for large shapes)
        grad_x = F.linear(grad_gate_up, weight)
        grad_weight = torch.mm(grad_gate_up.t(), x.float())
        return grad_x, grad_weight, None
```

### Backward Kernel Reference (from tritonify)

**GLU activation backward** (from glu-kernels.md):
- SiLU: `dact = sig * (1 + g * (1 - sig))`
- ReLU²: `dact = 2 * relu(g)`
- Tanh: `dact = 1 - tanh(g)²`
- All computed in float32 to prevent overflow

**Weight scatter backward** (from moe-kernels.md):
- Forward: `out[idx[n]] += expert_out[n] * weights[n]` (atomic_add)
- Backward: `grad_expert_out[n] = grad_out[idx[n]] * weights[n]` (gather)
- Backward: `grad_weights[n] = (grad_out[idx[n]] * expert_out[n]).sum()` (reduce)

**GEMM backward** (from llm-optimizations.md):
- `grad_x = grad_output @ weight` (cuBLAS)
- `grad_weight = grad_output.T @ x` (cuBLAS)
- Always use float32 for the GEMM to prevent overflow

### 4 Mandatory Rules for All Triton Kernels

Every kernel implementation and benchmark MUST satisfy ALL 4 rules:

**Rule 1: Gradient Equivalence**
> For every Triton implementation, the gradient must match the **baseline (original PyTorch)** — NOT the Triton-patched version.

- Baseline = PyTorch eager with no patches applied
- Tolerance: atol=1e-3, rtol=1e-3 (fp16); atol=1e-5 (fp32)
- Verified by: `verify_grads.py` + per-bench gradient checks
- If a kernel passes forward correctness but fails gradient equivalence, it is BROKEN for training

**Rule 2: NaN-Free Multi-Pass Stability**
> Every kernel must complete ≥2 full forward+backward passes with zero NaN losses.

- Run 2+ forward+backward cycles on the same input
- Check `loss.isnan().any() == False` after every pass
- Check `all(p.grad is not None and not p.grad.isnan().any() for p in model.parameters())`
- If any NaN → kernel is broken, do not proceed to benchmarking

**Rule 3: Three-Phase Benchmark (Fwd / Bwd / Fwd+Bwd)**
> Every kernel must be benchmarked separately for forward, backward, and forward+backward.

- **Forward-only**: `model(x)` with `torch.no_grad()` — measures inference speed
- **Backward-only**: Run forward once, then `loss.backward()` — measures gradient computation
- **Forward+Backward**: Full training step — measures total training cost
- Each phase runs **≥3 warmup steps + ≥3 timed steps** before averaging
- Report: median time (ms), speedup vs baseline, peak memory (MB)

**Rule 4: torch.profiler for All Benchmarks**
> All benchmarks MUST use `torch.profiler` for timing and kernel breakdowns.

- Use `torch.profiler` with `ProfilerActivity.CPU, ProfilerActivity.CUDA`
- Use `prof.key_averages().table(sort_by="cuda_time_total")` for kernel breakdown
- Use `torch.cuda.Event(enable_timing=True)` for wall-clock timing
- NEVER use `time.time()` for GPU benchmarking

### GEMM Policy: When to Use Triton vs cuBLAS

**Default: cuBLAS** for standard large GEMMs (M ≥ 128, K, N ≥ 512).

**Use Triton GEMMs when fusion eliminates HBM round-trips:**

| Case | Why Triton wins | BiBo target | Expected gain |
|------|----------------|-------------|---------------|
| GEMM + SwiGLU activation | Fuses gate_up GEMM + silu(gate)*up into 1 kernel. Eliminates (M,2I) write + read. | Dense MLP layers | 1.2-1.5x on activation |
| Small-M expert GEMM | cuBLAS overhead dominates at M<32. Memory-bound, not compute-bound. | MoE expert dispatch | 1.3-1.5x on MoE fwd |
| GEMM + bias + activation | Eliminates 2 HBM round-trips. | Conv expert gate | 1.3-1.4x |

**Do NOT write Triton GEMMs for:**
- Large standard shapes (M ≥ 128) without fusion — cuBLAS tensor cores win
- Router matmul (small N) — cuBLAS heuristic handles this efficiently

**Sources:** Liger-Kernel paper (arXiv:2410.10989), TritonMoE paper (arXiv:2605.23911),
Triton matmul tutorial, CSDN Triton matmul optimization, GitHub triton-fp8-matmul

**Key insight from research:** The win from "Triton GEMM" is almost always **fusion** —
combining the GEMM with a subsequent operation into a single kernel launch. A standalone
Triton GEMM that replaces cuBLAS without fusion rarely wins.

### Kernel Development Rules

1. **Liger-Kernel first** — check if Liger already provides the op before writing custom
2. **Monkey-patch pattern** — kernels applied via `patch_*` functions, never modify modeling code
3. **autograd.Function mandatory** — wrap every raw Triton kernel in `torch.autograd.Function`
4. **JIT + Autotune toggle** — every kernel has `_kernel` (fixed) and `_kernel_at` (autotuned), controlled by `USE_AUTOTUNE` flag
5. **NEVER use `register_buffer` for tensors that need gradients** — buffers are excluded from autograd
6. **Read the tritonify skill** — before implementing any kernel, load the `tritonify` skill and read its references
7. **Correctness is binary** — atol=1e-3 for fp16, atol=1e-5 for fp32
8. **Run `verify_grads.py` before promoting any kernel** — gradient equivalence check

### Running Benchmarks

```bash
.\\.venv\\Scripts\\python src/kernels/bench/bench_dense_mlp.py   # Dense MLP (3-variant head-to-head)
.\\.venv\\Scripts\\python src/kernels/bench/bench_moe.py         # MoE layer
.\\.venv\\Scripts\\python src/kernels/bench/bench_conv.py        # Conv fusion
.\\.venv\\Scripts\\python src/kernels/bench/bench_moe_fwdbwd.py  # Full fwd+bwd
.\\.venv\\Scripts\\python src/kernels/bench/verify_e2e.py        # E2E correctness
.\\.venv\\Scripts\\python src/kernels/bench/verify_grads.py      # Gradient verification
.\\.venv\\Scripts\\python src/kernels/bench/profile_benchmark.py # torch.profiler 4-way
```

---

## Verified Correct (May 16, 2026)

- **RoPE**: Bit-for-bit identical to Qwen3MoE
- **SSMax init**: `1.0 / log(max_pos_emb / 2)` — prevents 6× over-sharpening
- **output_attentions**: Works, returns actual weight tensors (manual fallback)
- **SDPA**: Flash Attention enabled, verified on RTX 3050 (compute 8.6)
- **Imports**: All clean, no crashes
- **Forward + backward**: Verified end-to-end with loss computation
- **MC simulation**: Correctly accounts for router_lambda normalization
- **Router bias**: Non-trainable, heuristic updates work correctly
- **Noise expert removed**: All references cleaned, model works with n-3 special experts
- **Ablation**: BiBo outperforms Qwen3MoE on sorting task (lower loss, fewer params)
- **Dense MLP Triton kernel**: Correct (max_diff=4.88e-04 fp16), 1.12x full model speedup, 20.7% memory savings (May 28, 2026)
- **Noise expert removed**: All references cleaned, model works with n-3 special experts
- **Ablation**: BiBo outperforms Qwen3MoE on sorting task (lower loss, fewer params)

---

## Git Conventions

- Push to main and revert to old commit hash when user asks for it.
- `logs/`, `bugs/`, `shaurya_notes.md`, `.venv/`, `wandb/`, `research_on_activations/` are all gitignored

---

## When In Doubt

1. Read `src/configuration_bibo.py` for all config params
2. Read `src/modeling/attn/base.py` for attention logic (SDPA + SSMax)
3. Read `src/modeling/ffn/moe.py` for MoE dispatch logic
4. Read `src/modeling/ffn/router.py` for routing logic (logit norm + bias heuristics)
5. Read `docs/ssmax.md` for SSMax theory; `docs/xsa.md` for Exclusive Self Attention
6. Read `docs/deprecated.md` for removed components
7. Read `docs/configuration_guide.md` for tuning guidance
8. Read `shaurya_notes.md` for research insights and findings
