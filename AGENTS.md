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
3. **Shared Conv1D expert** — causal convolution (gated, SwiGLU-style), always-active WHEN enabled. **OFF by default (since Jun 27 2026)** so BiBo stays param-matched to a no-shared baseline; opt in with `use_shared_expert=true`.
4. **MiMo-V2.5 / DeepSeek-V3 router** — sigmoid scoring, auxiliary-loss-free bias (selection-only) updated by `b += u·sign(mean−load)`, `norm_topk_prob` (top-k sum-to-1), `routed_scaling_factor`. Verified bit-equivalent to MiMo's gate (`src/.autoresearch/bench_router_vs_mimo.py`). No Skywork logit-norm.
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
    │   ├── base.py                # BiBoAttention — minimal shell: proj/QK-norm/RoPE/cache/SSMax → flavor dispatch → XSA
    │   ├── swa.py                 # swa_attention — SWA flavor (EAGER ONLY: band mask + sink; kernel target)
    │   ├── full_attention.py      # full_attention — global flavor (SDPA is_causal fast path / mask path)
    │   ├── ssmax.py               # apply_ssmax_query_scaling (mask-aware context_lens)
    │   ├── xsa.py                  # apply_xsa (Exclusive Self Attention rejection)
    │   └── utils.py               # repeat_kv, causal_band_mask, padding_bias, eager_attention_forward
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

**MoE**: First and last layers are dense MLP (layers 0 and N-1; `mlp_only_layers=[0, N-1]`). All remaining layers are MoE. Router = MiMo/DeepSeek-V3 sigmoid gate (no logit-norm). Bias heuristics for load balancing. Router bias is `requires_grad=False` (not optimizer-managed, updated heuristically).

**Expert layout (PolyGLU)**: `polyglu_expert_multiplier` groups of 3 (SiLU-GLU, ReLU²-GLU, Tanh-GLU) + `special_expert_pairs` × (Identity, Zero). Config default: 2×3 + 1×2 = 8 routed. **Bench configs (Jun 27 2026): 3×3 + 1×2 = 9 GLU + 2 specials = 11 routed**, shared expert OFF — param-matched to Qwen's 9 experts on BOTH total (137.5M) and active (71.4M).

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
| `norm_topk_prob` | True | MiMo/DeepSeek-V3: renormalize the top-k routed weights to sum to 1 (÷ their sum). Default flipped True (Jun 28). |
| `routed_scaling_factor` | 1.0 | MiMo/DeepSeek-V3 final routed-weight scale, applied after `norm_topk_prob`. 1.0 = no-op (MiMo-V2.5). |
| ~~`use_router_logit_norm` / `router_lambda`~~ | removed | **Skywork logit-norm REMOVED (Jun 28 2026)** — MiMo has none. Router is pure MiMo. |
| ~~`moe_shared_scaling`~~ | removed | **DEPRECATED (Jun 28 2026)** — shared expert adds directly (DeepSeek-V3/Gemma), no learned/MC scalar. |
| `mlp_only_layers` | [0, N-1] | Which layers use dense MLP instead of MoE (first + last) |
| `partial_rotary_factor` | 0.334 | **Dim-wise** partial RoPE: fraction of EACH head's `head_dim` that rotates; the rest is NoPE. Replaces head-wise `rope_nope_ratio` (Jul 1 2026). `rope_dim=even(round(factor·head_dim))`. 1.0 = full RoPE. No KV-alignment constraint (applies to all heads uniformly). |
| `rope_theta` | 1e7 | RoPE base. Default raised 10000→1e7 (matched to dim-wise partial RoPE / MiMo-V2.5). |
| `hybrid_layer_pattern` | None | Per-layer list: 1=sliding-window (SWA), 0=full/global. None = all-global (current). |
| `sliding_window` | 128 | SWA window `W` (keys visible per query on windowed layers). |
| `add_swa_attention_sink_bias` | True | Learnable per-head attention sink on SWA layers (unscaled; the norm). |
| `add_full_attention_sink_bias` | False | Sink on global layers (True → G1/G3; False → G2, current). |

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
    - ~~Unsynchronized bias buffers across DDP ranks~~ **FIXED (Jun 27 2026)**: `moe.py` all-reduces (SUM) the per-expert token counts at the bias-update threshold so load-balancing sees GLOBAL load, not just rank 0's shard; the sign()-based update is then identical on every rank → bias stays bit-synced (verified: 2-rank gloo, max|Δbias|=0). `wrap_ddp` sets `broadcast_buffers=False` so per-rank accumulation isn't clobbered each forward. All ranks hit the threshold the same step (identical per-step token count) → the collective stays in lockstep; eval (training=False) skips it, no deadlock.
    - CausalConv1D incompatible with sequence parallelism (needs halo exchange)
    - SSMax uses local kv_len (needs global seq_len under SP)
    - MC simulation non-deterministic (needs seed)

12. **Router noise across ranks is NOT an issue** — router is replicated, only experts are parallelized in EP.

### Ablation Results (Kaggle)

13. **BiBo vs Qwen3MoE on sorting task** — BiBo converges faster and reaches lower final loss (~0.1 vs ~0.25) despite fewer params (8.3M vs 11.2M).

14. **Router confidence insight**: BiBo's logit normalization gives moderate confidence (0.4-0.7) across top-k experts = better expert utilization. Qwen's raw softmax gives 0.9+ top-1 = wastes the other k-1 experts. High top-1 confidence is BAD when top_k > 1.

---

## Important Design Decisions

1. **Hybrid SWA + global attention — IMPLEMENTED (Jul 1 2026)** via `hybrid_layer_pattern` (1=SWA, 0=global; None=all-global G2, the bench default). **BINDING VERDICT (2026-07-01), honored by the implementation: SWA layers use ONLY an (unscaled) per-head attention sink — no SSMax, no value-scaling; global layers pick one of {SSMax+sink, SSMax-only, sink-only}, value-scaling never used** (G1 = sink+SSMax is config-guarded until the C-scaled sink lands). Full rationale (why SSMax is redundant on windowed layers, the SSMax×sink coupling / Option A, the head_dim-based escape analysis, why we skip value-scale and a per-dim sink) + reference impl in `docs/attention_layers.md`. See also `docs/ssmax.md`.
2. **SSMax init**: `1.0 / log(max_pos_emb / 2)` — ensures attention starts ~neutral, not 6× sharper than standard.
3. **Shared expert is NOT routed** — when enabled it's always active. **OFF by default (Jun 27 2026)**; `use_shared_expert=False` is the `BiBoConfig` default and `moe.py`'s fallback. The `moe_shared_scaling` 10K-iter MC in config init is now guarded by `use_shared_expert` (no wasted sim when off).
4. **`output_attentions=True`** works (falls back to manual attention).
5. **Router bias is non-trainable** — `requires_grad=False`, updated via heuristic `.add_()`.
6. **Noise expert was removed** — no evidence it helps. Identity covers the "dump bucket" use case. See `docs/deprecated.md`.
7. **Conv router** — gives router local context (sees previous `kernel_size-1` tokens). Novel — no other MoE paper uses convolutional routing.
8. **Logit norm prevents expert waste** — when top_k > 1, normalization ensures all selected experts contribute meaningfully (not just top-1 dominating).

### Kernel Design Decisions (June 2026)

9. **Triton kernels have custom backward** — all MoE Triton kernels use custom Triton backward kernels, not PyTorch fallback. Gold standard: `_FusedSwiGLUFull` in dense_mlp.py (matched forward+backward Triton kernels with autotune).

9b. **`is_causal=True` SDPA, no precomputed mask (June 26 2026)** — the model no longer builds an
    explicit additive triangular mask; attention passes `is_causal=q_len>1, attn_mask=None`. An
    explicit float mask makes the mem-efficient cutlass FMHA compute all S² blocks; `is_causal`
    lets it SKIP the upper triangle — **confirmed on T4 sm_75: attention backward 100ms→47ms**,
    step 727→599ms, MFU 13.9%→16.8%. Bit-equivalent (is_causal vs manual-causal logits Δ3e-6).
    `_update_causal_mask` removed; `masks.py` dormant (re-exported only). Padded batches ARE
    supported since Jul 2 2026 (2D mask threaded to attention; None keeps this fast path) — the
    packed training pipeline still passes no mask, so the is_causal skip is unchanged there.
    **optmaxx arc (T4, B16/H512): morning 2.4s/~5% MFU → now 599ms/16.8%** (hidden 320→512+head_dim
    128, tl.dot-CE→cuBLAS-chunked-CE, additive-mask→is_causal). Now GEMM-bound (~69% cuBLAS); near
    the practical T4 ceiling for this size. Next lever = flash-attention-turing (attention 8%→~4%).

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

- **(July 2 2026) Attention/infra review-fix batch (10 findings).** Fixed across `src/`:
  (1) **Padded batches now WORK** — `BiBoModel.forward` threads a 2D (B,K) padding mask to every
  layer (was hardcoded `None` → masks silently ignored, verified bit-identical before the fix);
  attention folds it into its causal/band mask (`finfo.min`, NaN-safe), position_ids are
  mask-aware (cumsum−1, left-pad safe), and **SSMax's per-query `n` counts REAL keys only**
  (`context_lens = mask.cumsum` — grid positions over-counted by the pad width). All-ones masks
  (generate's default) short-circuit to `None` so the packed/unpadded is_causal fast path is
  untouched. Verified: padded forward == unpadded reference (4e-7), padded greedy generate ==
  unpadded generate. Only 2D masks accepted (4D raises).
  (2) **Attention modularized into per-flavor files (same day):** `base.py` is now a minimal
  shell (proj → QK-norm → partial RoPE → cache → SSMax → flavor dispatch → XSA → o_proj);
  the flavors live in `attn/swa.py` and `attn/full_attention.py` (imported like ssmax/xsa),
  with shared helpers (`causal_band_mask`, `padding_bias`, `eager_attention_forward`) in
  `attn/utils.py`. **SWA is EAGER-ONLY by design** — no SDPA/mem-efficient on windowed layers;
  the eager core is the exact numerics target for a future dedicated sink-aware banded kernel.
  **Global layers** keep the SDPA `is_causal` fast path; padding / cached prefill / G3 sink use
  SDPA with an explicit mask — the per-head sink rides as one extra additive-mask column (β_h)
  + a zero K/V column (no fp32 S² prob materialization; sink grads flow through the mask,
  Δ9e-10 vs eager). All flavor masking is built INSIDE the flavor modules; base passes only
  q/k/v/sinks + the raw 2D padding mask.
  (3) **SWA KV cache is window-evicted**: `DynamicCache(config=...)` + new `config.layer_types`
  builds `DynamicSlidingWindowLayer` for SWA layers → O(W) decode memory (verified swa_kv=7 vs
  global_kv=24 at W=8; incremental decode == full forward 3.7e-7). Bottom-right mask slices
  (`[..., -kv_len:]`) keep geometry correct for the cropped window.
  (4) **Dynamic-NTK no longer syncs per step**: the model passes `seq_len` as a host int
  (`get_seq_length()+q_len`) to the rotary — no `int(position_ids.max())` GPU sync / compile
  graph break on the default path; the sync survives only as a fallback for external callers.
  (5) `config.sliding_window` serializes as **None when no layer is SWA** (interop: HF machinery
  keys off it); `config.layer_types` always emitted. (6) `eager_attention_forward` uses its
  `sinks` param (was reading the module attr). (7) type hints are `Optional[Cache]` (legacy
  tuple lie removed); (8) both forwards take `**kwargs` (GenerationMixin 5.x injects unfiltered
  model inputs); (9) deprecated `config.use_return_dict` dropped (`return_dict` defaults True).
  (10) this file's Design Decision #1 updated (SWA is implemented, not "removed"). All verified:
  24/24 checks + fp16-autocast hybrid+padding fwd/bwd NaN-free (RTX 3050, torch 2.6).
- **(July 1 2026) Full-codebase code-review fix batch.** Fixed across `src/`:
  (1) **transformers 5.x port** — the model targets HF 5.x now: the custom `prepare_inputs_for_generation`
  + legacy-cache (`to/from_legacy_cache`) + the broken StaticCache mask branch (undefined
  `_prepare_4d_causal_attention_mask_with_cache_position`, removed `get_max_length`) were deleted;
  `BiBoModel.forward` returns the `Cache` object directly and relies on `GenerationMixin`'s default
  prepare-inputs. **`model.generate()` and `use_cache=True` now work** (verified).
  (2) **`num_experts`/`num_shared_experts` config knobs removed** — `num_experts` is now derived
  ≡ `num_routed_experts` (all PolyGLU + specials); the shared expert is a single always-on module.
  (3) **top-k validation** now checks `num_experts_per_tok ≤ num_routed_experts` (was `num_experts`,
  which over-allowed and crashed `torch.topk`).
  (4) **config `head_dim`/`rope_dim` derived AFTER `super().__init__`** so a stale serialized value
  can't survive a `config.json` edit of `partial_rotary_factor`/dims.
  (5) **SWA sliding-window now enforced at `q_len==1` decode**; **SDPA cached-prefill** (`q_len>1,
  kv_len>q_len`) uses an explicit bottom-right mask (is_causal is top-left-wrong there).
  (6) **XSA aligns V to the query positions** (`value_states[..., -q_len:, :]`) — fixes a decode-time
  `q_len!=kv_len` crash.
  (7) **ReLU² expert squares in fp32** (fp16 overflow→NaN guard on T4).
  (8) **DDP bias-update is step-triggered** (host-side forward-step counter, no per-step `.item()`
  sync, no ragged-batch `all_reduce` deadlock); MoE boundary slices use one `.tolist()` (was a sync
  per expert).
  (9) **dynamic-NTK is stateless** (order-independent `inv_freq`; in-window it's a no-op).
  (10) **G1 (global sink + SSMax) is GUARDED** (config raises) — the C-scaled sink is NOT implemented
  yet; enable only when it lands. (11) `_init_weights` now resets RMSNorm(→1.0)/Conv1d; RMSNorm eps
  default 1e-5→1e-6; router `getattr` fallbacks match config defaults; dead `masks.py` + unused
  imports/attrs removed; new config guards (`num_key_value_heads>0`, `bias_update_threshold>0`,
  SWA requires positive `sliding_window`, `mlp_only_layers` deduped).
  ⚠️ **Requires transformers ≥5** (installed 5.7.0). ⚠️ Existing checkpoints incompatible. NOT done
  (optional perf): building the SWA mask / SSMax `log_n` once at model level instead of per-layer.

- **(June 30 2026) ALL custom kernels REMOVED — BiBo + bench run pure PyTorch eager + torch.compile.**
  The entire `src/kernels/` tree (MoE dispatch, XSA, fused-linear CE, fused conv-router, Liger
  RMSNorm/RoPE patches), the kernel benches, the vendored fused Muon (`bench/fused_muon.py`), and the
  kernel-only bench/profile scripts (`ckpt_compare.py`, `e2e_ce_matrix.py`, `e2e_profile_1024.py`,
  `_compare_final.py`, `profile_bibo.py`) were deleted. **Kernels now live in a SEPARATE repo** and are
  consumed from there. Consequently: `bench/models.py` no longer has `apply_triton_kernels`; `train.py`
  dropped `--no_triton`/`--no_fused_ce` and the kernel-apply block; CE is always standard
  `F.cross_entropy` (the `use_fused_linear_ce` path is gone from `models.py`); the conv router runs the
  eager `nn.Conv1d` path (the `use_fused_conv_router` fast path is gone from `router.py`); Muon is
  eager-only (the `use_fused_muon` path is gone from `optim.py`); configs dropped `use_fused_ce` and the
  dead `triton:` key. **Everything below this entry that describes a custom Triton/Liger kernel, a
  `src/kernels/...` path, or fused Muon is HISTORY — those files no longer exist in this repo.** Speed
  levers that remain in-repo: `torch.compile` (`hardware.compile`) and `compile_optimizer`.
- `moe_shared_scaling` is a **learnable nn.Parameter** (per MoE layer) as of this session — the config float is only its init. Rationale (from the shared-expert research): no production MoE hardcodes a fixed shared scalar — DeepSeek/Gemma just add (Gemma sizes 3×), Qwen2 used a learnable per-token gate (dropped in Qwen3), Qwen3/MiMo-V2 dropped the shared expert entirely. The old fixed 0.40 was ~7.5× below the measured init-balance (~3.0) AND the MC auto-formula diverges as experts scale; a learnable scalar (LayerScale-style) fixes both. Disable the shared expert with `use_shared_expert=False` / `--no-shared-expert`. ⚠️ The scalar lands in the AdamW group **with weight_decay** (norms/biases do too in this repo) — WD gently pulls it toward 0; acceptable but worth excluding if it underperforms.
- `moe_shared_scaling=1.0` triggers a 10K-iteration Monte Carlo in config init (now used as the learnable param's INIT) — pass explicit value to skip
- The `legacy/` folder has the old monolithic code — don't touch it, it's reference only
- No tests currently exist — they were removed during cleanup. New tests needed.
- Qwen baseline requires real `transformers` package (not stubs)
- CausalConv1D needs halo exchange for sequence parallelism (future work)
- MoE dispatch loop is fine for ≤16 experts on single GPU; needs grouped GEMM for 64+ experts with EP
- All MoE Triton kernels now have custom Triton backward kernels (not PyTorch fallback)
- GQA uses `repeat_kv` + plain SDPA (NOT `enable_gqa` — that hit the slow MATH backend; see design decision #10, reversed Jun 25 2026)
- **(June 28 2026) Conv kernel REMOVED; BiBo kernel set is now exactly 3.** `conv_fused.py` (fused conv router + conv expert kernels) + its benches (`bench_conv.py`, `bench_conv_router.py`) and the conv tests/sections in `verify_grads.py`/`bench_isolated_kernels.py` are deleted; `__init__.py` and `bench/models.py` no longer import/patch it. **The 3 custom kernels: MoE (`moe_dispatch.py`), XSA (`xsa_fused.py`), fused-linear CE (`fused_ce.py`)** — plus external Liger RMSNorm/RoPE. The conv ROUTER capability still exists in the model (`router_type="conv"` runs eager); only its Triton kernel is gone (bench configs use `"mlp"`). **`dense_mlp.py` and `moe_grouped.py` also REMOVED** (dense MLP → torch.compile both BiBo and Qwen; grouped = dead tl.dot on T4). The Qwen baseline's CE uses **our** `fused_ce` kernel (`patch_qwen3_fused_ce`), same as BiBo — symmetric.
  **Planned kernels (add as benched, local + T4):** (1) a FUSED whole-router (conv OR mlp variant); (2) fuse the WHOLE MoE layer (RMSNorm → router → experts → RMSNorm) into one; (3) fuse the attention block into one. Each lands only after grad-equivalence + a T4 win vs the compiled baseline (Rules 1–4).
- **(June 26 2026) FlashAttention-Turing — PLANNED attention lever for seq ≥ 2k (scoped, NOT integrated).**
  Repo: `ssiu/flash-attention-turing` (https://github.com/ssiu/flash-attention-turing) — a from-scratch
  FlashAttention for **sm_75 / T4** (Dao's official flash-attn refuses Turing). Fits BiBo: **head_dim
  64/128** (we use 128 ✅), **fwd+bwd** (training), **causal + GQA + varlen** ✅; no dropout / KV-cache /
  local-mask (all fine — training is full-causal, attn_dropout=0). Benchmarks vs PyTorch mem-efficient
  (xformers, our current backend) on T4: **fwd causal 1.95×, bwd causal 1.51×, "for long sequences"**
  (forward 66% SoL; win shrinks at short S). **ROI: ~4% at S=1024 (NOT worth it — `is_causal` already
  captured the causal skip; design decision 9b). But attention is O(S²): at seq 2048 its share roughly
  quadruples → flash-turing's speed + O(S) memory become the real lever, and the win grows with S.**
  This is BiBo's regime (length-gen: SSMax/NoPE are about long context). **Integrate when we move to
  2k**: gate `flash_attn_func` behind a config flag with an **SDPA fallback** (so local/non-T4 still
  runs) + a **grad-exactness gate vs SDPA** before trusting it (Rules 1–2). API args differ from stock
  flash-attn (see `flash_attention_interface.py`). XSA (post-attn on V) + SSMax (pre-scales q) are both
  unaffected — drops in cleanly. **Build risk:** `pip install -v .` from source (CUDA 12.4, ~10–20 min);
  tested on torch 2.5.1/2.8 but we're on 2.6 — may need a build fix. Verify the build + grads on T4 first.
- **(June 24 2026) Benchmark in fp16, not fp32** — T4 (training GPU) is fp16-only. The old fp32 numbers in `docs/kernel_benchmark_report.md` are unreliable (see correction header there). Use `triton.testing.do_bench`, never the hand-rolled 3-sample timer.
- **(June 24 2026) MoE dispatch sync fix** — `triton_moe_experts_forward` now builds expert boundaries as CPU ints (was comparing CUDA scalar tensors per expert → implicit GPU sync per iter). Took the per-expert path from 0.84x regression → 1.40x vs eager (fp16).
- **(June 28 2026) MoE per-expert path got a MANUAL backward + fused fp32 combine (`_BiBoPerExpertMoE` in `moe_dispatch.py`).** Was autograd-composition (forward-only Triton GLU; autograd auto-generated backward → grad-accum `add_` + per-op `fill_` glue ≈ 21% of fwd+bwd). Now one custom `autograd.Function`: per-expert dX/dW cuBLAS GEMMs + the existing GLU fwd/bwd kernels + a **fused weighted-scatter/gather combine** (`_combine_scatter_kernel`/`_combine_bwd_kernel`: `(eo·w)`→fp32 atomic-add; bwd gathers `grad_out[tok]`, emits `grad_eo=go·w` and `grad_w=Σ_h(go·eo)` — collapses the old mul+cast+index_add into 1 kernel each way), and ONE `index_add_` per expert for grad_hidden. **Scales m and n**: loop branches on expert index — `< num_polyglu`(=3·`polyglu_expert_multiplier`) GLU (weight slot e), `< zero_start` Identity (weighted passthrough, no GEMM), else Zero (skip). **Ported from `triton-kernel-fused` (T4-verified there: fwd+bwd 1.80→3.03× / 1.11× less mem vs compiled Qwen3MoE-eager, backward 2.5→5.9× — specials skip the GLU backward).** **Autocast-correct**: backward casts params to the compute dtype for the matmuls and grads back to each param's dtype on return (mirrors autograd-AMP). Verified vs eager `BiBoFusedExperts` on the 3050: fp32 grads ~1e-7, autocast-fp16 out ~6e-6 / weight-grads bit-identical, NaN-free, full-model patched loss matches eager 1.4e-5 — across m∈{2,3,4}, n∈{0,1,2}. All GLU kernels' tanh = **`libdevice.tanh`** (hardware, matches eager `torch.tanh` exactly, stable) — replaced BiBo's old forward `(e^2x−1)/(e^2x+1)` which OVERFLOWS to NaN for large gate; now bit-identical to tkf (cross-check `src/.autoresearch/bench_moe_vs_tkf.py`: fp32 0.0 across m∈{1..4}/n∈{0..2}). ⚠️ `_combine_bwd_kernel` reduces grad_w over the full H in one block (fine for hidden_size ≤ ~1024; needs a 2-stage reduce above that). Old forward-only `triton_fused_glu_activation`/`_FusedWeightScatterFunction` kept as reference (unused by the patched path). The tkf bench (`bench.py`) grew `--profile` (torch.profiler launch-count + per-op CUDA-time map) and `--no-special` (A/B the 9-GLU-only vs 9+Identity+Zero stack).
- **(June 24 2026; module REMOVED Jun 28) Grouped-GEMM MoE** (`moe_grouped.py`) — forward ~2–2.5x, fwd+bwd ~2x at 4k–8k tokens vs eager (fp16). **Certify the 16384-tok training shape on T4** — the 4GB RTX 3050 can't measure it (thermal + allocator pressure). Cert harness: `src/kernels/.autoresearch/bench_real.py`. Stays opt-in until T4-verified; per-expert path remains default.
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
- **(June 28 2026) CE upgraded to FUSED-FWD+BWD — `_CEFusedFwdBwd` replaces the cuBLAS-recompute
  `_CECublasChunked`** (`fused_ce.py`, still the default `fused_linear_cross_entropy`). CE's grad
  w.r.t. logits = (softmax−onehot)/n needs ONLY logits+labels (loss is scalar → upstream grad is a
  scalar), so the FULL gradient is computed in the FORWARD chunk loop while each (chunk,V) tile is
  live, then discarded — backward is just a scalar scale of stashed grad_h/grad_w. **3 GEMMs over
  the data (logits, grad_h, grad_w), NO recompute** (recompute was the 4th GEMM = a pure latency tax,
  no memory upside). Per chunk: cuBLAS logits → `_fwd_reduce_kernel` (per-row online-softmax → lse+tgt,
  no fp32 (C,V) buffer; replaces the `.float()+logsumexp+gather`) → `_grad_logits_inplace` (2D-grid,
  overwrites logits in place with grad) → grad GEMMs. ⚠️ The 2D-grid grad kernel is load-bearing: a
  one-program-per-row fused reduce+grad kernel was tried and LOST on T4 (only ~chunk programs, V
  streamed twice serially → launch/occupancy-bound); the 2D grid (~chunk/32 × V/256 programs)
  saturates the SMs. **This BEATS Liger's fused-linear CE — the new external baseline.** T4 ce_fit
  (N=16384/V=81000/H512, 192MB budget, --compile): ours **260ms / 904MB / 0.76× compiled** vs
  **Liger@chunk1024 321ms / 836MB** (19% faster, +68MB) vs **Liger@chunk2048 283ms / 1083MB** (faster
  AND 180MB less). Curve flat 260–264ms across 192–384MB (GEMM-bound, no longer launch-bound). Grads
  are TIGHTER to fp32 eager than Liger's: grad_hidden rel **1.9e-3** (Liger 1.1e-2), grad_weight
  **7e-4** (both); loss bit-identical to Liger, eager to 1e-6. Standalone kernel/bench lives in the
  `triton-kernel-fused` repo (autoresearch round 3). Default budget = 192MB (T4 latency knee). The
  recompute `_CECublasChunked` is removed from `fused_ce.py` (dominated); `_FLCE`/`fused_linear_cross_entropy_tldot`
  kept as the dead tl.dot reference. ⚠️ **Still SUPERSEDED at ce_fit by compiled std CE on raw speed**
  (197ms vs 260ms) — fused CE is the MEMORY play (3.4× less peak) + the only path when std CE OOMs
  (≥4096 tok / long-context); both models still default to compiled std CE (`use_fused_ce: false`)
  when logits fit. The fused kernel is now the right choice the moment memory is the constraint.
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
- **(June 27 2026) Conv router FULLY fused** (`_FusedConvRouterFull` in `conv_fused.py`) — the WHOLE
  router (conv projection → **sigmoid fused into the conv store epilogue** → +bias selection →
  top-k → unbiased-weight gather) now collapses into ONE autograd node. Backward is manual:
  scatter grad into selected scores → ×sigmoid′ (`scores*(1−scores)`) → reuse the existing
  transpose-free `_conv_router_dx_kernel`/`_conv_router_dw_kernel`. `torch.topk` is kept in eager
  (robust tie-break, grad-free); `norm_topk_prob` is applied in eager so autograd carries its
  Jacobian. **Bench `src/kernels/bench/bench_conv_router.py`** (RTX 3050, fp16): fwd ~1.95×,
  **fwd+bwd 2.83× at the training shape (B16/S1024)** (1.4× at small B8/S512 — launch-bound, the
  dx/dw bwd doesn't amortize). **GEMM/launch count: matmul-conv 7→3, total launches 128→59** — the
  fused router never issues more matmuls than eager's cuDNN conv (1 fwd + 2 bwd) and collapses the
  ~6 glue launches (permute×2, pad, cast, sigmoid, bias, gather) into the conv kernel. Grad-exact
  vs an eager reference holding the conv path fixed (fp32: idx-agree 1.0, w Δ1.2e-7, gx Δ8e-6; gw
  Δ4e-4 rides the dw-kernel's documented TF32 long-reduction caveat). fp16 routing diverges from
  true cuDNN-conv eager by ~0.03% because the fused path sigmoids fp32 logits (MORE precise than
  eager's fp16-rounded logits) — expected, not a bug. E2E-verified (full model trains, no NaN).
  Fused path covers the conv-router default (gate_type=sigmoid, router_activation=none); other
  combos fall back to `_original_forward`. ⚠️ Only active when `router_type="conv"` (all current
  bench configs use `"mlp"`).
- **(June 27 2026) Muon switched to the Moonlight/Kimi recipe; muon_lr 0.02 → 3e-4.** The old Muon
  was raw modded-nanogpt-style: a plain cubic Newton-Schulz (which UNDER-orthogonalized, SVs~0.4),
  coupled weight decay, momentum applied AFTER orthogonalization, and lr=0.02 — an effective per-element
  step ~6-10× hotter than any production Muon run and shape-inconsistent across layers. Now (`bench/optim.py`):
  (1) tuned **quintic** Newton-Schulz `(3.4445,-4.7750,2.0315)` in fp32 (SVs→~1); (2) momentum on the raw
  grad THEN orthogonalize (canonical order); (3) **consistent-RMS scaling** `update *= 0.2·√max(A,B)`
  (Moonlight arXiv:2502.16982) so the update element-RMS is ~0.2 for EVERY matrix shape; (4) decoupled
  (AdamW-style) weight decay. With this, `muon_lr` lives in the AdamW-aligned band the recent reports use
  (Kimi K2 2e-4, Moonlight 4.2e-4, Mellum 3e-4); default **3e-4**. Verified `src/.autoresearch/test_muon_lr.py`:
  per-element `|Δp|/lr` is flat at ~0.19 across square/tall/wide/gate_up AND 3D per-expert shapes (was
  0.085 + shape-dependent with the cubic); tinylm passkey trains stably at 3e-4. **Per-expert Muon (both
  models):** the stacked 3D expert tensors (`...experts.gate_up_proj` / `...experts.down_proj` — identical
  layout in BiBo and Qwen3MoE) are orthogonalized PER EXPERT SLICE — `newton_schulz_iteration` batches over
  the expert dim, `create_optimizer` routes them to Muon. So Muon now covers attention + dense MLP + ALL
  experts (**95.9M** of ~137M; was 11M when experts sat on AdamW). 3D **conv kernels** (`gate_conv` — used by
  BOTH the conv router `router_type=conv` AND the conv shared expert `shared_expert_type=conv`) are excluded
  by name and stay on AdamW (a conv kernel isn't a matrix to orthogonalize). AdamW now holds only embeddings
  (~41.5M) + norms + conv. Verified: experts_in_muon 16/16, conv_in_muon 0, both models 4-step NaN-free.
  ⚠️ The tinylm passkey task saturates for both 3e-4 and 0.02 — it confirms attribution + stability, NOT
  that 0.02 is worse; the real LR discriminator is the LM-loss Kaggle bench.
- **(June 27 2026) Muon-step fusion investigated → compile is the only lever; Triton/batching rejected.**
  Profiled the Muon step (`bench/bench_muon.py`): **GEMM-bound (~68-71% CUDA), gemm count = the NS
  algorithmic floor** (3 matmuls/iter), cuBLAS bmm engine. So "fewer GEMMs than eager" is impossible
  (unlike the conv router, NS has no elementwise glue to absorb into the matmul) and a Triton `tl.dot`
  megakernel would LOSE (cuBLAS > Triton-GEMM, proven 3× in this repo; a 512² matrix won't fit SRAM).
  **Cross-param shape-bucketing** (one batched NS over all same-shape matrices) was implemented, benched,
  and **REVERTED**: numerically exact (1e-6 vs per-slice) but only ~1.1× on the FLOP-bound expert matmuls
  at 2× memory → thrashed the 4 GB local GPU (22× slower with the model resident). Step stays **per-param**
  (3D experts still batch over the expert dim inside NS — cheap). The surviving lever is **`compile_ns`**
  (`hardware.compile_optimizer: false` default): `torch.compile`+cudagraphs on the per-param Newton-Schulz
  cuts launch overhead without the batched-transient memory blowup. **Kaggle-only — torch.compile is broken
  locally** (triton `AttrsDescriptor` mismatch); `bench/bench_muon.py` auto-skips it locally, run it on T4
  to A/B eager-vs-compiled (numerics gate + speed) before trusting.
- **(June 28 2026) Router = MiMo-V2.5 / DeepSeek-V3 gate, verified equivalent; `moe_shared_scaling`
  REMOVED; Skywork logit-norm REMOVED (pure MiMo).** Changes:
  1. **`routed_scaling_factor` added** (config, default 1.0 = MiMo-V2.5's no-op). The router now
     matches MiMo's gate exactly: sigmoid scoring → bias-added-for-SELECTION-only → weight gathered
     from UNBIASED scores → `norm_topk_prob` (eps 1e-20) → `× routed_scaling_factor`. Verified bit-for-bit
     (norm off) / 2e-7 (norm on, fp32 reduction noise) vs a faithful `MiMoV2MoEGate` replica with tied
     weights — outputs, selected-expert sets, grad→gate-weight, grad→input, and loss all match. Bench:
     `src/.autoresearch/bench_router_vs_mimo.py`. (BiBo already did sigmoid+bias+gather; only the scale
     was missing. MiMo's group/device-limited routing is a no-op at its shipped `n_group=1` and BiBo is
     single-node → not implemented.)
  2. **`moe_shared_scaling` REMOVED (DEPRECATED).** The learnable per-layer scalar + its 10K-iter MC
     init are gone from `config` and `moe.py`. When the shared expert is enabled it is now **added
     directly** (`final_routed + shared_combined`, DeepSeek-V3/Gemma style). Old configs/benches passing
     `moe_shared_scaling=` are harmlessly ignored (lands in `**kwargs`). Shared expert still OFF by default.
  3. **Skywork logit-norm REMOVED entirely (pure MiMo).** `use_router_logit_norm` + `router_lambda`
     deleted from `config` and `router.py` — MiMo/DeepSeek-V3 has **no** z-score logit normalization
     (the only normalization in MiMo's gate is `norm_topk_prob`, the top-k sum-to-1, which is a
     different op). Selection is now just `scores + bias`, always. `norm_topk_prob` default flipped
     **False→True** to match MiMo. Old configs passing `router_lambda=`/`use_router_logit_norm=` are
     harmlessly ignored (`**kwargs`). The Triton fused-router kernel (`moe_dispatch.py`
     `_fused_router_kernel` / `triton_fused_router`) and its `bench_moe.py` test were **REMOVED**
     (Jun 28) — it carried the Skywork logit-norm; router is eager-only (sigmoid+bias is a trivial
     2-op elementwise the compiler fuses for free). The fused CONV router (`conv_fused.py`) has no
     logit-norm, so eager and fused-conv agree. Stage-by-stage MiMo parity (scores → selection →
     before-norm → after-norm) verified in the bench: all 0.0 except 3e-8 post-norm; norm takes
     Σw 2.07→1.0000. (The 3e-8 is fp32 non-associative summation — MiMo `topk(sorted=False)` vs
     BiBo `sorted=True` sum the k weights in different orders; not a divergence.)
  4. **Router + MoE combine now fully fp32 (MiMo-style).** The gate keeps weights in **fp32**
     (`router.py` returns `norm_weights.float()`, no down-cast), and `BiBoFusedExperts` accumulates
     the weighted combine in an **fp32 output buffer**, casting to hidden dtype only at the end. So
     the whole router→combine path is fp32 even under bf16/fp16 training (verified fp32/fp16/bf16
     fwd+bwd NaN-free). ⚠️ The OPT-IN Triton MoE kernels (`triton_moe_experts_forward`, `moe_grouped`)
     now receive fp32 weights — re-verify/cast there if you enable them; the default eager path is done.
  5. **Default training kernel set changed** (`bench/models.py::apply_triton_kernels`, BiBo branch):
     **dropped the dense-MLP SwiGLU Triton patch** (`patch_dense_mlp_with_triton`) — torch.compile's
     lifted SiLU-mul ties a hand kernel (it's noise), so dense-MLP activation is left to the compiler;
     **added the XSA kernel** (`patch_xsa_with_triton()`, gated on `config.use_xsa`); fused CE stays on
     (`use_fused_ce=True` → `config.use_fused_linear_ce`). Net BiBo Triton set now = Liger RMSNorm+RoPE,
     MoE dispatch, XSA, fused-linear CE — NO dense-MLP kernel, NO conv-router kernel (removed Jun 28).
     The Qwen branch ALSO dropped `patch_qwen_dense_mlp_with_triton` (Jun 28) — both models now
     compile dense MLP and both route CE through OUR `fused_ce` kernel (`patch_qwen3_fused_ce`):
     fully symmetric except BiBo's PolyGLU experts vs Qwen's homogeneous SwiGLU.
- **(June 27 2026) BiBo↔Qwen now PARAM-MATCHED on BOTH axes; shared expert OFF by default.** Bench
  expert layout bumped 6→**9 GLU** (`polyglu_expert_multiplier: 2→3`) + 2 param-free specials = 11
  routed; Qwen `num_experts: 8→9`. A PolyGLU expert and a SwiGLU expert are param-identical, so 9
  GLU == 9 SwiGLU exactly; specials are free; **shared expert turned OFF** (it's always-active, so
  it would otherwise leave BiBo +13% active with no Qwen equivalent — the one asymmetry no Qwen
  expert count can close). Result: **BiBo 137.48M/71.41M active vs Qwen 137.47M/71.41M, Δ<0.01M on
  both** — the ONLY architectural difference is diverse PolyGLU activations + param-free specials vs
  homogeneous SwiGLU (no size confound either direction). **Active-param convention** (`count_params`
  in `bench/models.py`): BiBo's active-routed denominator is the **GLU-expert count (9), NOT
  num_routed_experts (11)** — params live only in GLU experts, so including the 2 free Identity/Zero
  specials in the denominator would under-count active (logged 68M instead of 71.4M) and break the
  apples-to-apples vs Qwen's all-GLU 2-of-9. It's an UPPER BOUND: real active drops below 71.4M
  whenever the router actually picks a free special (BiBo-only capability; only ever makes it cheaper).
  `top_k=2` unchanged (more experts add
  specialization capacity, not active compute). `use_shared_expert` default is **False** in both
  `BiBoConfig` and `moe.py`; the `moe_shared_scaling` MC is guarded by it (no wasted 10K sim when
  off). Configs touched: `bibo.yaml`, `qwen3moe.yaml`, `bibo_vanilla.yaml`, `bibo_no_ssmax.yaml`
  (the last is a stale divergent 16L config — bumped for layout consistency only). ⚠️ Existing
  8-expert checkpoints are now architecture-incompatible (router/expert dims changed) — retrain.
- **(June 24 2026; REMOVED Jun 28 — per-expert is the only MoE path) `patch_moe_auto`** — per-call dispatch: grouped path for `n_tokens >= GROUPED_MIN_TOKENS` (default 4096), else the fixed per-expert path. Beats PyTorch eager across all token regimes by construction (both branches grad-verified). Tune `GROUPED_MIN_TOKENS` after T4 cert.
- **(July 1 2026) Hybrid SWA + attention sink + DIM-WISE partial RoPE implemented.** Per-layer attention type via `hybrid_layer_pattern` (1=SWA, 0=global; None=all-global). **SWA layers** = sliding-window band mask (`sliding_window`, default 128) + a learnable per-head **attention sink** (unscaled, appended as a value-less softmax column, dropped before the V matmul) + **SSMax forced OFF** (window caps `n` → SSMax redundant). **Global layers** = current behavior (G2: SSMax on, no sink) unless `add_full_attention_sink_bias`. Single parameterized `BiBoAttention` (NOT subclasses): global-no-sink keeps the SDPA `is_causal` fast path; anything with a sink/window/`output_attentions` runs the **eager core** (`_attn_bias_mask` builds causal-or-band; matches `src/.autoresearch/ssmax_sink_ref.py` / MiMo `eager_attention_forward`). QK-norm, XSA, GQA, cache shared across both. **Partial RoPE switched head-wise → dim-wise** (`partial_rotary_factor`, default 0.334): the first `rope_dim` of EVERY head rotates, rest NoPE; model-level rotary emits `rope_dim`-sized cos/sin; `rope_theta` default 10000→1e7. Removes the KV-group-alignment constraint entirely. Verified: all-global builds with 0 sink params; hybrid puts sink params + SSMax-off only on SWA layers; fwd+bwd NaN-free; SWA window exact in `output_attentions` (q=20,W=8 → keys 13–20). Full spec: `docs/attention_layers.md`. ⚠️ Existing checkpoints incompatible (RoPE layout + new params). Sink is ON for SWA by default per the verdict; global sink is opt-in (G1/G3 not wired into bench configs).
- **(June 26 2026; head-wise SUPERSEDED Jul 1 2026 — see above; kept for its ablation data) Partial RoPE head-split (`rope_nope_ratio=0.5` = 2:2, DEFAULT)** — first `round(num_heads*(1-ratio))` query heads + corresponding KV heads get full RoPE+NTK; the remaining heads are NoPE (no positional encoding). At the **12h/2kv default**: 6 RoPE+NTK heads (1 RoPE KV group) + 6 NoPE content heads (1 NoPE KV group) — clean KV-group split, no GQA geometry change needed (group size 6; 6%6=0). Empirical (synthetic passkey + MQAR length-gen, train@128, eval to 32×, 3 seeds): more NoPE monotonically improves extrapolation — pure NoPE hits XG=1.000 on both tasks vs full-RoPE+NTK 0.93 (passkey) / 0.33 (MQAR); 2:2 gets 0.99 (passkey) / 0.50 (MQAR). **2:2 chosen as the default** (not higher NoPE): our retrieval evals treat position as *noise* (RoPE's worst case), so they overstate NoPE's value for a general LM where position is signal — 2:2 keeps half the heads positional, and is the only partial ratio that's KV-aligned at 12h/2kv without changing GQA. Mechanism: RoPE position-encodes key tokens by *where* they're planted → breaks key-matching; NoPE heads match position-independently; SSMax sharpens; causal mask gives implicit scale-free position. **The boundary must align with KV groups** (config validation enforced; only {0.0, 0.5} valid at 12h/2kv). `rope_nope_ratio=0.0` restores original all-RoPE. **IID/downstream cost of NoPE is UNMEASURED** (our synthetic evals saturate to 1.00 at train length) — Kaggle ablation on real-LM perplexity needed before pushing the ratio higher (would require changing GQA geometry to unlock 0.25/0.75). See `src/.autoresearch/FINDINGS.md` (2026-06-26 V10 + MQAR).

---

## Kernels — moved to a separate repo (June 30 2026)

**There are no custom GPU kernels in this repo anymore.** BiBo and the Qwen baseline run pure PyTorch
eager + `torch.compile`. All Triton/Liger kernels (MoE dispatch, XSA, fused-linear CE, fused conv-router,
Liger RMSNorm/RoPE), the kernel benches, and the vendored fused Muon were removed and now live in a
**separate kernels repo**, consumed from there. Do not re-introduce `src/kernels/` here. See the top of
"Known Quirks / TODOs" for exactly what the removal changed in `models.py`, `train.py`, `optim.py`,
`router.py`, and the configs.

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
5. Read `docs/ssmax.md` for SSMax theory; `docs/xsa.md` for Exclusive Self Attention; `docs/attention_layers.md` for the SWA/global layer verdict (sink × SSMax × value-scale)
6. Read `docs/deprecated.md` for removed components
7. Read `docs/configuration_guide.md` for tuning guidance
8. Read `shaurya_notes.md` for research insights and findings
