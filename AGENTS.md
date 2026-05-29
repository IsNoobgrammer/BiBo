# BiBo — Agent Onboarding

> This file is the system prompt for any AI agent working on this repo.
> Read this FIRST before doing anything.

---

## What Is BiBo

BiBo is a **Mixture-of-Experts (MoE) Transformer** for causal language modeling. It's a research model — not a product. The goal is to explore diverse expert architectures and SSMax attention for long-context performance.

**Key differentiators from vanilla MoE (like Qwen3MoE):**
1. **SSMax** — learnable per-head query scaling (`scale * log(kv_len)`) that prevents attention fading at long sequences
2. **Diverse experts** — PolyGLU layout: groups of 3 GLU experts with different activations (SiLU, ReLU², Tanh) + Identity/Zero special experts
3. **Shared Conv1D expert** — always-active causal convolution (gated, SwiGLU-style)
4. **Router logit normalization** — `router_lambda` scales normalized logits + threshold-based bias heuristics for load balancing
5. **Flash Attention (SDPA)** — uses `F.scaled_dot_product_attention` when `output_attentions=False`
6. **Conv router option** — `router_type="conv"` gives router local context awareness

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
legacy/                            # Old monolithic code (DO NOT USE for new work)
```

---

## Environment

- **OS**: Windows (local dev), Linux (Kaggle)
- **Python**: Use `.\venv\Scripts\python` ALWAYS (never system python)
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **GPU (local)**: RTX 3050 Laptop 4GB — compute capability 8.6 (supports Flash Attention)
- **GPU (Kaggle)**: 2×T4 16GB
- **Transformers**: v4.50+
- **Run commands from repo root**: `c:\Users\shaur\OneDrive\Documents\BiBo`

### Quick smoke test
```bash
.\venv\Scripts\python -c "from src.modeling_bibo import BiBoForCausalLM; print('OK')"
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

**Attention**: SDPA (Flash Attention) by default. Falls back to manual matmul when `output_attentions=True`. GQA (fewer KV heads). QK-norm. SSMax query scaling.

**MoE**: First 2 layers and last layer are dense MLP (layers 0, 1, and N-1). All remaining layers are MoE. Router uses logit normalization. Bias heuristics for load balancing. Router bias is `requires_grad=False` (not optimizer-managed, updated heuristically).

**Expert layout (PolyGLU)**: `polyglu_expert_multiplier` groups of 3 (SiLU-GLU, ReLU²-GLU, Tanh-GLU) + `special_expert_pairs` × (Identity, Zero). Default: 2×3 + 1×2 = 8 routed experts.

---

## Key Config Parameters

| Param | Default | What it does |
|-------|---------|-------------|
| `use_ssmax` | True | Enable SSMax query scaling |
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
| `mlp_only_layers` | [0, 1, N-1] | Which layers use dense MLP instead of MoE (first 2 + last) |

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

1. **No sliding window / recurrent attention** — removed. Only standard softmax + SSMax.
2. **SSMax init**: `1.0 / log(max_pos_emb / 2)` — ensures attention starts ~neutral, not 6× sharper than standard.
3. **Shared expert is NOT routed** — it's always active. Only the Conv1D is shared.
4. **`output_attentions=True`** works (falls back to manual attention).
5. **Router bias is non-trainable** — `requires_grad=False`, updated via heuristic `.add_()`.
6. **Noise expert was removed** — no evidence it helps. Identity covers the "dump bucket" use case. See `docs/deprecated.md`.
7. **Conv router** — gives router local context (sees previous `kernel_size-1` tokens). Novel — no other MoE paper uses convolutional routing.
8. **Logit norm prevents expert waste** — when top_k > 1, normalization ensures all selected experts contribute meaningfully (not just top-1 dominating).

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

---

## Triton Kernels (`src/kernels/`)

**All custom GPU kernels MUST be written in `src/kernels/`.**

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
    ├── bench_moe.py         # MoE benchmark suite (correctness + perf)
    ├── bench_dense_mlp.py   # Dense MLP benchmark (correctness + perf)
    ├── bench_conv.py        # Conv fusion benchmark
    ├── bench_moe_fwdbwd.py  # MoE full fwd+bwd training step benchmark
    └── verify_e2e.py        # Full model E2E verification
```

### What's Optimized

| Component | Kernel | Speedup | Source |
|-----------|--------|---------|--------|
| RMSNorm | Liger-Kernel `LigerRMSNormFunction` | 8-9x | `patch.py` |
| RoPE | Liger-Kernel `LigerRopeFunction` | 2-3x | `patch.py` |
| MoE GLU Activation | Custom Triton `_fused_glu_act_kernel` | 1.5x (full model) | `moe_dispatch.py` |
| Dense MLP SwiGLU | Custom Triton `_fused_swiglu_kernel` | 1.8-2.25x kernel, 1.12x model | `dense_mlp.py` |
| Conv Permute+Act+Gate | Custom Triton `_fused_conv_gate_multiply` | 1.34-1.41x training | `conv_fused.py` |
| Router Scoring | Custom Triton `_fused_router_kernel` | Available | `moe_dispatch.py` |

### Dense MLP Kernel Design (May 28, 2026)

**Strategy**: Fuse gate_proj + up_proj into single GEMM + Triton SwiGLU activation.

The fused SwiGLU kernel (`_fused_swiglu_kernel`) replaces:
```python
gate = gate_proj(x)          # (M, I) — separate GEMM
up = up_proj(x)              # (M, I) — separate GEMM
activated = F.silu(gate)     # (M, I) — 1 intermediate tensor
result = activated * up      # (M, I) — 1 intermediate tensor
```
With:
```python
gate_up = F.linear(x, fused_weight)  # (M, 2I) — single GEMM (cuBLAS)
result = triton_fused_swiglu(gate_up) # (M, I) — 1 Triton kernel, 0 intermediates
```

**Benchmark results (RTX 3050, sm_86):**
- Kernel-level: **1.80-2.25x** speedup (activation fusion only)
- MLP module: **1.17-1.27x** (small/medium batches, GEMM-dominated at large)
- Full model fwd+bwd: **1.12x** additional over existing patches
- Memory: **20.7% reduction** (35 MB saved on dense MLP)
- Forward correctness: max_diff=4.88e-04 (fp16), 9.54e-07 (fp32)
- Loss: identical (8.582651 ≈ 8.582869, diff=2.17e-04)

### MoE Kernel Design (May 28, 2026)

**Strategy**: cuBLAS for GEMMs (already optimal) + Triton for activation fusion.

The fused GLU activation kernel (`_fused_glu_act_kernel`) replaces:
```python
gate, up = gate_up.chunk(2, dim=-1)  # 2 intermediate tensors
activated = F.silu(gate)              # 1 intermediate tensor
result = activated * up               # 1 intermediate tensor
```
With a single Triton kernel that:
1. Loads gate and up from the fused (M, 2I) tensor
2. Applies activation (SiLU/ReLU²/Tanh) in registers
3. Multiplies gate × up in registers
4. Stores result — **zero intermediate tensors in global memory**

**Benchmark results (RTX 3050, sm_86):**
- Full model fwd+bwd: **1.51x speedup** (337ms → 223ms)
- MoE-only forward: **1.51x** (inference), **1.34x** (small batch)
- Forward correctness: max_diff=2.38e-07 (essentially perfect)
- Loss: identical (8.612711 = 8.612711)

### Usage

```python
from src.kernels import patch_bibo_with_triton, patch_moe_with_triton, patch_dense_mlp_with_triton

model = BiBoForCausalLM(config).cuda()
patch_bibo_with_triton(model)          # RMSNorm + RoPE (Liger-Kernel)
patch_moe_with_triton(model)           # MoE GLU activation (custom Triton)
patch_dense_mlp_with_triton(model)     # Dense MLP SwiGLU (custom Triton)
# Also available: patch_conv_router_with_triton, patch_conv_expert_with_triton
```

### Running Benchmarks

```bash
.\venv\Scripts\python src/kernels/bench/bench_dense_mlp.py   # Dense MLP
.\venv\Scripts\python src/kernels/bench/bench_moe.py         # MoE layer
.\venv\Scripts\python src/kernels/bench/bench_conv.py        # Conv fusion
.\venv\Scripts\python src/kernels/bench/bench_moe_fwdbwd.py  # Full fwd+bwd
.\venv\Scripts\python src/kernels/bench/verify_e2e.py        # E2E correctness
```

### Rules for New Kernels

1. **All kernels go in `src/kernels/`** — never inline Triton in modeling code
2. **All benchmarks go in `src/kernels/bench/`** — one bench file per kernel
3. **Liger-Kernel first** — check if Liger already provides the op before writing custom
4. **cuBLAS for GEMMs** — don't write custom matmul kernels (cuBLAS is faster for typical shapes)
5. **Monkey-patch pattern** — kernels are applied via `patch_*` functions, never modify modeling code
6. **Benchmark before promoting** — run the bench script, record exact numbers
7. **Correctness is binary** — atol=1e-3 for fp16, atol=1e-5 for fp32
8. **Integrate into `bench/train.py` once verified** — every kernel that passes correctness AND shows measurable speedup MUST be enabled by default in `bench/train.py` (under the `if not args.no_triton:` block). Users can disable with `--no_triton`. This ensures training always uses the fastest available path.
9. **NEVER use `register_buffer` for tensors that need gradients** — buffers are excluded from autograd. If you need a cached tensor derived from parameters, recompute it from live parameters on every forward (the `torch.cat` cost is negligible vs GEMM).
10. **ALWAYS wrap raw Triton kernel calls in `torch.autograd.Function`** — when a Triton kernel writes into `torch.empty()`, autograd has no record of how output depends on input. The graph is severed. Use `autograd.Function` with: forward = Triton kernel (fast), backward = PyTorch ops (correct).
11. **Run `verify_grads.py` before promoting any kernel** — `python src/kernels/bench/verify_grads.py` checks gradient equivalence, frozen params, and multi-step convergence. A kernel that passes forward correctness but fails gradient verification is BROKEN for training.

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
- `logs/`, `bugs/`, `shaurya_notes.md`, `venv/`, `wandb/`, `research_on_activations/` are all gitignored

---

## When In Doubt

1. Read `src/configuration_bibo.py` for all config params
2. Read `src/modeling/attn/base.py` for attention logic (SDPA + SSMax)
3. Read `src/modeling/ffn/moe.py` for MoE dispatch logic
4. Read `src/modeling/ffn/router.py` for routing logic (logit norm + bias heuristics)
5. Read `docs/ssmax.md` for SSMax theory
6. Read `docs/deprecated.md` for removed components
7. Read `docs/configuration_guide.md` for tuning guidance
8. Read `shaurya_notes.md` for research insights and findings
