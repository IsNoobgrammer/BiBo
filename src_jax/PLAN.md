# BiBo-JAX — Plan

Port BiBo to JAX/Flax for benching on **Kaggle TPU v5e-8** (8 cores, 128 GB HBM).
Driver: **bigger models** — T4's 16 GB×2 caps param count; 128 GB HBM removes it.
Throughput/MFU is a bonus, not the reason.

**Strategy: EXTEND [EasyDeL](https://github.com/erfanzar/EasyDeL)** (Flax NNX) + its kernel repo
[ejkernel](https://github.com/erfanzar/ejkernel) (Triton GPU / Pallas TPU). Not a fork, not a
from-scratch rewrite. Source-verified: clean extension, no EasyDeL core fork needed.

Why JAX and not torch-xla: our plan centers on a custom fused-PolyGLU kernel. torch-xla's custom
path makes you author the kernel in JAX-Pallas *anyway*, plus a bridge + hand-wired autograd — so
native JAX is less to debug. (No autograd engine differentiates through Pallas either way; backward
is always hand-written.)

---

## Extension-point map

| # | BiBo feature | Verdict | How |
|---|---|---|---|
| 1 | Model registration | **SUBCLASS** | `@register_config("bibo")` + `@register_module(TaskType.CAUSAL_LM, ...)` decorators — that's the whole registration; trainer / sharding / `from_pretrained` resolve off the registry. |
| 2 | Attention + SSMax/XSA/NoPE | **SUBCLASS** | `UnifiedAttention` is fully decomposed (q/k/v proj → rope → performer → o_proj); inject around the kernel. |
| 3 | PolyGLU MoE | **OWN** | The one real build — EasyDeL's `moe_call` assumes homogeneous experts. See below. |
| 4 | RMSNorm + RoPE | **REUSE** | `from easydel.layers.norms import RMSNorm`; `from easydel.layers.rotary import RotaryEmbedding, get_rope`. |
| 5 | Muon optimizer | **REUSE** | EasyDeL ships `EasyDeLOptimizers.MUON` (via `eformer`) with the *same* Moonlight quintic-NS recipe BiBo adopted. |
| 6 | Kernels (ejkernel) | **REUSE** | `ejkernel.modules.grouped_matmul` (GMM) as MoE backend; `jax.custom_vjp` template for the fused PolyGLU kernel later. |

### Detail per feature

**1. Registration** — subclass `EasyDeLBaseConfig` + `BaseCausalLMModule[ModelT, ConfigT]`, decorate.
Registry: `easydel/infra/factory.py` (`_task_registry` keyed by `(TaskType, model_type)`). Implement
only `get_decoder` / `get_lm_head` / `get_embedding`. Sharding is opt-in (`get_partition_rules()` →
`None` triggers auto-sharding).

**2. Attention** (`easydel/layers/attention/_unified.py`; ref `Qwen3MoeAttention`)
- **SSMax** → override `_postprocess_qkv` (Qwen3Moe already overrides it for QK-norm). A per-head
  scalar multiply on q runs *before* RoPE but **commutes** with RoPE's rotation → numerically
  equivalent to BiBo's post-RoPE scaling. One method.
- **XSA** → no pre-o_proj hook; override the `forward()` tail (o_proj is in the attention class, with
  `value_states` in scope). Verify head-merge layout when reshaping `v`.
- **Partial RoPE/NoPE head-split** → ⚠️ EasyDeL's `partial_rotary_factor` splits the *head_dim* axis,
  NOT the heads axis. Override `_apply_rotary`: slice q/k along heads, RoPE the RoPE-heads, pass NoPE
  heads through, concat.

**3. MoE — OWN** (`easydel/layers/moe/_moe_module.py`; ref `Qwen3MoeSparseBlock`)
- `moe_call` uses **stacked weights `[E,H,M]` + one shared `act_fn`** → structurally can't express
  PolyGLU (per-group SiLU/ReLU²/Tanh) or param-free Identity/Zero experts.
- **Reuse**: router slot (gate is a swappable `nn.Module` → `moe_call(gate_layer=...)`, emits
  `[N, num_experts]` → conv router drops in), registration, sharding, `ejkernel.grouped_matmul`.
- **Write**: the expert-computation path. Default = **3 GMMs grouped by activation** (SiLU / ReLU² /
  Tanh) + skip-matmul for Identity/Zero. Fused single-kernel PolyGLU is a later optimization.

**4. Norm + RoPE — REUSE** as-is. Qwen3Moe builds rope via `config.get_basic_rope(...)`.

**5. Muon — REUSE.** `optimizer=EasyDeLOptimizers.MUON`, hyperparams via `extra_optimizer_kwargs`.
BiBo's custom per-expert 3D-slice NS → write an `OptimizerBuilder` + `@register_optimizer("muon_bibo")`
into eformer's registry. ⚠️ `TrainingArguments` has no field for a pre-built optax `tx` — use the
registry seam, not a raw transform.

**6. Kernels — REUSE pattern.** ejkernel backends self-register via
`@kernel_registry.register("algo", Platform.TRITON, Backend.GPU)` / `(PALLAS, TPU)`; `detect_platform()`
dispatches GPU⇒Triton / TPU⇒Pallas / else XLA. Autograd = `jax.custom_vjp` with `_fwd`/`_bwd` +
`.defvjp()`. Mirror this for the fused PolyGLU op.

---

## Build order

1. **Skeleton** — `BiBoConfig` + `BiBoForCausalLM` with the decorators. Smoke-test registration +
   a forward pass on CPU/GPU. Self-verify: `from_pretrained`/auto-class resolves "bibo", forward runs.
2. **Reuse wiring** — RMSNorm, RoPE, Muon imported and wired; no custom code.
3. **`BiBoAttention(UnifiedAttention)`** — SSMax (`_postprocess_qkv`), NoPE split (`_apply_rotary`),
   XSA (`forward()` tail). Grad-verify each vs the CUDA-BiBo reference.
4. **Conv router** module → `moe_call(gate_layer=...)`.
5. **BiBo MoE module** (the real work) — PolyGLU experts on `grouped_matmul`, 3-GMM-by-activation.
6. **SPMD mesh** for v5e-8 (decide DP/FSDP/EP layout); grad-verify full model vs CUDA-BiBo; then the
   fused-PolyGLU Pallas kernel via ejkernel's `custom_vjp` template.
7. **Qwen baseline** on the same stack (EasyDeL's Qwen3MoE) — both models on one stack or the loss
   comparison is confounded.

## Open setup questions (decide before step 1)
- Separate venv for JAX (`jax[cuda]` locally vs `jax[tpu]` on Kaggle) — likely yes, to avoid clashing
  with the PyTorch CUDA `.venv`.
- EasyDeL version pin (it moves fast — pin a known-good release).
- SPMD axis layout for v5e-8 (DP vs FSDP vs EP for the experts).
- Parity target: which CUDA-BiBo numerics are the grad-correctness ground truth (bf16 on TPU vs fp16
  on CUDA — expect tolerance, not bit-equality).

## Status
Scoping. Nothing implemented beyond this dir.
