# BiBo Configuration Guide

Complete reference for all BiBo model configuration parameters with implementation details, tuning guidance, and research references.

---

## Table of Contents

1. [Router Configuration](#router-configuration)
2. [MoE Architecture](#moe-architecture)
3. [RoPE Scaling](#rope-scaling)
4. [Attention Configuration](#attention-configuration)
5. [Additional Router / MoE Parameters](#additional-router--moe-parameters)
6. [Quick Reference Table](#quick-reference-table)

---

## Router Configuration

### Load Balancing via Bias Updates

BiBo balances expert load with aux-loss-free **bias updates** (DeepSeek-V3 / MiMo-V2.5). The bias
shifts *which* experts get selected without touching the combine weights.

> Historical note: BiBo also had a Skywork-style **logit-normalization** mechanism (`router_lambda`,
> `use_router_logit_norm`) that controlled routing *confidence* independently of load balancing.
> Both were **removed Jun 28 2026** — the router is pure MiMo now, whose only normalization is
> `norm_topk_prob`. Bias updates are the sole balancing mechanism.

#### **Bias Update Mechanism** (`bias_update_*`)

**Purpose:** Ensure experts are selected evenly over the long run

**What it does:** Adjusts router bias to favor under-utilized experts

**Mechanism:** Tracks token counts per expert, updates bias periodically
```python
deviation = mean_tokens_per_expert - tokens_per_expert
bias += factor × sign(deviation)
```

**Controls:** load distribution across experts, and prevents expert collapse. It does **not** touch
the combine weights — the bias affects selection only.

---

### ~~`router_type`~~ — REMOVED (Jul 26 2026)

The conv-router option and the `router_type` config key are **gone**. BiBo uses the MLP router
exclusively: the conv router never outperformed it. Old configs passing `router_type=` log a warning
and have it dropped — it is not stored on the config and not written back to `config.json`.

The router is one linear projection, with **experts as the row dimension**:

```python
# src/modeling/ffn/router.py
self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
# gate_proj.weight.shape == (num_routed_experts, hidden_size)
# Forward: router_logits = self.gate_proj(flat_hidden).float()
```

`kernel_size` still exists but now serves **only** the conv shared expert
(`shared_expert_type="conv"`), which is unaffected by this removal.

---

## MoE Architecture

### `bias_update_threshold`

**Default:** `8000`

**Location:** `src/configuration_bibo.py:57`

**What it does:**
Number of tokens (batch_size × seq_len) to process before updating router bias for **load balancing**.

**Purpose:** Ensure experts are selected evenly over the long run (load balancing)

**Implementation** (`src/modeling/ffn/moe.py::_balance_step`):
```python
self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
self._fwd_step = 0        # host-side int: no CUDA buffer, no per-step .item() sync
self._update_every = None # derived once from bias_update_threshold / tokens_per_forward

self.accumulated_tpe += current_tpe.float()
self._fwd_step += 1
if self._fwd_step % self._update_every == 0:
    dist.all_reduce(self.accumulated_tpe, op=dist.ReduceOp.SUM)   # if distributed
    tokens_per_expert = self.accumulated_tpe.clone()
    self.accumulated_tpe.zero_()
```
The trigger counts **forward steps**, not device tokens, so every DDP rank fires the update on the
same step and the `all_reduce` can never desync. (A `tokens_processed` buffer did this until
Jul 1 2026; it forced a per-step host sync.)

**How it works:**
1. Accumulate token counts per expert across batches
2. When threshold is reached, compute load imbalance
3. Update router bias to favor under-utilized experts
4. Reset counters

**Bias Update Logic:**
```python
# src/modeling/ffn/moe.py:35-47
def update_bias(self, tokens_per_expert: torch.Tensor):
    tpe = tokens_per_expert.detach().float()
    mean_tpe = tpe.mean()
    deviation = mean_tpe - tpe  # Positive if expert is under-utilized
    
    # bias += factor * sign(deviation)
    # bias ↑ if deviation > 0 (expert under-utilized)
    # bias ↓ if deviation < 0 (expert over-utilized)
    self.gate.bias.add_(self.bias_update_factor * deviation.sign())
```

**What this controls:** load balancing / expert utilization / long-term selection fairness. It does
**not** decide which expert is best for a given token — that is learned.

**Tuning guidance:**
- **8000 (default):** Update every ~8k tokens
  - With batch_size=8, seq_len=2048 (16,384 tok/batch): about every batch
  - With batch_size=2, seq_len=2048 (4,096 tok/batch): ~2 batches
- **Lower (2,000-4,000):** More frequent updates, faster load balancing response
- **Higher (16,000-50,000):** Less frequent updates, more stable routing

**Pros:**
- Lower threshold: Faster adaptation to load imbalance
- Higher threshold: More stable routing, less bias oscillation

**Cons:**
- Lower threshold: May cause routing instability
- Higher threshold: Slower response to load imbalance

**Example calculation:**
```
batch_size = 2
seq_len = 2048
tokens_per_batch = 2 × 2048 = 4,096

batches_until_update = 8,000 / 4,096 ≈ 2 batches
```

---

### `bias_update_factor`

**Default:** `0.001` — a FIXED small step, deliberately independent of `num_routed_experts`.

> An auto-computed Hill function (0.07 at n=8 growing to 0.35) was removed Jul 26 2026: it was
> backwards. With independent per-expert scoring (`sigmoid`/`situ`) the score distribution does not
> move with `n`, only the order statistics get denser, so the bias distance needed to flip a top-k
> selection *shrinks* as experts are added — measured mean gap at the k|k+1 boundary: **0.041 (n=8)
> → 0.016 (n=32) → 0.0064 (n=128) → 0.0045 (n=512)** for sigmoid.
>
> **Why it must stay small:** `sign()` never returns 0, so the bias dithers ±`u` forever and never
> settles. `u` *is* the balancer's steady-state routing-noise floor. If `u` exceeds the boundary gap
> it reshuffles top-k selections on its own even at perfect balance. At n=128 that ratio is ~5× for
> `u=0.03` but ~0.16× for `u=0.001`. 0.001 also matches DeepSeek-V3. Raise it for faster response at
> the cost of routing jitter; `0` disables balancing entirely.

**Location:** `src/configuration_bibo.py:56`

**What it does:**
Step size for router bias updates. Controls how aggressively the bias is adjusted to **balance load across experts**.

**Purpose:** Control the speed of load balancing (works with `bias_update_threshold`)

**Implementation:**
```python
# src/modeling/ffn/moe.py:46
self.gate.bias.add_(self.bias_update_factor * deviation.sign())
```

**How it works:**
- Computes deviation from mean load: `deviation = mean_tpe - tpe`
- Updates bias by: `bias += factor × sign(deviation)`
- Only the sign matters (not magnitude), so this is a fixed-step update

**What this controls:** speed of load balancing and the step size per update. Update *frequency* is
`bias_update_threshold`.

**Tuning guidance:**
- **1e-3 (default):** dither stays below the top-k boundary gap, so the bias settles
- **1e-4 to 5e-4:** very conservative
- **5e-3+:** faster response, but the dither starts competing with the boundary gap

**Pros:**
- Higher factor: Faster load balancing
- Lower factor: More stable routing

**Cons:**
- Higher factor: Risk of bias oscillation
- Lower factor: Slow to correct imbalance

**Interaction with threshold:**
These two parameters work together:
- **High threshold + High factor:** Infrequent but large corrections
- **Low threshold + Low factor:** Frequent but small corrections
- **Low threshold + High factor:** Frequent and large corrections (may be unstable)
- **High threshold + Low factor:** Infrequent and small corrections (may be too slow)

**Recommended combinations:**
```python
# Conservative (stable training)
bias_update_threshold = 16_000
bias_update_factor = 5e-4

# Balanced (defaults)
bias_update_threshold = 8_000
bias_update_factor = 0.001  # default

# Aggressive (fast load balancing)
bias_update_threshold = 4_000
bias_update_factor = 5e-3
```

**Key distinction:**
- This parameter affects **load distribution** (which experts get used)
- It does NOT affect **routing confidence** (how decisively experts are selected)
- For routing confidence, use `router_lambda`

---

## RoPE Scaling

### `rope_scaling`

**Default:** `{"type": "dynamic", "factor": 1.0}` (auto-set if None — dynamic NTK-aware, identity within the trained window)

**Location:** `src/configuration_bibo.py:208-209`

**What it does:**
Configures Rotary Position Embedding (RoPE) scaling for handling sequences longer than the model was trained on.

**Implementation:**
```python
# src/configuration_bibo.py:208-209
if self.rope_scaling is None:
    self.rope_scaling = {"type": "dynamic", "factor": 1.0}
```

**How it works:**

RoPE encodes position information by rotating embeddings. When extending to longer sequences, scaling prevents position embeddings from becoming too large.

**Scaling types:**

1. **Linear scaling:**
   ```python
   rope_scaling = {"type": "linear", "factor": 2.0}
   # Effective position = actual_position / factor
   # Allows 2× longer sequences
   ```

2. **Dynamic scaling (NTK-aware):**
   ```python
   rope_scaling = {"type": "dynamic", "factor": 2.0}
   # Adjusts base frequency dynamically
   # Better preservation of local relationships
   ```

**Tuning guidance:**

**Default (dynamic NTK, identity within trained window):**
```python
rope_scaling = {"type": "dynamic", "factor": 1.0}
# Identity for sequences ≤ max_position_embeddings (32768); smooth base growth beyond.
# Set {"type": "none"} for plain RoPE.
```

**Extending context (2× longer):**
```python
rope_scaling = {"type": "linear", "factor": 2.0}
# Allows sequences up to 65536 tokens
```

**Extending context (4× longer):**
```python
rope_scaling = {"type": "linear", "factor": 4.0}
# Allows sequences up to 131072 tokens
```

**Dynamic scaling (better quality):**
```python
rope_scaling = {"type": "dynamic", "factor": 2.0}
# Better for long-range dependencies
```

**Pros:**
- Linear: Simple, predictable
- Dynamic: Better quality for long sequences

**Cons:**
- Linear: May degrade quality at very long sequences
- Dynamic: More complex, requires tuning

**When to use:**
- **factor = 1.0:** Training or inference at native context length
- **factor > 1.0:** Inference on longer sequences than training
- **Dynamic:** When quality matters more than simplicity

---

## Attention Configuration

### `use_xsa`

**Default:** `True` · **Location:** `src/configuration_bibo.py:29`

Enables **Exclusive Self Attention** — a parameter-free step that rejects each token's attention
output from its own value vector: `z = y − (y·v̂)v̂` (applied after value-aggregation, before
`o_proj`). Forces the output to be orthogonal to the self-value direction. Full details, the GQA
in-kernel broadcast, and the fused Triton kernel: **[docs/xsa.md](xsa.md)**.

### `use_ssmax`

**Default:** `True` · **Location:** `src/configuration_bibo.py:31`

Enables **SSMax** (scalable-softmax) learnable per-head query scaling (`scale · log(kv_len)`) to
prevent attention fading at long context. Details: **[docs/ssmax.md](ssmax.md)**.

---

## Additional Router / MoE Parameters

### `load_balance_strategy`

**Default:** `"bias"` · **Location:** `src/configuration_bibo.py:62`

How load balancing is enforced across experts:
- `"none"` — no balancing.
- `"bias"` — heuristic router-bias updates (see `bias_update_*`). The BiBo default.
- `"none"` — no balancing.

Only those two are valid; the config rejects anything else. (An `"aux_loss"` strategy with an
`aux_loss_coef` was documented here but never existed in BiBo — the Qwen baseline has its own
`router_aux_loss_coef`.)

### `router_activation`

**Default:** `"none"` · **Location:** `src/configuration_bibo.py:66`

Activation applied to raw router logits before softmax/selection: `"none"` (standard softmax),
`"relu"` (DECO-style), or `"silu"`.

### `gate_type`

**Default:** `"sigmoid"` · **Location:** `src/configuration_bibo.py:71`

Gating mechanism: `"sigmoid"` (DeepSeek-V3, independent per-expert gates) or `"softmax"` (legacy,
competitive across experts).

### `use_shared_expert`

**Default:** `False` · **Location:** `src/configuration_bibo.py:44`

Whether the always-on shared expert is enabled. Off by default to **match Qwen3MoE** (no shared
expert). `shared_expert_type` (`"mlp"` SwiGLU / `"conv"` CausalConv1D) and `moe_shared_scaling`
only take effect when this is `True`.

---

## Quick Reference Table

### All Parameters

| Parameter | Default | Purpose | Tuning Range |
|-----------|---------|---------|--------------|
| `bias_update_threshold` | 8000 | **Load balancing frequency** (tokens between updates) | 2k-50k |
| `bias_update_factor` | 0.001 | **Load balancing step size** (fixed, not a fn of n) | 1e-4 to 1e-2 |
| `load_balance_strategy` | "bias" | How load is balanced | "none" / "bias" |
| `router_activation` | "none" | Activation on the logits, before the gate | "none"/"relu"/"silu" |
| `gate_type` | "sigmoid" | Gating mechanism | "sigmoid" / "situ" / "softmax" |
| `norm_topk_prob` | True | Softmax the gathered top-k weights to sum to 1 | bool |
| `routed_scaling_factor` | 1.0 | Post-norm routed-weight scale | 1.0 = no-op |
| `kernel_size` | 3 | Conv SHARED-EXPERT kernel size (conv router removed Jul 26 2026) | 3-7 (odd) |
| `moe_shared_scaling` | 1.0 (auto) | Shared expert output scaling | 0.3-1.5 |
| `use_shared_expert` | False | Enable the always-on shared expert (off = match Qwen3MoE) | bool |
| `shared_expert_type` | "mlp" | Shared expert kind (only if `use_shared_expert`) | "mlp" / "conv" |
| `mlp_only_layers` | [0, N-1] | Layers using dense MLP instead of MoE (first + last) | list of layer indices |
| `use_xsa` | True | Exclusive Self Attention rejection (see `docs/xsa.md`) | bool |
| `use_ssmax` | True | SSMax scalable-softmax query scaling (see `docs/ssmax.md`) | bool |
| `rope_scaling` | {"type": "dynamic", "factor": 1.0} | Position embedding scaling (NTK-aware) | factor: 1.0-4.0 |

---

## Configuration Presets

### Conservative (Stable Training)
```python
config = BiBoConfig(
    bias_update_threshold=16_000,  # Infrequent updates
    bias_update_factor=5e-4,     # Small steps
)
```

### Balanced (Default)
```python
config = BiBoConfig(
    bias_update_threshold=8_000,    # Regular updates (default)
    bias_update_factor=0.001,    # default fixed step
)
```

### Aggressive (Fast Specialization)
```python
config = BiBoConfig(
    bias_update_threshold=4_000,    # Frequent updates
    bias_update_factor=5e-3,     # Large steps
)
```

### Long Context (Extended Sequences)
```python
config = BiBoConfig(
    max_position_embeddings=32768,
    rope_scaling={"type": "dynamic", "factor": 2.0},  # 2× context
    # ... other params as needed
)
```

---

## Monitoring and Debugging

### Key Metrics to Track

**1. Expert Load Balance:**
```python
# During training, monitor:
tokens_per_expert = torch.bincount(expert_indices)
load_balance = tokens_per_expert.std() / tokens_per_expert.mean()
# Lower is better (< 0.3 is good)
```

**2. Router Entropy:**
```python
# Higher entropy = more uniform routing
entropy = -(routing_weights * torch.log(routing_weights + 1e-10)).sum(dim=-1).mean()
# Target: 1.5-2.5 for good balance
```

**3. Bias Magnitude:**
```python
# Monitor router bias growth
bias_magnitude = model.moe_layer.gate.bias.abs().mean()
# Should stabilize after initial training
```


### Common Issues and Solutions

**Issue: Expert collapse (some experts never used)**
- **Symptom:** Some experts have near-zero token counts
- **Solution:** 
  - Decrease `router_lambda` (0.5-0.8)
  - Decrease `bias_update_threshold` (~4000 — more frequent rebalancing; default 8000)
  - Increase `bias_update_factor` (5e-3)

**Issue: Poor expert specialization**
- **Symptom:** All experts learn similar representations
- **Solution:**
  - Increase `router_lambda` (1.5-2.0)
  - Decrease `moe_shared_scaling` (0.3-0.5)

**Issue: Training instability**
- **Symptom:** Loss spikes, gradient explosions
- **Solution:**
  - Decrease `router_lambda` (0.8)
  - Increase `bias_update_threshold` (~16000 — gentler rebalancing; default 8000)
  - Decrease `bias_update_factor` (5e-4)

**Issue: Shared expert dominates**
- **Symptom:** Routed experts contribute little
- **Solution:**
  - Decrease `moe_shared_scaling` (0.3-0.5)
  - Increase `router_lambda` (1.5)

---

## Research References

1. **Skywork-MoE** (Gating Logit Normalization)
   - Paper: [arxiv.org/abs/2406.06563](https://arxiv.org/abs/2406.06563)
   - Key contribution: `router_lambda` normalization technique
   - Section 4.1: Prevents high-entropy routing distributions

2. **DeepSeek-V2/V3** (Shared Expert Architecture)
   - Paper: [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
   - Key contribution: Shared experts for common knowledge
   - Architecture: Always-active shared experts + routed specialists

3. **Switch Transformer** (MoE Fundamentals)
   - Paper: [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)
   - Key contribution: Top-k routing, auxiliary loss for load balancing

4. **Muon Optimizer** (Training Efficiency)
   - Blog: [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
   - Relevant for: Efficient MoE training, scaling considerations

---

## Code Navigation

**Configuration:**
- Main config: `src/configuration_bibo.py`
- Auto-scaling logic: Lines 130-152

**Router Implementation:**
- Router class: `src/modeling/ffn/router.py`
- Logit normalization: Lines 45-50
- Noise injection: Lines 38-41

**MoE Layer:**
- MoE class: `src/modeling/ffn/moe.py`
- Bias update: Lines 35-47
- Token counting: Lines 60-68
- Shared scaling: Line 115

**Experts:**
- Expert implementations: `src/modeling/ffn/experts.py`
- Shared expert (Conv1D): `BiBoCausalConv1D`

---

## Changelog

**Version 1.0** (Current)
- Auto-computation of `moe_shared_scaling`
- Skywork-MoE style logit normalization via `router_lambda`
- Threshold-based bias updates
- MLP router only (the Conv router variant was removed Jul 26 2026)

---

## Contributing

When adding new configuration parameters:
1. Add to `BiBoConfig.__init__()` with default value
2. Document in this guide with implementation details
3. Add validation in `BiBoConfig.__init__()` validation section
4. Update Quick Reference Table
5. Add monitoring metrics if applicable

---

## Weight Decay Policy for Routing Parameters

> **Lesson learned from:** PolyGLU RED-0001 (danielxmed/PolyGLU) — L2 regularization on a routing preference parameter silently suppressed specialization for 10,000 training steps.

### The Rule

**Never apply L2 weight decay to parameters that directly encode routing preferences, scales, or temperatures.**

### Why `gate_proj` Is Safe Under L2

> ⚠️ **This subsection is stale beyond the conv-router removal.** It reasons from the Skywork-MoE
> logit-normalization and `router_lambda`, both of which were **removed Jun 28 2026** (the router is
> pure MiMo now). The L2 conclusion still holds — `gate_proj` is an ordinary projection — but the
> justification below no longer describes the code.

The router projection weight (`gate_proj.weight`) is a standard projection matrix. It maps hidden states into logit space. Weight decay shrinks its magnitude, but the (now-removed) Skywork-MoE normalization step:

```python
z̃ = λ · (z - μ) / σ
```

...removes all magnitude information via z-score normalization before `router_lambda` re-scales. So L2 on these weights:
- ✓ Regularizes the projection (good for generalization)
- ✓ Keeps gradients well-behaved
- ✗ Does NOT affect routing confidence (that's `router_lambda`)
- ✗ Does NOT push routing toward uniform

### What Must NEVER Have L2

If any of these are added as learnable parameters in the future:

| Parameter | Why L2 Kills It |
|-----------|-----------------|
| Learnable `routed_scaling_factor` | L2 → scale→0 → the routed branch is silenced |
| Per-expert scale vector (`expert_scale`) | L2 → →0 → per-expert preference erased |
| Per-expert preference bias (α) | L2 → α→0 → routing preferences erased → uniform selection |
| Any scalar gate on routing logits | Same class — directly controls routing sharpness |

### Optimizer Grouping Template

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim == 1:  # biases, LayerNorm/RMSNorm weights
        no_decay_params.append(param)
    elif any(k in name for k in ('router_scale', 'expert_scale', 'routing_alpha')):
        no_decay_params.append(param)  # routing scale/preference — never decay
    else:
        decay_params.append(param)  # standard weight matrices — decay is fine
```

### Current Status (Safe, No Action Needed)

| Parameter | Status |
|-----------|--------|
| `gate_proj.weight` | L2 OK — ordinary projection |
| `router.bias` | N/A — `requires_grad=False`, optimizer ignores it |

---

## Benchmarking & Config Editing

For commands to run the benchmark suite (smoke tests, throughput, full training), editing BiBoConfig with `%%writefile` on Kaggle/Jupyter, and param sweeps, see:

**[Benchmarking Guide](benchmarking.md)**

Quick links:
- [Quick Smoke Test](benchmarking.md#quick-smoke-test)
- [Running the Benchmark](benchmarking.md#running-the-benchmark)
- [Editing BiBoConfig with %%writefile](benchmarking.md#editing-biboconfig-with-writefile)
- [Throughput & Memory Benchmarks](benchmarking.md#throughput--memory-benchmarks)
- [Full Training Run (Kaggle 2×T4)](benchmarking.md#full-training-run-kaggle-2t4)

---

*Last updated: 2026-06-25*
