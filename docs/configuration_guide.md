# BiBo Configuration Guide

Complete reference for all BiBo model configuration parameters with implementation details, tuning guidance, and research references.

---

## Table of Contents

1. [Router Configuration](#router-configuration)
2. [MoE Architecture](#moe-architecture)
3. [Shared Expert Scaling](#shared-expert-scaling)
4. [RoPE Scaling](#rope-scaling)
5. [Attention Configuration](#attention-configuration)
6. [Additional Router / MoE Parameters](#additional-router--moe-parameters)
7. [Quick Reference Table](#quick-reference-table)

---

## Router Configuration

### Two Independent Mechanisms

The BiBo router uses **two independent mechanisms** that serve different purposes:

#### 1. **Router Logit Normalization** (`router_lambda`) — Controls Confidence

**Purpose:** Increase confidence in expert selection (low entropy in router logits)

**What it does:** Makes the router more or less decisive when picking experts

**Mechanism:** Normalizes and scales logits before softmax
```python
z̃ = λ · (z - μ) / σ
g = softmax(z̃)
```

**Controls:**
- ✓ Entropy of routing distribution (confidence)
- ✓ Sharpness of expert selection (decisiveness)
- ✗ Does NOT control which experts get selected
- ✗ Does NOT control load balancing

**Key insight:** Higher λ → lower entropy → more confident decisions

---

#### 2. **Bias Update Mechanism** (`bias_update_*`) — Controls Load Balancing

**Purpose:** Ensure experts are selected evenly over the long run

**What it does:** Adjusts router bias to favor under-utilized experts

**Mechanism:** Tracks token counts per expert, updates bias periodically
```python
deviation = mean_tokens_per_expert - tokens_per_expert
bias += factor × sign(deviation)
```

**Controls:**
- ✓ Load distribution across experts (fairness)
- ✓ Prevents expert collapse (some experts never used)
- ✗ Does NOT control routing confidence/entropy
- ✗ Does NOT control sharpness of decisions

**Key insight:** Bias updates ensure all experts get used, regardless of routing confidence

---

### How They Work Together

| Mechanism | What it controls | Analogy |
|-----------|------------------|---------|
| `router_lambda` | **HOW CONFIDENT** the router is | "How decisively should I pick?" |
| `bias_update_*` | **WHICH EXPERTS** get selected over time | "Am I picking all experts fairly?" |

**Example scenario:**
- `router_lambda = 2.0` → Router is very confident (low entropy), picks experts decisively
- `bias_update_factor = 1e-2` → If some experts are under-used, bias increases to make them more likely
- **Result:** Confident routing + fair load distribution

**They are independent:**
- You can have high confidence (low entropy) AND balanced load
- You can have low confidence (high entropy) AND imbalanced load
- They solve different problems

---

### `router_temperature`

**Default:** `1.3`

**Location:** `src/configuration_bibo.py:68`

**What it does:**
Temperature parameter for router logits. Controls the "sharpness" or "smoothness" of the expert selection distribution.

**Implementation:**
The temperature is NOT explicitly used in the current implementation. Instead, router logits are normalized using the `router_lambda` parameter (see below). The temperature parameter is kept for backward compatibility but the actual sharpening/desharpening is now handled implicitly through logit normalization.

**Code Reference:**
```python
# src/modeling/ffn/router.py:45-50
# z = lambda * (z - mean) / std (Skywork-MoE normalization)
mean = router_logits.mean(dim=1, keepdim=True)
std = router_logits.std(dim=1, keepdim=True) + 1e-6
router_logits_norm = (router_logits - mean) / std
router_logits_scaled = self.router_lambda * router_logits_norm
```

**How it works:**
- **Higher values** → sharper distribution → more confident expert selection → fewer experts get significant weight
- **Lower values** → smoother distribution → more uniform expert selection → load balancing but less specialization

**Tuning guidance:**
- **Default (1.3):** Good starting point for most cases
- **Increase (1.5-2.0):** If you want stronger expert specialization and have good load balancing
- **Decrease (0.8-1.2):** If experts are too specialized and you're seeing load imbalance

**Pros:**
- Higher temp: Better expert specialization, potentially better performance
- Lower temp: Better load balancing, more stable training

**Cons:**
- Higher temp: Risk of expert collapse (some experts never used)
- Lower temp: Experts may not specialize enough, reduced model capacity

**Research Reference:**
This parameter is conceptually related to temperature in softmax, but the actual implementation follows the Skywork-MoE gating logit normalization approach (see `router_lambda`).

---

### `router_lambda`

**Default:** `1.0`

**Location:** `src/configuration_bibo.py:69`

**What it does:**
Scaling factor for normalized router logits before softmax. Controls **confidence/decisiveness** in expert selection by creating **low-entropy routing distributions**.

**Purpose:** Increase confidence in expert selection (low entropy in router logits)

**Implementation:**
```python
# src/modeling/ffn/router.py:45-50
mean = router_logits.mean(dim=1, keepdim=True)
std = router_logits.std(dim=1, keepdim=True) + 1e-6
router_logits_norm = (router_logits - mean) / std
router_logits_scaled = self.router_lambda * router_logits_norm
routing_weights = F.softmax(router_logits_scaled, dim=1)
```

**How it works:**
1. Router logits are first normalized to zero mean and unit std
2. Then scaled by `router_lambda`
3. Finally passed through softmax

This ensures consistent behavior regardless of the magnitude of raw logits.

**Mathematical formulation (Skywork-MoE, Equation 6):**
```
z̃ = λ · (z - μ) / σ
g = softmax(z̃)
```

Where:
- `z` = raw router logits
- `μ` = mean of logits
- `σ` = standard deviation of logits
- `λ` = router_lambda (scaling factor)
- `z̃` = normalized and scaled logits
- `g` = final routing weights

**What this controls:**
- **Entropy of routing distribution** (confidence in expert selection)
- **Sharpness of expert selection** (how decisively the router picks experts)
- **Separation between expert logits** (forces model to create clear preferences)

**Does NOT control:**
- Load balancing across experts (that's `bias_update_*` parameters)
- Which experts get selected (that's learned through training)

**Tuning guidance:**
- **λ = 1.0:** Standard normalization, balanced routing
- **λ = 1.5-2.0:** Sharper routing, stronger expert specialization, **lower entropy**
- **λ = 0.5-0.8:** Softer routing, **higher entropy**, more exploration

**Pros:**
- Higher λ: **Lower entropy** → clearer expert differentiation, more confident decisions, better specialization
- Lower λ: **Higher entropy** → more uniform expert usage, stable training, more exploration

**Cons:**
- Higher λ: Risk of token dropping, load imbalance (some experts never selected)
- Lower λ: Experts may not learn distinct specializations, mushy routing decisions

**Research Reference:**
[Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models](https://arxiv.org/abs/2406.06563)

Section 4.1 "Gating Logit Normalization" - This technique prevents high-entropy gate distributions and improves expert diversification.

**Key findings from Skywork-MoE:**
- Without normalization: gate outputs become nearly uniform (**high entropy** = no confidence)
- With λ=1: Significant improvement in loss and token drop rate (**lower entropy** = more confidence)
- With λ=2: Even sharper distributions (**very low entropy** = very confident), but λ=1 is sufficient for most cases

**Entropy interpretation:**
- **High entropy** (λ < 1.0): Router is uncertain, spreads probability across many experts
- **Low entropy** (λ > 1.0): Router is confident, concentrates probability on few experts
- **Goal:** Low entropy = confident, decisive expert selection

---

### `router_noise`

**Default:** `0` (disabled)

**Location:** `src/configuration_bibo.py:55`

**What it does:**
Would add Gaussian noise to router logits during training for exploration. **Currently run at 0 and the
noise-injection code in `router.py` is commented out (DEPRECATED, do not remove)** — forward-time
randomness breaks gradient checkpointing's recompute without RNG preservation, and we train with
`router_noise=0`. Kept for compat / future re-enable.

**Implementation (DEPRECATED, commented out in `router.py`):**
```python
# if self.training and self.router_noise > 0:
#     noise_stddev = math.sqrt(self.router_noise)
#     noise = torch.randn_like(router_logits) * noise_stddev
#     router_logits = router_logits + noise.detach()
```

**Tuning guidance:**
- **0 (default):** No noise, deterministic routing (required for gradient checkpointing exactness).
- Re-enabling requires uncommenting the block and preserving RNG state across checkpoint recompute.

---

### `router_type`

**Default:** `"mlp"`

**Options:** `"mlp"` or `"conv"`

**Location:** `src/configuration_bibo.py:68`

**What it does:**
Determines the architecture of the router network.

**Implementation:**

**MLP Router:**
```python
# src/modeling/ffn/router.py:20-21
self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
# Forward: router_logits = self.gate_proj(flat_hidden)
```

**Conv Router:**
```python
# src/modeling/ffn/router.py:23-24
self.gate_conv = nn.Conv1d(config.hidden_size, self.num_routed_experts, 
                           self.kernel_size, padding=0, bias=False)
# Forward: Uses causal convolution with padding
```

**Comparison:**

| Feature | MLP Router | Conv Router |
|---------|-----------|-------------|
| Context | Token-level (no context) | Local context (kernel_size tokens) |
| Parameters | hidden_size × num_experts | hidden_size × num_experts × kernel_size |
| Speed | Faster | Slightly slower |
| Routing | Independent per token | Context-aware |

**Tuning guidance:**
- **MLP:** Default choice, simpler, faster
- **Conv:** Use if you want routing to consider local context (e.g., kernel_size=3 looks at previous 2 tokens)

---

## MoE Architecture

### `bias_update_threshold`

**Default:** `8000`

**Location:** `src/configuration_bibo.py:57`

**What it does:**
Number of tokens (batch_size × seq_len) to process before updating router bias for **load balancing**.

**Purpose:** Ensure experts are selected evenly over the long run (load balancing)

**Implementation:**
```python
# src/modeling/ffn/moe.py:18-20
self.register_buffer("tokens_processed", torch.tensor(0, dtype=torch.long))
self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))

# During forward pass (moe.py:60-68)
batch_tokens = bsz * seq_len
self.tokens_processed += batch_tokens
self.accumulated_tpe += current_tpe.float()

if self.tokens_processed >= self.bias_update_threshold:
    tokens_per_expert = self.accumulated_tpe.clone()
    self.tokens_processed.zero_()
    self.accumulated_tpe.zero_()
```

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

**What this controls:**
- **Load balancing** (ensuring all experts get used evenly)
- **Expert utilization distribution** (prevents expert collapse)
- **Long-term fairness** in expert selection

**Does NOT control:**
- Confidence/entropy of routing decisions (that's `router_lambda`)
- Sharpness of expert selection (that's `router_lambda`)
- Which expert is best for a given token (that's learned through training)

**Key distinction from `router_lambda`:**
- `router_lambda`: Controls **HOW CONFIDENT** the router is (entropy of logits)
- `bias_update_*`: Controls **WHICH EXPERTS** get selected over time (load distribution)

**Analogy:**
- `router_lambda` = "How decisively should I pick?" (confidence)
- `bias_update_*` = "Am I picking all experts fairly?" (fairness)

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

**Default:** `None` → **auto-computed** as a Hill function of `num_routed_experts`
(`(1 - exp(-n/48)) * 0.5`); pass an explicit float (e.g. `1e-2`) to override.

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

**What this controls:**
- **Speed of load balancing** (how fast under-utilized experts get boosted)
- **Magnitude of bias adjustments** (step size per update)

**Does NOT control:**
- Confidence/entropy of routing decisions (that's `router_lambda`)
- Frequency of updates (that's `bias_update_threshold`)

**Tuning guidance:**
- **1e-2 (default):** Moderate adjustment
- **1e-3 to 5e-3:** Conservative, slow adaptation
- **2e-2 to 5e-2:** Aggressive, fast adaptation

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
bias_update_factor = 5e-3

# Balanced (default — factor auto-computed if left as None)
bias_update_threshold = 8_000
bias_update_factor = None   # auto

# Aggressive (fast load balancing)
bias_update_threshold = 4_000
bias_update_factor = 2e-2
```

**Key distinction:**
- This parameter affects **load distribution** (which experts get used)
- It does NOT affect **routing confidence** (how decisively experts are selected)
- For routing confidence, use `router_lambda`

---

## Shared Expert Scaling

### `moe_shared_scaling`

**Default:** `1.0` (auto-computed if left as 1.0)

**Location:** `src/configuration_bibo.py:70`

**What it does:**
Scaling factor for shared expert output in MoE block. Balances the contribution of shared vs. routed experts.

**Implementation:**
```python
# src/modeling/ffn/moe.py:115-116
final_output = final_routed + (getattr(self, 'moe_shared_scaling', 1.0) * shared_combined)
```

**Auto-computation (if moe_shared_scaling == 1.0):**
```python
# src/configuration_bibo.py:130-152
if moe_shared_scaling == 1.0:
    try:
        import numpy as np
        def softmax(x):                  # numerically stable (matches the code)
            e = np.exp(x - x.max())
            return e / e.sum()
        
        n = self.num_routed_experts  # e.g., 16
        k = self.num_experts_per_tok  # e.g., 6
        s = self.num_shared_experts   # e.g., 1
        
        factors = []
        for _ in range(10000):
            logits = np.random.randn(n - s)
            p = np.sort(softmax(logits))[::-1][:k - s]
            factors.append(s**0.5 / (np.sum(p**2)**0.5))
        
        approx_lambda = float(np.mean(factors))
        self.moe_shared_scaling = round(approx_lambda, 2)
```

**Mathematical intuition:**

The formula computes:
```
λ = √s / ||p||₂
```

Where:
- `s` = number of shared experts
- `p` = top-k routing probabilities (sorted, excluding shared experts)
- `||p||₂` = L2 norm of routing probabilities

**Why this formula?**

The goal is to balance the magnitude of contributions:
- **Routed experts:** Output is weighted by routing probabilities `p`, so effective magnitude ~ `||p||₂`
- **Shared experts:** Always active with weight 1, so effective magnitude ~ `√s`
- **Scaling factor:** `λ = √s / ||p||₂` makes both contributions comparable

**Example calculation:**
```python
# Configuration
num_routed_experts = 16
num_experts_per_tok = 6
num_shared_experts = 1

# Simulation (10,000 random routing scenarios)
# Typical result: λ ≈ 0.35 - 0.45

# With s=1: λ ≈ 0.40
# With s=2: λ ≈ 0.57 (√2 ≈ 1.41 times larger)
# With s=4: λ ≈ 0.80 (√4 = 2 times larger)
```

**Tuning guidance:**

**Auto mode (recommended):**
```python
moe_shared_scaling = 1.0  # Triggers auto-computation
```

**Manual tuning:**
- **< 0.5:** Reduce shared expert influence (more specialization)
- **0.5-1.0:** Balanced contribution
- **> 1.0:** Increase shared expert influence (more shared knowledge)

**When to tune manually:**
- If shared expert is learning too slowly → increase scaling
- If routed experts are underutilized → decrease scaling
- If you want to emphasize common knowledge → increase scaling
- If you want to emphasize specialization → decrease scaling

**Pros:**
- Higher scaling: Stronger common knowledge, more stable
- Lower scaling: More expert specialization, higher capacity

**Cons:**
- Higher scaling: Routed experts may be underutilized
- Lower scaling: May lose common knowledge benefits

**Research References:**

1. **DeepSeek-V2/V3:**
   - [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
   - Uses shared experts to capture common knowledge across all tokens
   - Shared experts are always active, routed experts provide specialization

2. **Muon Optimizer Context:**
   - [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/)
   - While Muon is an optimizer, the shared expert scaling concept appears in discussions of MoE training efficiency
   - Proper scaling of shared vs. routed experts is crucial for stable training

**Architecture insight:**

```
Input
  ↓
  ├─→ Shared Expert (always active) ──→ × moe_shared_scaling ──┐
  │                                                              │
  └─→ Router → Top-K Routed Experts → Weighted Sum ────────────┤
                                                                 ↓
                                                            Final Output
```

The shared expert acts as a "baseline" that all tokens pass through, while routed experts provide token-specific specialization.

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
- `"aux_loss"` — Switch-Transformer / Qwen-style auxiliary load-balancing loss (uses `aux_loss_coef`).

### `aux_loss_coef`

**Default:** `0.01` · **Location:** `src/configuration_bibo.py:63`

Coefficient on the auxiliary load-balancing loss. **Only used when `load_balance_strategy="aux_loss"`.**
`1e-2` is the ablated consensus (Switch-T, ST-MoE, OLMoE).

### `use_router_logit_norm`

**Default:** `False` · **Location:** `src/configuration_bibo.py:60`

z-score normalize router logits before softmax (Skywork-MoE style). Note `router_lambda` scaling is
applied on top of this normalization in the router.

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

### Router Mechanisms Summary

| Mechanism | Parameters | Purpose | What it Controls |
|-----------|-----------|---------|------------------|
| **Logit Normalization** | `router_lambda` | Confidence in expert selection | Entropy of routing distribution (decisiveness) |
| **Bias Updates** | `bias_update_threshold`, `bias_update_factor` | Load balancing | Which experts get selected over time (fairness) |

**Key insight:** These are independent mechanisms that solve different problems.

### All Parameters

| Parameter | Default | Purpose | Tuning Range |
|-----------|---------|---------|--------------|
| `router_lambda` | 1.0 | **Routing confidence (entropy control)** | 0.5-2.0 |
| `bias_update_threshold` | 8000 | **Load balancing frequency** (tokens between updates) | 2k-50k |
| `bias_update_factor` | None (auto) | **Load balancing step size** (auto: Hill fn of n) | 1e-3 to 5e-2 |
| `load_balance_strategy` | "bias" | How load is balanced | "none" / "bias" / "aux_loss" |
| `aux_loss_coef` | 0.01 | Aux load-balance loss coef (only if strategy="aux_loss") | 1e-3 to 1e-2 |
| `use_router_logit_norm` | False | z-score normalize logits before softmax (Skywork) | bool |
| `router_activation` | "none" | Activation on raw logits before softmax | "none"/"relu"/"silu" |
| `gate_type` | "sigmoid" | Gating mechanism | "sigmoid" / "softmax" |
| `router_temperature` | 1.3 | Legacy parameter (not actively used) | 0.8-2.0 |
| `router_noise` | 0 | Exploration noise (DEPRECATED, code commented out) | 0 |
| `router_type` | "mlp" | Router architecture | "mlp" or "conv" |
| `kernel_size` | 3 | Conv router kernel size (sees prev kernel_size-1 tokens) | 3-7 (odd) |
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
    router_lambda=0.8,           # Softer routing
    bias_update_threshold=16_000,  # Infrequent updates
    bias_update_factor=5e-3,     # Small steps
    moe_shared_scaling=1.0,      # Auto (or 0.6 for more shared influence)
)
```

### Balanced (Default)
```python
config = BiBoConfig(
    router_lambda=1.0,           # Standard normalization
    bias_update_threshold=8_000,    # Regular updates (default)
    bias_update_factor=None,     # Auto-computed (Hill fn of num_routed_experts)
    moe_shared_scaling=1.0,      # Auto-computed
)
```

### Aggressive (Fast Specialization)
```python
config = BiBoConfig(
    router_lambda=1.5,           # Sharper routing
    bias_update_threshold=4_000,    # Frequent updates
    bias_update_factor=2e-2,     # Large steps
    moe_shared_scaling=1.0,      # Auto (or 0.3 for more specialization)
)
```

> `router_noise` is omitted from these presets — it is DEPRECATED and run at 0 (see its section).

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

**4. Shared vs. Routed Contribution:**
```python
# Compare magnitudes
shared_norm = shared_output.norm()
routed_norm = routed_output.norm()
ratio = shared_norm / routed_norm
# Should be close to 1.0 with auto-scaling
```

### Common Issues and Solutions

**Issue: Expert collapse (some experts never used)**
- **Symptom:** Some experts have near-zero token counts
- **Solution:** 
  - Decrease `router_lambda` (0.5-0.8)
  - Decrease `bias_update_threshold` (~4000 — more frequent rebalancing; default 8000)
  - Increase `bias_update_factor` (2e-2)

**Issue: Poor expert specialization**
- **Symptom:** All experts learn similar representations
- **Solution:**
  - Increase `router_lambda` (1.5-2.0)
  - Decrease `moe_shared_scaling` (0.3-0.5)
  - Use conv router for context-aware routing

**Issue: Training instability**
- **Symptom:** Loss spikes, gradient explosions
- **Solution:**
  - Decrease `router_lambda` (0.8)
  - Increase `bias_update_threshold` (~16000 — gentler rebalancing; default 8000)
  - Decrease `bias_update_factor` (5e-3)

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
- Support for both MLP and Conv routers

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

### Why `gate_proj` / `gate_conv` Are Safe Under L2

The router projection weights (`gate_proj.weight`, `gate_conv.weight`) are standard projection matrices. They map hidden states into logit space. Weight decay shrinks their magnitude, but the Skywork-MoE normalization step:

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
| Learnable `router_lambda` | L2 → λ→0 → post-normalization logits collapse → uniform routing |
| Learnable `router_temperature` | L2 → τ→0 → softmax degenerates |
| Per-expert preference bias (α) | L2 → α→0 → routing preferences erased → uniform selection |
| Any scalar gate on routing logits | Same class — directly controls routing sharpness |

### Optimizer Grouping Template

```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim == 1:  # biases, LayerNorm/RMSNorm weights
        no_decay_params.append(param)
    elif any(k in name for k in ('router_lambda', 'router_temperature', 'routing_alpha')):
        no_decay_params.append(param)  # routing scale/preference — never decay
    else:
        decay_params.append(param)  # standard weight matrices — decay is fine
```

### Current Status (Safe, No Action Needed)

| Parameter | Status |
|-----------|--------|
| `gate_proj.weight` | L2 OK — projection, magnitude decoupled by Skywork norm |
| `gate_conv.weight` | L2 OK — same logic |
| `router.bias` | N/A — `requires_grad=False`, optimizer ignores it |
| `router_lambda` | Currently a config constant, not a parameter |
| `router_temperature` | Currently unused (legacy config field) |

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
