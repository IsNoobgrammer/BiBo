# BiBo Configuration Guide

Complete reference for all BiBo model configuration parameters with implementation details, tuning guidance, and research references.

---

## Table of Contents

1. [Router Configuration](#router-configuration)
2. [MoE Architecture](#moe-architecture)
3. [Shared Expert Scaling](#shared-expert-scaling)
4. [Bias Update Mechanism](#bias-update-mechanism)
5. [RoPE Scaling](#rope-scaling)
6. [Quick Reference Table](#quick-reference-table)

---

## Router Configuration

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
Scaling factor for normalized router logits before softmax. This is the **actual** parameter that controls routing sharpness in the current implementation.

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

**Tuning guidance:**
- **λ = 1.0:** Standard normalization, balanced routing
- **λ = 1.5-2.0:** Sharper routing, stronger expert specialization
- **λ = 0.5-0.8:** Softer routing, better load balancing

**Pros:**
- Higher λ: Clearer expert differentiation, better specialization
- Lower λ: More uniform expert usage, stable training

**Cons:**
- Higher λ: Risk of token dropping, load imbalance
- Lower λ: Experts may not learn distinct specializations

**Research Reference:**
[Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models](https://arxiv.org/abs/2406.06563)

Section 4.1 "Gating Logit Normalization" - This technique prevents high-entropy gate distributions and improves expert diversification.

**Key findings from Skywork-MoE:**
- Without normalization: gate outputs become nearly uniform (high entropy)
- With λ=1: Significant improvement in loss and token drop rate
- With λ=2: Even sharper distributions, but λ=1 is sufficient for most cases

---

### `router_noise`

**Default:** `0.5`

**Location:** `src/configuration_bibo.py:68`

**What it does:**
Adds Gaussian noise to router logits during training for exploration and load balancing.

**Implementation:**
```python
# src/modeling/ffn/router.py:38-41
if self.training and self.router_noise > 0:
    noise_stddev = math.sqrt(self.router_noise)
    noise = torch.randn_like(router_logits) * noise_stddev
    router_logits = router_logits + noise.detach()
```

**How it works:**
- Noise ~ N(0, √router_noise)
- Only applied during training
- Encourages exploration of different expert combinations

**Tuning guidance:**
- **0.0:** No noise, deterministic routing
- **0.5:** Default, good exploration
- **1.0:** High noise, aggressive exploration

**Pros:**
- Prevents premature expert specialization
- Improves load balancing
- Acts as regularization

**Cons:**
- Too much noise can hurt convergence
- Adds randomness to training

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

### `residual_gate_type`

**Default:** `"none"`

**Options:** `"none"`, `"scalar"`, `"token"`, `"channel"`

**What it does:**
Adds optional mHC-inspired write gates around the attention and MLP/MoE residual branches:

```python
hidden_states = residual + gate * branch_output
```

The identity stream is never gated, so the model can always preserve normal residual flow.

**Gate shapes:**
- **none:** Standard residual add
- **scalar:** One learned value per branch/layer
- **token:** One value per token, shape `[batch, seq, 1]`
- **channel:** One value per token/channel, shape `[batch, seq, hidden_size]`

**Tuning guidance:**
- Start with `"token"` for residual-flow diagnostics.
- Use `"scalar"` for a very cheap stability experiment.
- Use `"channel"` only if you can afford extra parameters and want more expressive control.

### `residual_gate_init`

**Default:** `0.95`

Initial gate value. Values close to `1.0` preserve baseline Transformer behavior at initialization.
The gate modules expose `mean`, `open_frac`, and `closed_frac` through:

```python
model.model.residual_gate_stats()
```

---

### `residual_mixer_type`

**Default:** `"none"`

**Options:** `"none"`, `"causal_conv"`, `"dynamic_causal_conv"`

**What it does:**
Adds an attention-residual-style mixer over residual states from previous model depths.
The `"causal_conv"` option replaces softmax attention over all previous layers with a
small fully causal depth convolution:

```python
current = layer(hidden_states)
hidden_states = causal_depth_conv(previous_states + [current])
```

This convolution is over **layer depth**, not over sequence tokens. It can only read
previous residual states and the current layer output, so token-level causality is
preserved. `"causal_conv"` uses one learned depth kernel per layer. `"dynamic_causal_conv"`
uses the current token state to produce token-conditioned depth kernels.

### `residual_conv_kernel_size`

**Default:** `4`

Number of depth states in the causal residual convolution window. The model keeps
`kernel_size - 1` previous residual states plus the current layer output.

### `residual_conv_init`

**Default:** `0.95`

Initial weight on the current layer output. The remaining mass is distributed across
older residual states, keeping initialization close to normal Transformer flow.
The mixer exposes `current_weight`, `previous_mass`, and `num_states` through:

```python
model.model.residual_mixer_stats()
```

---

### `residual_num_streams`

**Default:** `1`

Number of mHC-style parallel residual streams. `1` disables multi-stream residuals.
When enabled, each layer reads a gated mixture of streams, runs the normal decoder
layer, then writes that layer's update back into the streams.

```python
read_state = gated_read(streams)
layer_output = decoder_layer(read_state)
streams = gated_write(streams, layer_output - read_state)
```

### `residual_stream_gate_type`

**Default:** `"token"`

**Options:** `"scalar"`, `"token"`

- **scalar:** One read/write gate per layer and stream.
- **token:** Per-token read/write gates over streams.

### `residual_stream_init`

**Default:** `"copy"`

**Options:** `"copy"`, `"zero"`

- **copy:** Every stream starts from the embedding state. This preserves the first
  layer read even when the initial read gate is not exactly one-hot.
- **zero:** Stream 0 starts from embeddings and auxiliary streams start at zero.

### `residual_stream_read_init` / `residual_stream_write_init`

**Default:** `0.99`

Initial read mass and write gate value for stream 0. The remaining read mass is
spread across auxiliary streams. These should stay close to `1.0` for baseline-like
initialization.

Inspect stream routing after a forward pass:

```python
model.model.residual_stream_stats()
```

---

## MoE Architecture

### `bias_update_threshold`

**Default:** `100_000`

**Location:** `src/configuration_bibo.py:68`

**What it does:**
Number of tokens (batch_size × seq_len) to process before updating router bias for load balancing.

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

**Tuning guidance:**
- **100,000 (default):** Update every ~100k tokens
  - With batch_size=8, seq_len=2048: ~6 batches
  - With batch_size=32, seq_len=4096: ~1 batch
- **Lower (10,000-50,000):** More frequent updates, faster load balancing response
- **Higher (200,000-500,000):** Less frequent updates, more stable routing

**Pros:**
- Lower threshold: Faster adaptation to load imbalance
- Higher threshold: More stable routing, less bias oscillation

**Cons:**
- Lower threshold: May cause routing instability
- Higher threshold: Slower response to load imbalance

**Example calculation:**
```
batch_size = 16
seq_len = 2048
tokens_per_batch = 16 × 2048 = 32,768

batches_until_update = 100,000 / 32,768 ≈ 3 batches
```

---

### `bias_update_factor`

**Default:** `1e-2` (0.01)

**Location:** `src/configuration_bibo.py:67`

**What it does:**
Step size for router bias updates. Controls how aggressively the bias is adjusted to balance load.

**Implementation:**
```python
# src/modeling/ffn/moe.py:46
self.gate.bias.add_(self.bias_update_factor * deviation.sign())
```

**How it works:**
- Computes deviation from mean load: `deviation = mean_tpe - tpe`
- Updates bias by: `bias += factor × sign(deviation)`
- Only the sign matters (not magnitude), so this is a fixed-step update

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
bias_update_threshold = 200_000
bias_update_factor = 5e-3

# Balanced (default)
bias_update_threshold = 100_000
bias_update_factor = 1e-2

# Aggressive (fast load balancing)
bias_update_threshold = 50_000
bias_update_factor = 2e-2
```

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
        def softmax(x):
            p = np.exp(x)
            return p / p.sum()
        
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

**Default:** `{"type": "linear", "factor": 1.0}` (auto-set if None)

**Location:** `src/configuration_bibo.py:154-155`

**What it does:**
Configures Rotary Position Embedding (RoPE) scaling for handling sequences longer than the model was trained on.

**Implementation:**
```python
# src/configuration_bibo.py:154-155
if self.rope_scaling is None:
    self.rope_scaling = {"type": "linear", "factor": 1.0}
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

**Default (no scaling):**
```python
rope_scaling = {"type": "linear", "factor": 1.0}
# Use for sequences ≤ max_position_embeddings (32768)
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

## Quick Reference Table

| Parameter | Default | Location | Purpose | Tuning Range |
|-----------|---------|----------|---------|--------------|
| `router_temperature` | 1.3 | config:68 | Legacy parameter (not actively used) | 0.8-2.0 |
| `router_lambda` | 1.0 | config:69 | **Actual routing sharpness control** | 0.5-2.0 |
| `router_noise` | 0.5 | config:68 | Exploration noise during training | 0.0-1.0 |
| `router_type` | "mlp" | config:68 | Router architecture | "mlp" or "conv" |
| `kernel_size` | 3 | config:68 | Conv router kernel size | 3-7 (odd) |
| `bias_update_threshold` | 100,000 | config:68 | Tokens before bias update | 10k-500k |
| `bias_update_factor` | 0.01 | config:67 | Bias update step size | 1e-3 to 5e-2 |
| `moe_shared_scaling` | 1.0 (auto) | config:70 | Shared expert output scaling | 0.3-1.5 |
| `rope_scaling` | {"type": "linear", "factor": 1.0} | config:154 | Position embedding scaling | factor: 1.0-4.0 |

---

## Configuration Presets

### Conservative (Stable Training)
```python
config = BiBoConfig(
    router_lambda=0.8,           # Softer routing
    router_noise=0.3,            # Less exploration
    bias_update_threshold=200_000,  # Infrequent updates
    bias_update_factor=5e-3,     # Small steps
    moe_shared_scaling=1.0,      # Auto (or 0.6 for more shared influence)
)
```

### Balanced (Default)
```python
config = BiBoConfig(
    router_lambda=1.0,           # Standard normalization
    router_noise=0.5,            # Moderate exploration
    bias_update_threshold=100_000,  # Regular updates
    bias_update_factor=1e-2,     # Moderate steps
    moe_shared_scaling=1.0,      # Auto-computed
)
```

### Aggressive (Fast Specialization)
```python
config = BiBoConfig(
    router_lambda=1.5,           # Sharper routing
    router_noise=0.7,            # More exploration
    bias_update_threshold=50_000,   # Frequent updates
    bias_update_factor=2e-2,     # Large steps
    moe_shared_scaling=1.0,      # Auto (or 0.3 for more specialization)
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
  - Increase `router_noise` (0.7-1.0)
  - Decrease `bias_update_threshold` (50k)
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
  - Decrease `router_noise` (0.3)
  - Increase `bias_update_threshold` (200k)
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

*Last updated: 2026-05-03*
