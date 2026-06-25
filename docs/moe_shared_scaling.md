# MoE Shared Expert Scaling — How It's Calculated

## The Problem

In BiBo's MoE architecture, the final output of each MoE layer is:

```
output = routed_expert_output + λ × shared_expert_output
```

Where `λ` = `moe_shared_scaling`. If `λ = 1.0`, the shared expert's contribution may dominate or be dwarfed by the routed experts depending on how routing weights distribute probability mass. We need `λ` to balance their magnitudes.

---

## The Formula

```
λ = √s / ‖p‖₂
```

Where:
- `s` = number of shared experts (typically 1)
- `p` = top-k routing probability vector (after softmax, sorted descending, excluding shared slots)
- `‖p‖₂` = L2 norm of the routing probabilities

---

## Intuition

**Routed experts**: Each token's output is a weighted sum of top-k expert outputs, weighted by routing probabilities `p`. The effective magnitude of this sum scales as `‖p‖₂` (the L2 norm of the weight vector).

**Shared expert**: Always active with implicit weight 1.0. If there are `s` shared experts, their combined magnitude scales as `√s`.

**Goal**: Make both contributions comparable in magnitude at initialization:
```
magnitude(shared) ≈ magnitude(routed)
√s × λ ≈ ‖p‖₂
λ = √s / ‖p‖₂
```

---

## Monte Carlo Estimation

Since `p` depends on random router logits at initialization, we estimate `λ` by sampling:

```python
import numpy as np

def softmax(x):                       # numerically stable (matches the code)
    e = np.exp(x - x.max())
    return e / e.sum()

n   = num_routed_experts      # e.g., 16
k   = num_experts_per_tok     # e.g., 6
s   = num_shared_experts      # e.g., 1
lam = router_lambda           # e.g., 1.0  (logit-norm scaling, Skywork-MoE)

factors = []
for _ in range(10000):
    # Random router logits at init (weights are ~N(0,1))
    logits = np.random.randn(n - s)
    # Router logit normalization, THEN router_lambda scaling (must match router.py)
    logits = (logits - logits.mean()) / (logits.std() + 1e-6)
    logits = lam * logits
    # Top-k routing probabilities
    p = np.sort(softmax(logits))[::-1][:k - s]
    # Compute scaling factor
    factors.append(s**0.5 / (np.sum(p**2)**0.5))

lambda_scaling = round(np.mean(factors), 2)
```

> The MC sim **accounts for `router_lambda`** (the z-score logit normalization + λ scaling applied
> in the router) — sharper routing (higher λ) concentrates `p`, which changes `‖p‖₂` and hence the
> scaling. The softmax is the numerically stable `exp(x − max)` form, matching `configuration_bibo.py`.

---

## Typical Values

| num_routed_experts | num_experts_per_tok | num_shared | λ (approx) |
|-------------------|--------------------:|------------|------------|
| 8                 | 2                   | 1          | ~1.8       |
| 8                 | 4                   | 1          | ~1.2       |
| 16                | 6                   | 1          | ~2.0       |
| 16                | 2                   | 1          | ~3.5       |
| 64                | 6                   | 1          | ~3.8       |

Higher `λ` when fewer experts are selected (top-k is small relative to total) because the routing probability concentrates on fewer experts → higher `‖p‖₂` → wait, actually lower `‖p‖₂` when spread across more... Let me be precise:

- **More experts selected (higher k)**: Probability spreads → lower `‖p‖₂` → higher `λ`
- **Fewer experts selected (lower k)**: Probability concentrates → higher `‖p‖₂` → lower `λ`
- **More total experts (higher n)**: Softmax spreads more → lower per-expert prob → lower `‖p‖₂` → higher `λ`

---

## Why Auto-Compute?

If you change `num_routed_experts` or `num_experts_per_tok`, the optimal `λ` changes. Auto-computation ensures the shared expert contribution stays balanced regardless of MoE configuration.

**To skip auto-computation** (e.g., for faster config init), pass an explicit value:
```python
config = BiBoConfig(moe_shared_scaling=2.0)  # Skips Monte Carlo
```

---

## Novel Contribution: Causal Conv1D as Shared Expert

The shared expert in BiBo is a **Causal Conv1D** — not a standard MLP. This is a novel design choice:

1. **Always-active local pattern capture**: Every token passes through the conv expert regardless of routing, giving the model a guaranteed local-context pathway
2. **Complementary to routed MLPs**: Routed MLP experts capture token-level transformations; the shared conv captures sequential/local patterns (n-gram-like features)
3. **Causal padding**: Left-padded by `kernel_size - 1` to maintain autoregressive property
4. **Gated architecture**: `down_proj(act(conv(x)) * up_proj(x))` — same gating as SwiGLU but with conv instead of linear for the gate path

This means every MoE layer has two complementary pathways:
- **Routed path**: Token-level expert specialization (MLP + special experts)
- **Shared path**: Local sequential pattern capture (causal conv)

---

## References

- DeepSeek-V2/V3: Shared expert architecture concept
- Muon optimizer discussions: Scaling considerations for shared vs routed
- BiBo novel: Using Conv1D (not MLP) as the shared expert
