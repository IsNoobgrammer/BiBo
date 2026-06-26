# SSMax (Scalable-Softmax) Technical Documentation

## Overview

**SSMax (Scalable-Softmax)** is a novel attention mechanism designed to address the **attention fading problem** in Transformer models when processing long contexts. It replaces the standard softmax function in attention layers with a sequence-length-aware variant that maintains attention sharpness regardless of context size.

**Paper**: [Scalable-Softmax Is Superior for Attention](https://arxiv.org/abs/2501.19399) by Ken M. Nakanishi

## The Problem: Attention Fading

### Standard Softmax Behavior

In standard Transformer attention, softmax is applied to compute attention scores:

```
attention_score_i = exp(z_i) / Σ exp(z_j)
```

As the sequence length `n` increases:
- The **denominator** grows (sum of more exponentials)
- The **numerator** stays constant (single exponential)
- Result: Maximum attention scores approach zero → **attention distribution flattens**

This phenomenon, called **attention fading**, reduces the model's ability to focus on key information in long contexts.

### Mathematical Analysis

For an input vector of size `n`, the maximum element after softmax:

```
max_output ≤ exp(z_max) / [(n-1)·exp(z_min) + exp(z_max)]
           = 1 / [(n-1)/exp(z_max - z_min) + 1]
```

As `n → ∞`, this approaches **0** unless `z_max - z_min` grows proportionally with `n`.

## The Solution: SSMax

### Core Formula

SSMax replaces the exponential base with a sequence-length-dependent term:

```
SSMax(z_i) = n^(s·z_i) / Σ n^(s·z_j)
           = exp((s·log(n))·z_i) / Σ exp((s·log(n))·z_j)
```

Where:
- `n` = sequence length (number of keys)
- `s` = **learnable scaling parameter** (per-head, per-layer)
- `z_i` = attention logit for position i

### Key Properties

1. **Prevents Attention Fading**: Maximum attention score approaches 1 when `z_max - z_2nd > 1/s`, regardless of sequence length
2. **Adaptive Temperature**: Effective temperature scales with `s·log(n)`, automatically adjusting for context size
3. **Learnable Control**: Each attention head learns its own `s` parameter to control sharpness

### Mathematical Guarantees

For SSMax with scaling parameter `s > 0`:

```
max_output ≥ 1 / [(n-1)/n^(s·(z_max - z_2nd)) + 1]
```

- If `z_max - z_2nd > 1/s`: max_output → 1 (focused attention)
- If `z_max - z_min < 1/s`: max_output → 0 (distributed attention)

This means attention focuses on elements exceeding others by approximately `1/s`, independent of sequence length.

## Implementation in Attention

### Standard Attention (Softmax)

```python
attn_scores = (Q @ K^T) / sqrt(d)
attn_weights = softmax(attn_scores)
output = attn_weights @ V
```

### SSMax Attention

```python
# Apply SSMax scaling to queries — PER CAUSAL POSITION (not a single global kv_len).
# Query at position t attends to t+1 keys under the causal mask, so n varies along the
# sequence. n_j = (kv_len - q_len) + j + 1 for query j:
#   training (q_len==kv_len==L):  n = 1..L   (the real length-adaptive signal)
#   single-token decode:          n = kv_len (one query)
log_n = log(n_per_position)          # shape (q_len,), broadcast as (1,1,q_len,1)
scaled_Q = Q * ssmax_scale * log_n

# Rest is identical to standard attention
attn_scores = (scaled_Q @ K^T) / sqrt(d)
attn_weights = softmax(attn_scores)  # Standard softmax!
output = attn_weights @ V
```

**Key Insight**: SSMax is implemented by scaling the query vectors by `s·log(n)` before computing attention scores. The softmax itself remains unchanged.

## The Initialization Bug

### Problem Description

**Current (Buggy) Initialization**:
```python
self.ssmax_scale = nn.Parameter(torch.full((1, num_heads, 1, 1), 1.0))
```

This initializes `ssmax_scale = 1.0` per head, which causes:

1. **Immediate Over-Sharpening**: During early training with `kv_len=512`:
   ```
   effective_scale = ssmax_scale × log(512) / sqrt(head_dim)
                   ≈ 1.0 × 6.2 / 8
                   ≈ 0.78
   ```

2. **Comparison to Standard Attention**:
   - Standard: `1/sqrt(head_dim) = 1/sqrt(64) ≈ 0.125`
   - SSMax (buggy): `≈ 0.78` → **~6× sharper**

3. **Consequences**:
   - Attention entropy collapses in first few thousand steps
   - Model focuses too aggressively on single tokens
   - Training instability and poor generalization

### Why This Happens

The effective query scale becomes:
```
Q_effective = Q × ssmax_scale × log(n) / sqrt(d)
```

With `ssmax_scale=1.0` and typical `n=512-2048`:
- `log(512) ≈ 6.2`
- `log(2048) ≈ 7.6`

This makes attention **much sharper** than the standard `1/sqrt(d)` scaling, causing premature convergence to peaked distributions.

### Correct Initialization

**Fixed Initialization**:
```python
import math

# Initialize so ssmax_scale × log(typical_seq_len) ≈ 1.0
typical_log = math.log(max(config.max_position_embeddings / 2, 2.0))
init_val = 1.0 / typical_log  # ≈ 0.13 for max_pos_emb=2048

self.ssmax_scale = nn.Parameter(
    torch.full((1, num_heads, 1, 1), init_val),
    requires_grad=True
)
```

**Rationale**:
- For `max_position_embeddings=2048`: `typical_log = log(1024) ≈ 6.9`
- `init_val = 1.0 / 6.9 ≈ 0.145`
- Effective scale at start: `0.145 × log(512) ≈ 0.9` → close to standard attention
- Model can learn to deviate from this baseline as needed

## Empirical Results from Paper

### Training Efficiency
- SSMax models achieve **~0.008 lower training loss** compared to standard Transformers
- Faster convergence throughout pretraining

### Long-Context Generalization
- Standard Transformer: Loss increases significantly beyond training length
- SSMax: Maintains low loss up to **10× training sequence length**
- Robust to RoPE theta modifications (50× increase without retraining)

### Key Information Retrieval (Needle-in-Haystack)
- Standard Transformer: Fails beyond short contexts
- SSMax: High retrieval accuracy at **10× training length**
- Attention scores show SSMax focuses on key tokens even in long contexts

### Attention Score Analysis
- Standard softmax: Needle scores approach zero in long contexts
- SSMax: Maintains high attention allocation to key information
- Scaling parameter `s` is crucial for effective retrieval

## Design Variants

### With Bias Parameter (Not Recommended for Long Context)
```python
SSMax_bias(z_i) = exp((s·log(n) + b)·z_i) / Σ exp((s·log(n) + b)·z_j)
```

- Improves training efficiency slightly
- **Degrades long-context performance**
- Not recommended for length generalization tasks

### Without Scaling Parameter (Simplified)
```python
SSMax_fixed(z_i) = n^z_i / Σ n^z_j  # Equivalent to s=1.0
```

- Similar training curves to full SSMax
- **Lower key information retrieval accuracy**
- Scaling parameter provides important adaptability

## Implementation Guidelines

### When to Use SSMax
- ✅ Models requiring long-context generalization
- ✅ Tasks with key information retrieval (RAG, QA)
- ✅ Training from scratch with length generalization goals
- ⚠️ Can be added mid-training with warmup (partial benefits)
- ❌ Not needed for fixed short-context tasks
- ✅ **Full / global causal attention layers** — where the attended key count `n` grows with the
  sequence. This is where SSMax does real work.

### ⚠️ Do NOT use SSMax on sliding-window-attention (SWA) layers — it's redundant

**Recommendation: disable SSMax on any fixed-window layer (set its per-head `s = 0`, or don't apply
it there).** With a sliding window of fixed size `W` (e.g. 128, optionally + a few attention-sink
tokens), every query attends to **at most `W` keys regardless of total sequence length**, so:

1. **No length growth to compensate.** SSMax exists to counter fading *as `n` grows*; the window
   already caps `n ≤ W`, so fading never occurs at that layer. SWA and SSMax fight the same problem
   two ways — the window already holds it, SSMax has nothing to compensate (belt-and-suspenders).
2. **`n` is constant ⇒ SSMax degenerates to a constant temperature.** After the first `W` positions
   every query has `n = W`, so `s·log(n) = s·log(W)` is a fixed scalar — absorbable into the q/k
   weight norms, i.e. SSMax provides **zero additional inductive signal** there (same degeneracy as
   applying a single global `n` in training). It's not harmful, just dead weight.

The only place SSMax earns its keep under windowing is a **hybrid** model (some global layers, some
SWA): keep SSMax (`s > 0`) on the **global** layers, turn it off (`s = 0`) on the **windowed** ones.
Per-layer control is the right tool there (see the partial-SSMax research scope). Note: BiBo
currently uses full causal attention everywhere — this guidance applies if/when SWA is added.

### Initialization Best Practices
1. **Use sequence-aware initialization**: `init_val = 1.0 / log(typical_seq_len)`
2. **Typical sequence length**: Use `max_position_embeddings / 2` as estimate
3. **Per-head parameters**: Each head learns its own `s` (minimal overhead: 1 param/head)
4. **Make learnable**: Always set `requires_grad=True`

### Integration Checklist
- [ ] Initialize `ssmax_scale` with `1.0 / log(typical_seq_len)`
- [ ] Apply scaling in attention **per causal position**: `Q_scaled = Q × ssmax_scale × log(n_t)`, `n_t = (kv_len − q_len) + t + 1` (NOT a single global `log(kv_len)` — that collapses to a constant temperature during fixed-length training)
- [ ] Use standard softmax after scaling (no other changes needed)
- [ ] Monitor attention entropy during early training
- [ ] Verify no entropy collapse in first 1000 steps

## QK-norm × SSMax interaction (internal ablation, June 2026)

BiBo applies **QK-norm** (RMSNorm on Q and K) *and* **SSMax**. A 2×2 ablation on a synthetic passkey
length-generalization probe (tiny model, train @ 128, eval to 32×, dynamic-NTK RoPE, 3 seeds) shows
the pairing is **necessary, not incidental** — numbers are extrapolation accuracy (mean over 256–4096):

| | no SSMax | SSMax |
|---|---|---|
| **no QK-norm** | 0.86 | 0.97 |
| **QK-norm** | 0.57 | 0.96 |

- **QK-norm alone hurts length generalization** (0.86 → 0.57): bounding the logits removes the model's
  ability to sharpen attention by growing Q/K magnitude, and gives nothing back.
- **SSMax restores it** (0.57 → 0.96): with QK-norm on, SSMax is the *only* remaining sharpening lever,
  so it is far more load-bearing — SSMax's gain is **+0.39 with QK-norm vs +0.11 without**.
- **Takeaway:** in BiBo, SSMax is not optional decoration — it is the **required compensation** for the
  sharpening that QK-norm removes. QK-norm is kept for large-model training stability; SSMax is what
  makes that choice safe for length generalization. The two are a matched pair.

## Computational Overhead

**Minimal**: SSMax adds only:
- 1 learnable parameter per attention head
- 1 logarithm computation per forward pass
- 1 multiplication per query vector

For a 12-layer, 12-head model: **144 additional parameters** (negligible vs. 162M total)

## References

1. **Original Paper**: Nakanishi, K. M. (2025). "Scalable-Softmax Is Superior for Attention." arXiv:2501.19399
2. **Key Insight**: Attention fading occurs because softmax denominator grows with sequence length while numerator doesn't
3. **Solution**: Make exponential base sequence-length-dependent: `n^(s·z)` instead of `e^z`

## Related Work

- **RoPE (Rotary Position Embedding)**: Complementary positional encoding method
- **Sliding Window Attention**: Reduces computation but doesn't address attention fading
- **Linear Attention**: Different approach to long contexts with different tradeoffs
- **Focal Attention**: Uses temperature scaling but not sequence-length-aware

---

**Status**: SSMax is production-ready and can be seamlessly integrated into existing Transformer architectures with minimal code changes. The initialization fix is critical for stable training and optimal performance.
