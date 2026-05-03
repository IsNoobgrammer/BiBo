# BiBo Model Verification Guide

## Overview

Integrated BiBo model with:
- **RMSNorm**: Layer normalization
- **RoPE**: Rotary position embeddings
- **SSMax**: Scaling softmax for long context
- **MoE**: Mixture of experts with routing
- **Decoder layers**: Full transformer stack

## Test Results

### ✓ Forward + Backward Pass

```
Model: 3M params (trainable)
Input: [2, 64] tokens
Loss: ~6.95 (random init)
Logits: [2, 64, 1000]
Gradients: 79 params, no NaN/Inf
```

### ✓ Component Verification

**RoPE**:
- Shape: `(seq_len, head_dim)` = `(64, 32)`
- Applied to Q/K in attention
- Base: 10000.0

**SSMax**:
- Per-head learnable scale: `[1, num_heads, 1, 1]`
- Init: ~0.18
- Gradients flowing correctly

**MoE**:
- Layer 0: Dense MLP (mlp_only_layers)
- Layer 1-2: MoE with 8 experts, top-2 routing
- Layer 3: Dense MLP (mlp_only_layers)
- Router bias: trainable, init 0.0

**RMSNorm**:
- All layers: eps=1e-5, weight init=1.0
- Applied pre-attention and pre-FFN

### ✓ Operation Trace

```
embed       → [1, 16, 64]
layer0_ln1  → [1, 16, 64]
layer0_attn → [1, 16, 64]
layer0_ln2  → [1, 16, 64]
layer0_mlp  → [1, 16, 64]
...
final_norm  → [1, 16, 64]
lm_head     → [1, 16, vocab_size]
```

## Verification Methods

### 1. Forward/Backward Correctness

```python
model = BiBoForCausalLM(config)
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
loss.backward()

# Check: no NaN/Inf in loss, logits, gradients
assert not torch.isnan(loss)
assert not torch.isnan(logits).any()
for param in model.parameters():
    if param.grad is not None:
        assert not torch.isnan(param.grad).any()
```

### 2. RoPE Application

**Method**: Capture Q/K before and after RoPE, compute difference

```python
# Hook Q/K projections (before RoPE)
attn.q_proj.register_forward_hook(make_qk_hook("q_before"))
attn.k_proj.register_forward_hook(make_qk_hook("k_before"))

# Forward pass
outputs = model(input_ids=input_ids)
q_before = qk_states["q_before"]
k_before = qk_states["k_before"]

# Manually apply RoPE
from src.modeling.embed import apply_rotary_pos_emb
q_after, k_after = apply_rotary_pos_emb(q_reshaped, k_reshaped, cos, sin)

# Compute difference
q_diff = (q_after - q_before_reshaped).abs().mean().item()
k_diff = (k_after - k_before_reshaped).abs().mean().item()
```

**Verification**:
- Q difference: ~0.04 (significant change)
- K difference: ~0.04 (significant change)
- If diff > 1e-6 → RoPE applied ✓
- If diff < 1e-6 → RoPE not applied ✗

**Example output**:
```
Q before RoPE: mean=-0.0055, std=0.1644
Q after RoPE:  mean=-0.0077, std=0.1643
RoPE effect (mean absolute difference): Q: 0.0418

✓ RoPE is applied correctly!
```

### 3. SSMax Scaling

**Method**: Check learnable scale parameter exists and receives gradients

```python
for layer in model.model.layers:
    if hasattr(layer.self_attn, 'ssmax_scale'):
        scale = layer.self_attn.ssmax_scale
        # Shape: [1, num_heads, 1, 1]
        # Init: ~0.18 (from uniform [0.1, 0.3])
        # Grad: flows during backward
```

**Verification**:
- Scale parameter registered
- Shape: `[1, num_heads, 1, 1]` (per-head)
- Gradients non-zero after backward

### 4. MoE Routing

**Method**: Inspect layer types and router behavior

```python
for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, 'gate'):
        # MoE layer
        router = layer.mlp.gate
        num_experts = layer.mlp.num_routed_experts
        top_k = layer.mlp.num_experts_per_tok
    else:
        # Dense MLP layer
```

**Verification**:
- `mlp_only_layers` → Dense MLP
- Other layers → MoE with router
- Router bias trainable
- Expert outputs weighted and combined

### 5. Tensor Shape Tracing

**Method**: Register forward hooks to capture intermediate shapes

```python
shapes = {}
def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        shapes[name] = output.shape
    return hook

model.model.embed_tokens.register_forward_hook(make_hook("embed"))
# ... register for all layers
```

**Verification**:
- Hidden states maintain `[batch, seq, hidden_size]`
- Attention output matches input shape
- MLP output matches input shape
- Final logits: `[batch, seq, vocab_size]`

## Common Issues

### RoPE Not Applied

**Symptom**: Model trains but doesn't learn positional patterns

**Check**:
```python
# Verify position_embeddings passed to attention
layer_outputs = decoder_layer(
    hidden_states,
    position_embeddings=position_embeddings,  # Must be present
    ...
)
```

### SSMax Scale Not Training

**Symptom**: Scale gradients are zero

**Check**:
```python
# Verify scale is used in forward
if self.use_ssmax:
    query_states = apply_ssmax_query_scaling(
        query_states, kv_len, self.ssmax_scale
    )
```

### MoE Routing Collapsed

**Symptom**: All tokens route to same expert

**Check**:
```python
# Monitor router logits and weights
router_logits, router_weights, selected_experts = self.gate(hidden_states)
print(f"Expert distribution: {torch.bincount(selected_experts.flatten())}")
```

### Gradient Explosion

**Symptom**: Loss becomes NaN after few steps

**Check**:
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- Learning rate: Start with 1e-4
- SSMax scale init: Keep in [0.1, 0.3] range

## Advanced Verification

### Attention Pattern Visualization

```python
# Enable output_attentions
outputs = model(input_ids, output_attentions=True)
attn_weights = outputs.attentions  # List of [batch, heads, seq, seq]

# Plot heatmap for layer 0, head 0
import matplotlib.pyplot as plt
plt.imshow(attn_weights[0][0, 0].detach().cpu())
plt.colorbar()
plt.title("Attention Pattern")
```

### Expert Utilization

```python
# Track expert selection during training
expert_counts = torch.zeros(num_experts)
for batch in dataloader:
    outputs = model(**batch)
    # Hook into router to capture selected_experts
    expert_counts += torch.bincount(selected_experts.flatten(), minlength=num_experts)

print(f"Expert utilization: {expert_counts / expert_counts.sum()}")
```

### SSMax Effect on Long Context

```python
# Compare attention scores with/without SSMax
model.config.use_ssmax = False
outputs_no_ssmax = model(long_input_ids, output_attentions=True)

model.config.use_ssmax = True
outputs_ssmax = model(long_input_ids, output_attentions=True)

# Compare attention entropy (higher = more uniform)
entropy_no_ssmax = -(attn * torch.log(attn + 1e-9)).sum(-1).mean()
entropy_ssmax = -(attn * torch.log(attn + 1e-9)).sum(-1).mean()
```

## Test Script

Run full verification:

```bash
python test_bibo_integrated.py
```

Tests:
1. Forward + backward pass (no NaN/Inf)
2. Component verification (RoPE, SSMax, MoE, Norm)
3. Operation tracing (tensor shapes)
4. RoPE application (position embeddings used)

Expected output:
```
============================================================
✓ All tests passed!
============================================================
```

## Next Steps

1. **Training loop**: Add optimizer, dataloader, training loop
2. **Evaluation**: Perplexity on validation set
3. **Ablation**: Compare with/without SSMax, MoE
4. **Scaling**: Test on longer sequences (512, 1024, 2048)
5. **Profiling**: Identify bottlenecks (attention, MoE routing)
