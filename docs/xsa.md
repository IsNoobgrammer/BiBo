# XSA (Exclusive Self Attention) Technical Documentation

## Overview

**XSA (Exclusive Self Attention)** is a parameter-free, two-line modification to standard
self-attention. After the usual value aggregation, it removes the component of each token's
attention output that lies **along that same token's own value vector** — making the output
"exclusive" of the self-value direction.

**Paper**: [Exclusive Self Attention](https://arxiv.org/abs/2603.09078) (arXiv:2603.09078) ·
notes: https://www.k-a.in/XSA.html

## The Idea

In standard self-attention, the output at position `i` is the value-weighted sum

```
y_i = Σ_j a_{i,j} · v_j
```

Because the attention weights `a_{i,*}` always include the diagonal term `a_{i,i}` (a token
attends to itself), `y_i` carries a persistent component pointing along its own value vector
`v_i`. XSA strips that component out: each token's output is forced to be **orthogonal to its
own value**, so the residual stream is enriched only by what the *other* tokens (and the
self-token's off-diagonal contribution) bring in along directions `v_i` does not already span.

## Core Formula

```
z_i = y_i − (y_iᵀ · v_i) · v_i / ‖v_i‖₂²
```

This is the **vector rejection** of `y_i` from `v_i` — `y_i` minus its projection onto `v_i`.
After the operation, `z_i · v_i = 0` by construction.

### Equivalent normalized form (the implementation)

With `v̂_i = v_i / ‖v_i‖₂` (an L2-normalized value):

```
z_i = y_i − (y_iᵀ · v̂_i) · v̂_i
```

These are algebraically identical:
`(y·v̂)·v̂ = (y·v/‖v‖)·(v/‖v‖) = (y·v)·v/‖v‖²`. The normalized form is also numerically
safer — `F.normalize` clamps the denominator with an `eps`, avoiding a divide-by-zero when a
value vector is ~0, whereas dividing by `‖v‖²` directly does not.

## Where It Is Applied

XSA runs **after** the attention value-aggregation and **before** the output projection:

```
Q, K, V  →  QK-norm → RoPE → (SSMax) → softmax·V  =  y     (standard attention output)
y        →  XSA rejection                          =  z
z        →  o_proj                                          (output projection)
```

It does **not** touch the softmax, the logits, or the KV cache. It is a pure post-processing
step on the attention output tensor.

## Implementation

`src/modeling/attn/xsa.py`:

`src/modeling/attn/xsa.py` (eager) — `enable_gqa=True` broadcasts V across the query group
**without materializing a `repeat_kv` copy** (SDPA-style):

```python
def apply_xsa(attn_output, value_states, enable_gqa=True):
    # attn_output: (B, H, S, D);  value_states: (B, H_kv, S, D)
    B, n_heads, S, D = attn_output.shape
    n_kv = value_states.shape[1]; g = n_heads // n_kv
    if not enable_gqa and g != 1:                       # legacy: materialize repeat_kv
        value_states = repeat_kv(value_states, g); n_kv, g = n_heads, 1
    Yg = attn_output.view(B, n_kv, g, S, D)             # group view, no copy
    Vn = F.normalize(value_states, dim=-1).unsqueeze(2) # (B, n_kv, 1, S, D), broadcasts over g
    return (Yg - (Yg * Vn).sum(-1, keepdim=True) * Vn).reshape(B, n_heads, S, D)
```

Wired into `BiBoAttention.forward` (`src/modeling/attn/base.py`) behind `config.use_xsa`:

```python
if self.use_xsa:
    attn_output = apply_xsa(attn_output, value_states, enable_gqa=True)
```

### GQA handling

Under grouped-query attention the value tensor has fewer heads (`H_kv < H`) than the attention
output. Each query head in a group is rejected against its group's shared value vector — the
correct interpretation of the per-position "self value" `v_i` under GQA.

- **`enable_gqa=True` (default):** the group is handled by **broadcasting** — `Yg` is a free
  reshape to `(B, H_kv, g, S, D)` and the normalized V broadcasts over the `g` axis. The
  `(B, H, S, D)` `repeat_kv` copy and the full-size normalized-V are **never materialized**
  (mirrors SDPA's `enable_gqa`). `enable_gqa=False` falls back to the legacy `repeat_kv` path.
- **`output_attentions=True` path:** `value_states` is already `repeat_kv`'d to `H` heads before
  the manual matmul, so `g == 1` and the broadcast is a no-op — consistent result on both paths.

### Fused Triton kernel (`src/kernels/xsa_fused.py`)

For training, `patch_xsa_with_triton()` swaps the eager rejection for a fused Triton kernel
(`fused_xsa`) — one forward + one backward kernel that read each tensor once, broadcast V across
the GQA group in-kernel (no `repeat_kv`, no `Vn` HBM write), and reduce in **fp32 in-register**.
Backward uses the identity `grad_Y = reject(grad_z, v̂)` (the rejection operator is
symmetric-idempotent), with `grad_V` analytic and accumulated over the group. Grad-exact vs eager
(fp32 ~2e-7; fp16 *more* accurate than fp16-eager since the eager reductions run in fp16).

Standalone module, RTX 3050 fp16 (`do_bench` median), exact ms / peak MB vs the old `repeat_kv` path:

| shape (H5 Hkv1 D64) | fwd repeat_kv → fused | fwd+bwd repeat_kv → fused | fwd+bwd peak MB |
|---|---|---|---|
| 1024 tok  | 0.109 → **0.042** ms | 0.495 → **0.153** ms | 5.3 → **4.0** |
| 4096 tok  | 0.334 → **0.142** ms | 1.430 → **0.559** ms | 21.1 → **16.0** |
| 16384 tok | 1.207 → **0.547** ms | 5.180 → **2.074** ms | 84.3 → **64.0** |
| big H32 Hkv8 (4096 tok) | 2.067 → **0.900** ms | 8.286 → **3.365** ms | 152.5 → **104.0** |

In the **whole `BiBoAttention`** block (SDPA-dominated, Liger norm on), using fused XSA costs
~0.2–0.3 ms fwd+bwd vs not using XSA, and **~0 extra peak memory**. Full tables + methodology in
`docs/kernel_benchmark_report.md`.

## Properties

- **Parameter-free** — no learnable weights, no initialization, identical in training and
  inference.
- **Per-head** — the rejection is computed independently per head over the head-dim `D`.
- **Cheap** — one elementwise multiply, one reduction, one normalize over the head dimension;
  `O(B·H·S·D)`, negligible vs. the attention matmuls.
- **Orthogonality guarantee** — `z_i · v_i = 0` after the operation.

## Verification (June 25, 2026)

`apply_xsa` was checked against the paper formula directly:

- **Formula match**: `max|z − z_ref| = 2.4e-7` (fp32) vs. the explicit
  `y − (y·v)·v/‖v‖²` reference — algebraically exact.
- **Rejection property**: `max|z · v| = 3.8e-6` (≈0) — output is orthogonal to the self-value.
- **GQA**: `repeat_kv` path produces correct per-group rejection; non-GQA path
  (`H_kv == H`) also correct.
- **End-to-end**: `BiBoAttention` forward+backward runs NaN-free with gradients flowing on
  **both** the SDPA and `output_attentions=True` paths.

## Config

| Param | Default | What it does |
|-------|---------|-------------|
| `use_xsa` | `True` | Enable Exclusive Self Attention rejection on the attention output |
