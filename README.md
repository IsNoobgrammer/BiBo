# BiBo

A research Mixture-of-Experts transformer for causal language modeling. This file records the
**architecture decisions that are confirmed and adopted** — the configuration a new BiBo training
run should start from, and why each piece is there.

Every entry below was settled by an A/B at 524M or 1.05B tokens with matched seed and data order.
Where a mechanism was tested and rejected it is listed under [Refuted](#refuted), because knowing
what does not work is what stops it being re-tried.

---

## The adopted configuration

### Feed-forward

- **Sparse MoE**, 64 routed experts, **top-6** per token.
- **First and last layers are dense** (`mlp_only_layers=[0, N-1]`); every layer between them is MoE.
- **Radial NormSiLU** in every expert: `r^p * SiLU(g/r)` with `r = rms(gate)` and `p = sigmoid(theta)`
  learned per expert. `p` trains to a **depth ramp** (low early, high late) rather than a single
  value, so it needs its own learning rate (`--act_scale_lr 0.01`); left at the default LR it pins
  near its 0.5 init and looks like a dead axis when it is not.
- **No shared expert** and **no special (±Identity) experts** in the adopted config. Both exist in
  the codebase and both were measured; neither earned a slot.

### Routing

- **Sigmoid gate** (DeepSeek-V3 style): each expert is scored independently, not in competition.
- **Sum normalization** of the gathered top-k scores — `p_i = sigma_i / sum_j sigma_j`.
  **Not softmax.** `norm_topk_prob=True` maps to the string `"sum"` in `BiBoConfig`, and softmax
  over sigmoid scores would bound the max/min weight ratio at `e ~ 2.72`, which floors the minimum
  routed weight near 0.10. The measured minimum is **~0.04**, which only sum-normalization produces.
- **Bias-based load balancing**, heuristic and outside the optimizer (`requires_grad=False`,
  updated by `.add_()` on a token-count trigger). The balancing step must stay under the router's
  top-k boundary gap or one update flips a large share of tokens and the router dithers.
- **Router trained by Muon**, not AdamW.
- `routed_scaling_factor` stays at **1.0** — monotonically harmful above it, and a learnable
  per-layer version landed on par with the fixed default.

### Attention

- **Hybrid sliding-window** on a `[G, S, S]` block schedule with window 128. On 10 layers that puts
  global attention at **L0 / L3 / L6 / L9**, which are also the AttnRes block boundaries.
- **GQA** — 4 query heads, 2 KV heads.
- **QK-norm on every layer.** Global layers always get it; `swa_qk_norm` (default `True`) decides
  whether the windowed layers do as well, and they do.
- **XSA** — remove the token's own value component from its attention output:
  `Y <- Y - tanh(alpha) * (Y . V_hat) V_hat`, applied **per head, before `o_proj`**, with `alpha`
  a learnable per-head logit **initialised to 0** so XSA starts off and the model switches it on.
- **Partial RoPE**, `partial_rotary_factor=0.334`. At `head_dim=128` that rotates **42 dimensions**
  and passes the remaining 86 through as NoPE.
- **Tied input/output embeddings.**

### Residual topology (Kimi K3 Attention Residuals, modified)

Each layer reads a learned mixture over depth instead of only the running stream:

```python
Ht_A     = AR(s, block_residual)            # depth mix -- a READ, never enters the stream
attn_out = self_attn(input_layernorm(Ht_A))
Ht_M     = Ht_A + c * attn_out              # per-dimension carry
mlp_out  = mlp(post_attention_layernorm(Ht_M))
s        = s + attn_out + mlp_out           # the stream accumulates both, unweighted
```

- **Block size 3.** At every third layer the stream is committed as a depth candidate and then
  **restarted from attention alone** (`s = attn_out`, not `s + attn_out`). This is what makes the
  residual-stream norm sawtooth rather than grow monotonically.
- **One mix site per layer**, not Kimi's two. Halves the AttnRes parameters and the depth-mix
  compute; `mlp_res_norm` / `mlp_res_proj` are not even allocated.
- **Per-dimension learnable carry** `c`, a `hidden_size` vector per layer, **unbounded**, init 1.0.
  Attention therefore reaches two destinations at two different weights: the MLP at `c`, the
  residual stream at 1. Downstream layers always see the full attention contribution regardless of
  what `c` did locally.
- **Softmax** over the depth candidates.

### Numerics and optimizer

- **Muon** for matrices, **AdamW** for vectors. Vector-shaped `(1, H)` matrices — the AttnRes
  pseudo-queries — are routed to AdamW (`vec_matrices_adamw=True`). The flag is inert when AttnRes
  is off, since those are the only `(1, H)` parameters in the model.
- **bf16 residual stream.** An fp32-stream A/B was a wash (0.00028, inside noise).

---

## Refuted

Measured and rejected. Do not re-run these without a reason the architecture has changed.

| Mechanism | Verdict |
|---|---|
| **SSMax** | Refuted at 524M; worse loss in every window from step 500 and +0.00263 bpb. Removed from `src/`. |
| **Attention sinks** | Refuted at 524M. Loss indistinguishable from the control while costing 2.5% throughput and **2.1x the router-boundary-gap volatility** — XSA and the sink drain the same bucket. |
| **Sparse top-k depth mix** | Monotone dose-response in the wrong direction at 10 layers (top-6 and top-4 both worse than dense, top-4 worse than no AttnRes at all). The pool is only 2-11 candidates deep, so top-k is decimation rather than sparsity. Kept behind `--attn_res_topk` for deeper models. |
| **signorm depth mix** | `sigmoid(x_i)/sum(sigmoid(x_j))` lost at every configuration tested. |
| **Input-dependent carry gate** | A learned per-token gate on the carry path saturated open and degenerated. A static per-channel scalar beats it. |
| **Embedding skip term** | Redundant — the depth mix already routes ~48% of its weight to the raw embedding on average, so an explicit `d * emb` term adds little. |
| **Identity / special experts** | Closed twice. The expert-activation round refuted them at 25% of the pool over 8 pairings; a retry on the adopted AttnRes stack refuted them again at 5.9-11.1% (6+2, 6+0, 8+0 identity experts added alongside 64 GLU, all **4-6 sigma worse** than the no-specials control). The load balancer allotted slots proportionally to expert count as intended, so this is a real test rather than a broken setup. The throughput they buy is only +1.5-1.9%, which does not pay for +0.004-0.006 bpb. Some negative beats all positive (`6+2` was the best of the three), so signed pass-through is not purely additive — but not enough to matter. **The code is retained on purpose**: `--special_pairs`, `--no_pos_identity` / `--no_neg_identity`, and `--pos_identity_n` / `--neg_identity_n` for asymmetric counts. Do not delete on a cleanup pass. |
| **Zero expert** | `<grad_out, 0> = 0`, so the router never learns that Zero was the right pick. Replaced by the signed ±Identity pair. |
| **Skywork router logit-norm** | Removed; the router is pure MiMo/DeepSeek. |
| **Conv router** | Removed. MLP router only. |
| **`routed_scaling_factor > 1`** | Monotonically harmful. |
| **LongCat `glu_token_budget`** | Removed from `src/` in the Aug 2026 debloat. Special-expert share is now governed by expert count plus the load balancer, not a target ratio. |

**Block size 1** is a special case: it *wins* on bpb and generation quality when paired with the
per-dimension carry, reversing its loss at 524M without one. It is **not adopted** — every layer
becomes a boundary, which costs throughput out of proportion to the gain and changes what the
architecture represents.

---

## Installation

```bash
git clone https://github.com/IsNoobgrammer/BiBo.git
cd BiBo
pip install -r requirements.txt
```

PyTorch >= 2.0, Transformers >= 4.40, einops >= 0.7.

## Project structure

```
src/                          stable model
  configuration_bibo.py       BiBoConfig + auto-derivation and validation
  modeling/
    attn/base.py              GQA + QK-norm + XSA + SDPA
    attn/xsa.py               apply_xsa
    ffn/moe.py                routing, dispatch, bias update
    ffn/router.py             BiBoMoERouter
    ffn/experts.py            Identity / ReLU2 / CausalConv1D
exp/                          experimental model (Attention Residuals)
  modeling_bibo.py            apply_attention_residual, BiBoDecoderLayer
ablate/common/                training + eval harness
  train.py                    every ablation flag
  eval/                       bpb, MCQ, ICL, sampling, router interp
  tools/                      W&B helpers
baseline/qwen3moe/            param-matched reference
```

## Citation

```bibtex
@software{bibo2025,
  title={BiBo: A Diverse-Expert MoE Transformer},
  author={Shaurya Sharthak and SedGram and adi-kmt},
  year={2025},
  url={https://github.com/IsNoobgrammer/BiBo}
}
```

## License

Apache 2.0

## Acknowledgments

- Attention Residuals from Kimi K3 (Moonshot AI)
- MoE architecture informed by Qwen3-MoE, DeepSeek-V2/V3, MiMo
- Radial activation and signed pass-through experts developed here
