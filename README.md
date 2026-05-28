# BiBo: Diverse-Expert MoE Transformer with SSMax Attention

**BiBo** is a research Mixture-of-Experts (MoE) Transformer for causal language modeling. It explores diverse expert architectures and sequence-length-aware attention scaling for improved long-context performance and expert utilization.

## Key Innovations

1. **SSMax (Scalable-Softmax)** — Learnable per-head query scaling (`scale * log(kv_len)`) that prevents attention fading at long sequences. Based on [arXiv:2501.19399](https://arxiv.org/abs/2501.19399).

2. **Diverse Expert Pool** — Not all experts are MLPs. The expert layout includes Identity, Zero, and ReLU² experts alongside standard SwiGLU MLPs, enabling the model to learn when tokens need no transformation, gated suppression, or simple non-linearity.

3. **Shared Causal Conv1D Expert** — An always-active gated causal convolution that provides local temporal context to every token, independent of routing decisions. Novel — no prior MoE work uses convolution as a shared expert.

4. **Router Logit Normalization** — `router_lambda` scales normalized logits, preventing top-1 confidence collapse when `top_k > 1`. This ensures all selected experts contribute meaningfully (not just the top-1 dominating).

5. **Threshold-Based Bias Heuristics** — Non-trainable router bias (`requires_grad=False`) updated via load-balancing heuristics. Avoids FSDP conflicts while maintaining expert utilization balance.

6. **Conv Router Option** — `router_type="conv"` gives the router local context awareness via causal 1D convolution over hidden states before expert selection.

7. **Flash Attention (SDPA)** — Uses `F.scaled_dot_product_attention` by default, with manual fallback when `output_attentions=True`.

## Architecture

```
BiBoForCausalLM
├── BiBoModel
│   ├── Embedding (vocab → hidden)
│   ├── BiBoRotaryEmbedding (RoPE)
│   ├── BiBoDecoderLayer × N
│   │   ├── RMSNorm → BiBoAttention (GQA + QK-Norm + SSMax + SDPA)
│   │   └── RMSNorm → BiBoMoELayer (or dense BiBoMLP for first/last layers)
│   │       ├── BiBoMoERouter (MLP or Conv, logit normalization)
│   │       ├── Routed: (n-3) SwiGLU MLPs + 1 Identity + 1 Zero + 1 ReLU²
│   │       └── Shared: 1 CausalConv1D (always active, scaled by moe_shared_scaling)
│   └── Final RMSNorm
└── LM Head
```

**Expert layout**: `[0..n-4]` = SwiGLU MLPs, `[n-3]` = Identity, `[n-2]` = Zero, `[n-1]` = ReLU²

**MoE layers**: First 2 decoder layers use dense MLP (configurable via `mlp_only_layers`). All remaining layers use MoE routing.

## Installation

```bash
git clone https://github.com/IsNoobgrammer/BiBo.git
cd BiBo
pip install -r requirements.txt
```

**Requirements**: PyTorch ≥ 2.0, Transformers ≥ 4.40, einops ≥ 0.7

## Quick Start

```python
import torch
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

config = BiBoConfig(
    vocab_size=5000,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    num_routed_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=256,
    intermediate_size=1024,
    moe_shared_scaling=2.0,
)

model = BiBoForCausalLM(config)
x = torch.randint(0, 5000, (2, 128))
out = model(x, labels=x)
print(f"Loss: {out.loss.item():.4f}")
```

## Configuration

Key parameters (see [`docs/configuration_guide.md`](docs/configuration_guide.md) for full reference):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ssmax` | `True` | Enable SSMax query scaling |
| `num_routed_experts` | `16` | Total routed experts (must be ≥ 4) |
| `num_experts_per_tok` | `6` | Top-K routing |
| `router_type` | `"mlp"` | Router architecture (`"mlp"` or `"conv"`) |
| `router_lambda` | `1.0` | Logit norm scaling (higher = more decisive) |
| `router_noise` | `0.5` | Exploration noise during training |
| `bias_update_factor` | auto | Load balancing step size (Hill function of n) |
| `bias_update_threshold` | `8000` | Tokens between bias updates |
| `moe_shared_scaling` | auto | Shared expert output scaling (Monte Carlo estimated) |
| `mlp_only_layers` | `[0, 1]` | Layers using dense MLP instead of MoE |
| `max_position_embeddings` | `32768` | Maximum context length |
| `rope_theta` | auto | RoPE base frequency (scales with context length) |

## Project Structure

```
src/
├── configuration_bibo.py          # BiBoConfig (all hyperparams + auto-derivation)
├── modeling_bibo.py               # Flat re-export for backward compat
└── modeling/
    ├── norm.py                    # BiBoRMSNorm
    ├── embed.py                   # BiBoRotaryEmbedding (Qwen3-compatible RoPE)
    ├── attn/
    │   ├── base.py                # BiBoAttention (GQA + SSMax + SDPA)
    │   ├── ssmax.py               # apply_ssmax_query_scaling
    │   └── utils.py               # repeat_kv
    ├── ffn/
    │   ├── mlp.py                 # BiBoMLP (SwiGLU)
    │   ├── experts.py             # Identity, ReLU², Zero, CausalConv1D
    │   ├── router.py              # BiBoMoERouter (MLP or Conv, logit norm)
    │   └── moe.py                 # BiBoMoELayer (routing + dispatch + bias update)
    ├── layers.py                  # BiBoDecoderLayer
    └── models.py                  # BiBoModel, BiBoForCausalLM

baseline/                          # Reference implementations
├── qwen3/                         # Qwen3 dense model
└── qwen3moe/                      # Qwen3MoE (primary baseline)

docs/                              # Technical documentation
├── ssmax.md                       # SSMax theory + implementation
├── moe_shared_scaling.md          # Monte Carlo scaling derivation
├── configuration_guide.md         # Full config parameter reference
└── deprecated.md                  # Removed components + reasoning

misc/kaggle/multi_gpu/             # 2×T4 parallel ablation
├── config.yaml, data.py, train.py
├── analyze_router.py              # Per-token router analysis + plots
├── metrics/                       # JSON metrics
├── plots/                         # Generated visualizations
└── report/                        # Next.js report (GitHub Pages)
```

## Ablation Results

BiBo was benchmarked against Qwen3MoE on a sequence sorting task (2×T4 GPUs, Kaggle):

| Metric | BiBo | Qwen3MoE |
|--------|------|----------|
| Parameters | 8.3M | 11.2M |
| Final Loss | ~0.10 | ~0.25 |
| Convergence | Faster | Slower |
| Top-1 Confidence | 0.4–0.7 (healthy) | 0.9+ (wasteful) |

**Key insight**: BiBo's logit normalization produces moderate confidence across top-k experts = better expert utilization. Qwen's raw softmax gives 0.9+ top-1 confidence, effectively wasting the other k-1 selected experts.

## Design Decisions

- **No sliding window / recurrent attention** — only standard softmax + SSMax
- **SSMax init**: `1.0 / log(max_pos_emb / 2)` — ensures attention starts ~neutral, not 6× sharper than standard
- **Shared expert is NOT routed** — always active CausalConv1D
- **Router bias is non-trainable** — `requires_grad=False`, updated via heuristic `.add_()`
- **Noise expert was removed** — no evidence it helps; Identity covers the "dump bucket" use case (see [`docs/deprecated.md`](docs/deprecated.md))
- **Logit norm prevents expert waste** — when `top_k > 1`, normalization ensures all selected experts contribute meaningfully

## Expert Types

| Expert | Behavior | Purpose |
|--------|----------|---------|
| SwiGLU MLP | Standard gated FFN | General transformation |
| Identity | `output = input` | Skip connection / no-op routing |
| Zero | `output = 0` | Learned suppression |
| ReLU² | `ReLU(Wx)²` | Sparse, high-activation features |
| CausalConv1D (shared) | Gated temporal convolution | Local context (always active) |

## Router Types

| Type | Mechanism | Use Case |
|------|-----------|----------|
| MLP | `Linear(hidden → n_experts)` | Fast, global token-to-expert mapping |
| Conv | `CausalConv1D(hidden → n_experts)` | Local temporal patterns in routing |

## Documentation

- [`docs/ssmax.md`](docs/ssmax.md) — SSMax theory, initialization, and integration
- [`docs/moe_shared_scaling.md`](docs/moe_shared_scaling.md) — Monte Carlo derivation for shared expert scaling
- [`docs/configuration_guide.md`](docs/configuration_guide.md) — Full parameter reference and tuning guidance
- [`docs/deprecated.md`](docs/deprecated.md) — Removed components and reasoning

## Citation

```bibtex
@software{bibo2025,
  title={BiBo: Diverse-Expert MoE Transformer with SSMax Attention},
  author={Shaurya Sharthak and SedGram and adi-kmt},
  year={2025},
  url={https://github.com/IsNoobgrammer/BiBo}
}
```

## License

Apache 2.0

## Acknowledgments

- SSMax from [Scalable-Softmax Is Superior for Attention](https://arxiv.org/abs/2501.19399) (Nakanishi, 2025)
- MoE architecture inspired by Qwen3-MoE, DeepSeek-V2/V3, MoE++ (Skywork AI)
- Router logit normalization from Skywork-MoE
- RoPE implementation compatible with Qwen3
