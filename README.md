# BiBo: Modular MoE Transformer

**BiBo** is a research-focused transformer architecture with modular components for experimentation:

- **RMSNorm**: Layer normalization
- **RoPE**: Rotary position embeddings
- **SSMax**: Scaling softmax for long context attention
- **MoE**: Mixture of experts with flexible routing (MLP or Conv)
- **Diverse experts**: MLP, Identity, Zero, Noise, ReLU, Causal Conv1D

## Features

### Architecture

- **Flexible MoE layers**: Configure which layers use MoE vs dense MLP
- **Mixed routers**: Support both MLP and Conv1D routers per layer
- **Expert diversity**: 
  - Standard MLP experts (SwiGLU activation)
  - Identity expert (skip connection)
  - Zero expert (learned gating)
  - Noise expert (regularization)
  - ReLU expert (simple non-linearity)
  - Shared causal Conv1D expert
- **SSMax attention**: Learnable per-head scaling for long context
- **Sliding window attention**: Optional local attention windows
- **RoPE**: Rotary position embeddings with configurable base

### Training Features

- **Adaptive router bias**: Automatic load balancing via threshold-based bias updates
- **Router noise**: Gumbel noise for exploration during training
- **Router lambda**: Skywork-MoE style logit normalization
- **Shared expert scaling**: DeepSeek-V2/V3 style shared expert weighting
- **Gradient checkpointing**: Memory-efficient training for large models

## Installation

```bash
git clone https://github.com/yourusername/BiBo.git
cd BiBo
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

# Create config
config = BiBoConfig(
    vocab_size=5000,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_routed_experts=8,
    num_experts_per_tok=4,
    use_ssmax=True,
)

# Create model
model = BiBoForCausalLM(config)

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 128))
outputs = model(input_ids=input_ids)
loss = outputs.loss
```

### Training Example

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Mixed Router Configuration

```python
# Layer 1,3 use Conv routers, Layer 2,4 use MLP routers
config = BiBoConfig(
    num_hidden_layers=6,
    num_routed_experts=8,
    mlp_only_layers=[0, 5],  # First and last layers are dense
    router_type="mlp",  # Default router type
    kernel_size=3,  # For conv routers
)

model = BiBoForCausalLM(config)

# Override specific layers to use conv routers
for i in [1, 3]:
    if hasattr(model.model.layers[i].mlp, 'gate'):
        conv_config = BiBoConfig(**{k: v for k, v in config.__dict__.items()})
        conv_config.router_type = "conv"
        from src.modeling.ffn.router import BiBoMoERouter
        model.model.layers[i].mlp.gate = BiBoMoERouter(conv_config)
```

## Configuration

Key configuration parameters:

```python
config = BiBoConfig(
    # Model architecture
    vocab_size=128000,
    hidden_size=1536,
    intermediate_size=4104,
    num_hidden_layers=8,
    num_attention_heads=12,
    num_key_value_heads=2,
    
    # MoE settings
    num_routed_experts=16,
    num_shared_experts=1,
    num_experts_per_tok=6,
    moe_intermediate_size=512,
    mlp_only_layers=[0, 7],  # Dense layers
    
    # Router settings
    router_type="mlp",  # "mlp" or "conv"
    router_temperature=1.3,
    router_noise=0.5,
    router_lambda=1.0,
    bias_update_factor=1e-2,
    bias_update_threshold=100_000,  # Tokens before bias update
    
    # Attention settings
    use_ssmax=True,
    use_sliding_window=True,
    sliding_window=512,
    max_window_layers=4,
    
    # Position embeddings
    max_position_embeddings=32768,
    rope_theta=10000.0,
)
```

## Testing

Run integration tests:

```bash
python tests/integration/test_bibo_integrated.py
```

Run training tests:

```bash
# 50-step basic training
python tests/training/train_bibo_steps.py

# 100-step mixed router training
python tests/training/train_mixed_routers.py
```

## Verification

See `VERIFICATION_GUIDE.md` for detailed component verification:

- Forward/backward pass correctness
- RoPE application verification
- SSMax scaling verification
- MoE routing verification
- Tensor shape tracing

## Training Results

See `TRAINING_RESULTS.md` for 50-step training results and `training_50steps.png` for visualizations.

Key findings:
- Model trains stably (no NaN/Inf)
- Expert load balancing works (balance ratio 0.68-0.78)
- Layer-wise expert specialization observed
- Router bias updates improve balance over time

## Model Architecture

```
BiBoForCausalLM
├── BiBoModel
│   ├── Embedding (vocab_size → hidden_size)
│   ├── BiBoRotaryEmbedding (RoPE)
│   ├── BiBoDecoderLayer × num_hidden_layers
│   │   ├── BiBoRMSNorm (input)
│   │   ├── BiBoAttention
│   │   │   ├── Q/K/V projections
│   │   │   ├── RoPE application
│   │   │   ├── SSMax scaling (optional)
│   │   │   └── Attention computation
│   │   ├── BiBoRMSNorm (post-attention)
│   │   └── BiBoMoELayer or BiBoMLP
│   │       ├── BiBoMoERouter (MLP or Conv)
│   │       ├── Routed experts (MLP, Identity, Zero, Noise, ReLU)
│   │       └── Shared expert (Causal Conv1D)
│   └── BiBoRMSNorm (final)
└── LM Head (hidden_size → vocab_size)
```

## Expert Types

1. **MLP Expert**: Standard SwiGLU FFN
2. **Identity Expert**: `output = input` (skip connection)
3. **Zero Expert**: `output = 0` (learned gating)
4. **Noise Expert**: `output = input + noise` (regularization)
5. **ReLU Expert**: `output = ReLU(linear(input))`
6. **Shared Causal Conv1D**: Temporal convolution across all tokens

## Router Types

### MLP Router
- Linear projection: `hidden_size → num_experts`
- Fast, parameter-efficient
- Global token-to-expert mapping

### Conv Router
- Causal 1D convolution: `hidden_size → num_experts`
- Captures local temporal patterns
- Kernel size configurable (default: 3)

## Performance

**Model sizes tested**:
- Small: 3M params (256 hidden, 4 layers)
- Medium: 27M params (512 hidden, 6 layers)
- Large: 46M params (512 hidden, 6 layers, 16 experts)

**Training stability**:
- ✓ No NaN/Inf in 100+ training steps
- ✓ Gradient norms stable (clipped at 1.0)
- ✓ Loss decreases consistently
- ✓ Expert utilization balanced

## Citation

```bibtex
@software{bibo2025,
  title={BiBo: Modular MoE Transformer},
  author={Shaurya Rohatgi},
  year={2025},
  url={https://github.com/IsNoobgrammer/BiBo}
}
```

## License

Apache 2.0

## Acknowledgments

- RoPE implementation inspired by Qwen3
- MoE architecture inspired by Qwen3-MoE, DeepSeek-V2/V3
- SSMax attention from scaling softmax research
- Router bias balancing from Skywork-MoE
