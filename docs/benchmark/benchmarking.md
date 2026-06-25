# BiBo Benchmarking Guide

> How to properly benchmark BiBo: smoke tests, throughput measurements, training runs, and config editing from Jupyter/Kaggle notebooks.

---

## Table of Contents

1. [Quick Smoke Test](#quick-smoke-test)
2. [Benchmark Suite Overview](#benchmark-suite-overview)
3. [Running the Benchmark](#running-the-benchmark)
4. [Editing BiBoConfig with %%writefile](#editing-biboconfig-with-writefile)
5. [Editing bench/config.py with %%writefile](#editing-benchconfigpy-with-writefile)
6. [Throughput & Memory Benchmarks](#throughput--memory-benchmarks)
7. [Full Training Run (Kaggle 2×T4)](#full-training-run-kaggle-2t4)
8. [Config Reference](#config-reference)
9. [Troubleshooting](#troubleshooting)

---

## Quick Smoke Test

Verify BiBo loads and runs a forward pass:

```python
# Local (Windows)
.\.venv\Scripts\python -c "from src.modeling_bibo import BiBoForCausalLM; print('OK')"

# Or from repo root
python -c "
import torch
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

cfg = BiBoConfig(vocab_size=5000, hidden_size=512, num_hidden_layers=4,
                 num_attention_heads=8, num_key_value_heads=2,
                 polyglu_expert_multiplier=2, special_expert_pairs=1,
                 num_experts_per_tok=2,
                 moe_intermediate_size=256, intermediate_size=1024,
                 moe_shared_scaling=2.0)
model = BiBoForCausalLM(cfg)
x = torch.randint(0, 5000, (2, 128))
out = model(x, labels=x)
print(f'Loss: {out.loss.item():.4f}')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

### Check Param Count of Baseline Config

```python
.\.venv\Scripts\python bench/config.py
```

Expected output:
```
BiBo Baseline Config:
  Total params: XX,XXX,XXX (XX.XXM)
  Embed params: XX,XXX,XXX
  LM head params: XX,XXX,XXX
  Hidden: 320
  Layers: 12
  Experts: 3 routed + 1 shared
  Top-K: 2
```

---

## Benchmark Suite Overview

The `bench/` directory is a self-contained training benchmark:

| File | Purpose |
|------|---------|
| `config.py` | BiBoConfig for ~50M baseline (no PolyGLU, uniform SwiGLU) |
| `data.py` | HF dataset loading, truncation to 1024, train/val split |
| `train.py` | Main training loop with FSDP2, torch.compile, WandB |
| `optim.py` | Muon + AdamW hybrid (falls back to pure AdamW) |
| `eval.py` | Validation loss + text generation with QTK-81K tokenizer |
| `utils.py` | WandB logging, checkpointing, throughput meter |

**Dataset:** `tinycompany/Instruct-packed-2K-Context-tk-QTK-81K` (pre-tokenized, packed to 2048, truncated to 1024)

**Tokenizer:** `fhai50032/QTK-81K` (81K vocab BPE, used only for inference/samples)

**Target:** < 2.8 validation loss on language modeling

---

## Running the Benchmark

### Local Single-GPU (RTX 3050, quick test)

```bash
# From repo root: C:\Users\shaur\OneDrive\Documents\BiBo

# Quick test — 100 steps, no WandB
.\.venv\Scripts\python bench/train.py --batch_size 4 --total_steps 100 --no_wandb --no_compile

# With torch.compile (slower startup, faster training)
.\.venv\Scripts\python bench/train.py --batch_size 4 --total_steps 500 --no_wandb

# Eval only (load checkpoint, run val, generate samples)
.\.venv\Scripts\python bench/train.py --eval_only --resume bench/checkpoints/final.pt
```

### Kaggle 2×T4 (full run)

```python
# Cell 1: Clone and install
!git clone https://github.com/IsNoobgrammer/BiBo.git
%cd BiBo
!pip install -qU transformers einops wandb bitsandbytes pyyaml seaborn datasets hf_transfer

# Cell 2: Set WandB key (optional)
import os
os.environ['WANDB_API_KEY'] = 'your_key_here'
# Or disable: os.environ['WANDB_MODE'] = 'disabled'

# Cell 3: Run benchmark (multi-GPU via torchrun)
!torchrun --nproc_per_node=2 bench/train.py \
    --batch_size 16 \
    --total_steps 50000 \
    --warmup_steps 1000 \
    --lr 3e-4 \
    --eval_every 500 \
    --sample_every 1000 \
    --seq_len 1024
```

### Command-Line Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--batch_size` | 32 | Batch size per GPU |
| `--total_steps` | 50000 | Total training steps |
| `--warmup_steps` | 1000 | Linear warmup steps |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--muon_lr` | 0.02 | Learning rate (Muon, if available) |
| `--weight_decay` | 0.1 | AdamW weight decay |
| `--grad_clip` | 1.0 | Gradient clipping norm |
| `--seed` | 42 | Random seed |
| `--seq_len` | 1024 | Sequence length (truncation) |
| `--val_split` | 0.05 | Validation split fraction |
| `--eval_every` | 500 | Validate every N steps |
| `--sample_every` | 1000 | Generate samples every N steps |
| `--ckpt_every` | 5000 | Checkpoint every N steps |
| `--log_every` | 10 | Console log every N steps |
| `--no_compile` | False | Skip torch.compile |
| `--no_wandb` | False | Disable WandB logging |
| `--eval_only` | False | Run eval only, no training |
| `--resume` | None | Path to checkpoint to resume from |
| `--wandb_project` | bibo-bench | WandB project name |
| `--wandb_name` | baseline-50m | WandB run name |

---

## Editing BiBoConfig with %%writefile

On Kaggle/Jupyter, you can't edit Python files with a GUI. Use `%%writefile` cell magic to overwrite files directly from a notebook cell.

### Writing a Custom BiBoConfig

```python
%%writefile /tmp/my_bibo_config.py
"""Custom BiBo config for benchmarking."""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.configuration_bibo import BiBoConfig
from src.modeling_bibo import BiBoForCausalLM

# ── Custom config ───────────────────────────────────────────
MY_CONFIG = BiBoConfig(
    vocab_size=81000,
    hidden_size=256,                 # Smaller than baseline (320)
    intermediate_size=768,           # 3x hidden
    num_hidden_layers=8,             # Fewer layers
    num_attention_heads=4,
    num_key_value_heads=2,           # GQA 2:1
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — PolyGLU layout (not baseline)
    polyglu_expert_multiplier=2,     # 2 groups = 6 experts (SiLU, ReLU², Tanh × 2)
    special_expert_pairs=1,          # + Identity + Zero = 8 total
    num_experts_per_tok=3,           # Top-3 routing
    moe_intermediate_size=256,
    use_shared_expert=True,
    shared_expert_type="mlp",        # "mlp" or "conv"
    # Router
    router_type="mlp",               # "mlp" or "conv"
    router_lambda=1.5,               # Sharper routing
    router_noise=0.5,
    bias_update_threshold=100_000,
    bias_update_factor=1e-2,
    # Other
    tie_word_embeddings=True,
)

# ── Verify ──────────────────────────────────────────────────
if __name__ == "__main__":
    model = BiBoForCausalLM(MY_CONFIG)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Config OK — {total:.2f}M params")
    print(f"  Experts: {MY_CONFIG.num_routed_experts} routed + {MY_CONFIG.num_shared_experts} shared")
    print(f"  Top-K: {MY_CONFIG.num_experts_per_tok}")
    print(f"  Router: {MY_CONFIG.router_type}, lambda={MY_CONFIG.router_lambda}")
```

Then run it:
```python
!python /tmp/my_bibo_config.py
```

### Overwriting bench/config.py Directly

To replace the baseline config used by `bench/train.py`:

```python
%%writefile bench/config.py
"""
BiBo Benchmark — Custom Config
Edit the values below and re-run this cell to change the model.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configuration_bibo import BiBoConfig
from src.modeling_bibo import BiBoForCausalLM

# ═══════════════════════════════════════════════════════════════
# EDIT THIS BLOCK — all training uses this config
# ═══════════════════════════════════════════════════════════════

BIBO_50M_BASELINE = BiBoConfig(
    vocab_size=81000,               # QTK-81K tokenizer
    hidden_size=320,
    intermediate_size=1024,         # 3.2x hidden
    num_hidden_layers=12,           # first 2 dense, 10 MoE
    num_attention_heads=5,          # hidden/64
    num_key_value_heads=1,          # GQA 5:1 (aggressive)
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — baseline (no PolyGLU)
    polyglu_expert_multiplier=1,    # 1 group = 3 SiLU experts
    special_expert_pairs=0,         # No Identity/Zero
    num_experts_per_tok=2,          # Top-2 routing
    use_shared_expert=True,
    shared_expert_type="mlp",       # SwiGLU shared expert
    # Router
    router_type="mlp",
    router_lambda=1.0,
    router_noise=0.0,               # Disabled for bench
    bias_update_threshold=100_000,
    bias_update_factor=1e-2,
    # Other
    tie_word_embeddings=True,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    attention_bias=False,
)

# ═══════════════════════════════════════════════════════════════
# Don't touch below this line
# ═══════════════════════════════════════════════════════════════


def count_params(config):
    """Count total params for a BiBoConfig (quick, no training overhead)."""
    model = BiBoForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.model.embed_tokens.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    unique = total
    del model
    return {
        "total": total,
        "total_m": total / 1e6,
        "embed": embed,
        "lm_head": lm_head,
    }


def build_model(config=None):
    """Build BiBoForCausalLM from config. Default: BIBO_50M_BASELINE."""
    if config is None:
        config = BIBO_50M_BASELINE
    model = BiBoForCausalLM(config)
    return model, config


if __name__ == "__main__":
    stats = count_params(BIBO_50M_BASELINE)
    print(f"BiBo Baseline Config:")
    print(f"  Total params: {stats['total']:,} ({stats['total_m']:.2f}M)")
    print(f"  Embed params: {stats['embed']:,}")
    print(f"  LM head params: {stats['lm_head']:,}")
    print(f"  Hidden: {BIBO_50M_BASELINE.hidden_size}")
    print(f"  Layers: {BIBO_50M_BASELINE.num_hidden_layers}")
    print(f"  Experts: {BIBO_50M_BASELINE.num_routed_experts} routed + {BIBO_50M_BASELINE.num_shared_experts} shared")
    print(f"  Top-K: {BIBO_50M_BASELINE.num_experts_per_tok}")
```

After writing, verify it:
```python
!python bench/config.py
```

Then run training:
```python
!python bench/train.py --batch_size 8 --total_steps 1000 --no_wandb
```

### Quick Param Sweep with %%writefile

```python
# Cell 1: Define sweep configs
configs = [
    {"name": "small",   "hidden": 192, "layers": 6,  "heads": 3, "kv_heads": 1},
    {"name": "medium",  "hidden": 320, "layers": 10, "heads": 5, "kv_heads": 1},
    {"name": "large",   "hidden": 384, "layers": 12, "heads": 6, "kv_heads": 2},
]

# Cell 2: Generate and test each config
for cfg in configs:
    code = f'''
import sys, os
sys.path.insert(0, ".")
from src.configuration_bibo import BiBoConfig
from src.modeling_bibo import BiBoForCausalLM

config = BiBoConfig(
    vocab_size=81000,
    hidden_size={cfg["hidden"]},
    intermediate_size={cfg["hidden"] * 3},
    num_hidden_layers={cfg["layers"]},
    num_attention_heads={cfg["heads"]},
    num_key_value_heads={cfg["kv_heads"]},
    max_position_embeddings=2048,
    use_ssmax=True,
    polyglu_expert_multiplier=1,
    special_expert_pairs=0,
    num_experts_per_tok=2,
    use_shared_expert=True,
    shared_expert_type="mlp",
    router_type="mlp",
    tie_word_embeddings=True,
)
model = BiBoForCausalLM(config)
total = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{cfg['name']}: {{total:.2f}}M params")
'''
    with open("/tmp/test_config.py", "w") as f:
        f.write(code)
    !python /tmp/test_config.py
```

---

## Editing bench/config.py with %%writefile (PolyGLU Variant)

To test BiBo with the full PolyGLU layout instead of baseline SwiGLU:

```python
%%writefile bench/config.py
"""
BiBo Benchmark — PolyGLU Config
Full diverse expert layout: SiLU + ReLU² + Tanh + Identity + Zero
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configuration_bibo import BiBoConfig
from src.modeling_bibo import BiBoForCausalLM


BIBO_50M_BASELINE = BiBoConfig(
    vocab_size=81000,
    hidden_size=256,
    intermediate_size=768,
    num_hidden_layers=10,
    num_attention_heads=4,
    num_key_value_heads=2,           # GQA 2:1
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — PolyGLU (diverse experts)
    polyglu_expert_multiplier=2,     # 2 groups × 3 = 6 GLU experts
    special_expert_pairs=1,          # + Identity + Zero = 8 total
    num_experts_per_tok=3,           # Top-3 routing
    moe_intermediate_size=256,
    use_shared_expert=True,
    shared_expert_type="conv",       # CausalConv1D shared expert
    # Router
    router_type="conv",              # Conv router (context-aware)
    router_lambda=1.5,               # Sharper routing
    router_noise=0.5,                # Exploration noise
    bias_update_threshold=100_000,
    bias_update_factor=1e-2,
    # Other
    tie_word_embeddings=True,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    attention_bias=False,
)


def count_params(config):
    model = BiBoForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.model.embed_tokens.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    del model
    return {"total": total, "total_m": total / 1e6, "embed": embed, "lm_head": lm_head}


def build_model(config=None):
    if config is None:
        config = BIBO_50M_BASELINE
    model = BiBoForCausalLM(config)
    return model, config


if __name__ == "__main__":
    stats = count_params(BIBO_50M_BASELINE)
    print(f"BiBo PolyGLU Config:")
    print(f"  Total params: {stats['total']:,} ({stats['total_m']:.2f}M)")
    print(f"  Hidden: {BIBO_50M_BASELINE.hidden_size}")
    print(f"  Layers: {BIBO_50M_BASELINE.num_hidden_layers}")
    print(f"  Experts: {BIBO_50M_BASELINE.num_routed_experts} routed + {BIBO_50M_BASELINE.num_shared_experts} shared")
    print(f"  Top-K: {BIBO_50M_BASELINE.num_experts_per_tok}")
    print(f"  Router: {BIBO_50M_BASELINE.router_type}, lambda={BIBO_50M_BASELINE.router_lambda}")
    print(f"  Shared expert: {BIBO_50M_BASELINE.shared_expert_type}")
```

---

## Throughput & Memory Benchmarks

### Measure tokens/sec and VRAM

```python
import torch
import time
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

def bench_throughput(config, batch_size=8, seq_len=1024, warmup=5, steps=20, device="cuda"):
    model = BiBoForCausalLM(config).to(device).train()
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup (trigger torch.compile, CUDA kernels)
    for _ in range(warmup):
        out = model(x, labels=x)
        out.loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    t0 = time.time()
    for _ in range(steps):
        out = model(x, labels=x)
        out.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens = batch_size * seq_len * steps
    tps = total_tokens / elapsed
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params_m:.2f}M")
    print(f"Tokens/sec: {tps:,.0f}")
    print(f"Step time: {elapsed/steps*1000:.1f}ms")
    print(f"Peak VRAM: {peak_mem:.2f}GB")
    print(f"Batch: {batch_size}×{seq_len}")

    return tps, peak_mem

# Run it
from bench.config import BIBO_50M_BASELINE
bench_throughput(BIBO_50M_BASELINE, batch_size=4, seq_len=1024)
```

### Compare configs

```python
configs = {
    "baseline_50M": BIBO_50M_BASELINE,
    # Add more configs here
}

for name, cfg in configs.items():
    print(f"\n{'='*40}")
    print(f"Config: {name}")
    print(f"{'='*40}")
    bench_throughput(cfg, batch_size=4, seq_len=512)
```

---

## Full Training Run (Kaggle 2×T4)

### Cell 1: Setup

```python
!git clone https://github.com/IsNoobgrammer/BiBo.git
%cd BiBo
!pip install -qU transformers einops wandb bitsandbytes pyyaml seaborn datasets hf_transfer
```

### Cell 2: (Optional) Edit config with %%writefile

```python
%%writefile bench/config.py
# ... paste full config.py with your custom values ...
```

### Cell 3: Verify config

```python
!python bench/config.py
```

### Cell 4: Run training

```python
import os
os.environ['WANDB_API_KEY'] = 'your_key_here'  # or WANDB_MODE=disabled

!torchrun --nproc_per_node=2 bench/train.py \
    --batch_size 16 \
    --total_steps 50000 \
    --warmup_steps 1000 \
    --lr 3e-4 \
    --eval_every 500 \
    --sample_every 1000 \
    --seq_len 1024 \
    --wandb_project bibo-bench \
    --wandb_name baseline-50m-v1
```

### Cell 5: View results

```python
# Check final metrics
!ls -la bench/checkpoints/

# If WandB disabled, check console output above for:
#   Final val loss: X.XXXX
#   Final val perplexity: XX.XX
#   Target (<2.8): HIT/MISSED
```

### Quick Iteration (fewer steps)

```python
# Fast sanity check — 500 steps, no WandB
!python bench/train.py --batch_size 8 --total_steps 500 --no_wandb --no_compile

# Medium run — 5000 steps with WandB
!python bench/train.py --batch_size 16 --total_steps 5000 --eval_every 250
```

---

## Config Reference

### BiBoConfig Parameters (bench-relevant)

| Parameter | Baseline | PolyGLU | Description |
|-----------|----------|---------|-------------|
| `hidden_size` | 320 | 256 | Hidden dimension |
| `intermediate_size` | 1024 | 768 | Dense MLP intermediate |
| `num_hidden_layers` | 12 | 10 | Total layers (first 2 dense) |
| `num_attention_heads` | 5 | 4 | Query heads |
| `num_key_value_heads` | 1 | 2 | KV heads (GQA) |
| `polyglu_expert_multiplier` | 1 | 2 | Groups of 3 GLU experts |
| `special_expert_pairs` | 0 | 1 | (Identity, Zero) pairs |
| `num_experts_per_tok` | 2 | 3 | Top-K routing |
| `moe_intermediate_size` | 256 | 256 | Per-expert FFN size |
| `shared_expert_type` | "mlp" | "conv" | Shared expert type |
| `router_type` | "mlp" | "conv" | Router architecture |
| `router_lambda` | 1.0 | 1.5 | Logit normalization scaling |
| `router_noise` | 0.0 | 0.5 | Training exploration noise |

**Expert count formula:**
```
routed_experts = polyglu_expert_multiplier × 3 + special_expert_pairs × 2
# Baseline: 1 × 3 + 0 × 2 = 3
# PolyGLU:  2 × 3 + 1 × 2 = 8
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | Batch too large | Reduce `--batch_size` (16→8→4) |
| `torch.compile failed` | MoE routing breaks fullgraph | Use `--no_compile` |
| `ModuleNotFoundError: transformers` | Missing dep | `!pip install transformers einops` |
| Muon not found | Optional dep missing | Falls back to AdamW automatically |
| Config changes not taking effect | Old config cached | Re-run `%%writefile` cell before training |
| Only 1 GPU on Kaggle | Accelerator not set | Settings > Accelerator > GPU T4 x2 |
| WandB login prompt | No API key | Set `WANDB_API_KEY` or `--no_wandb` |
| `FileNotFoundError: data/` | Data not generated | Data loads from HF Hub automatically |
| val_loss stuck > 3.0 | Model too small or LR wrong | Increase `hidden_size` or try `--lr 1e-4` |
| NaN loss | LR too high or grad explosion | Reduce `--lr`, check `--grad_clip 1.0` |

---

*Last updated: 2026-05-27*
