# BiBo Benchmark — Technical Design

## Overview

Training benchmark for BiBo MoE transformer on language modeling. Goal: reach < 2.8 val loss with a ~50M parameter baseline BiBo (MLP router, shared expert, uniform SwiGLU experts — no PolyGLU, no special experts). Runs on Kaggle 2×T4, multi-GPU via FSDP2/DistributedDataParallel.

---

## Hardware Target

| Spec | Value |
|------|-------|
| GPU | 2× NVIDIA T4 (16GB each) |
| VRAM | 32GB total |
| Compute | 7.5 (no bf16 native — use fp16 or bf16 via amp) |
| Strategy | FSDP2 or DDP across 2 GPUs |

**Why T4x2 instead of T4x1:** Tests multi-GPU scaling for BiBo MoE. Single T4 is 16GB — enough for 50M but doesn't stress-test distributed MoE routing.

---

## Model Config: Baseline BiBo ~50M

Baseline = no PolyGLU, no special experts. Standard SwiGLU experts + MLP router + shared expert.

```python
BiBoConfig(
    vocab_size=81000,           # QTK-81K tokenizer
    hidden_size=384,            
    intermediate_size=1024,     # SwiGLU inner dim
    num_hidden_layers=10,       # 10 layers (first+last dense, 8 MoE)
    num_attention_heads=6,      
    num_key_value_heads=2,      # GQA 3:1
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — baseline (no PolyGLU)
    polyglu_expert_multiplier=1,   # Minimal: 1 group = 3 experts
    special_expert_pairs=0,        # No Identity/Zero experts
    num_experts_per_tok=2,         # Top-2 routing
    use_shared_expert=True,        # Shared expert ON
    shared_expert_type="mlp",      # SwiGLU shared expert
    # Router
    router_type="mlp",             # MLP router (baseline)
    gate_type="sigmoid",           # DeepSeek-V3 sigmoid gating
    load_balance_strategy="bias",  # Heuristic bias updates
    # Optimizations
    tie_word_embeddings=True,      # Save params: embed ↔ lm_head tied
)
```

**Param count estimate (~50M):**
- Embedding: 81000 × 384 = 31.1M (tied with lm_head)
- Attention (per layer): Q(384→384) + K(384→128) + V(384→128) + O(384→384) ≈ 295K × 10 = 2.95M
- MoE layer (8 layers): Router(384→5) + 3 SwiGLU experts(384→1024→384 × 2) + shared expert ≈ 4.8M × 8 = 38.4M... that's too high

Let me recalculate with proper SwiGLU param counts:
- Single SwiGLU expert: gate(384→1024) + up(384→1024) + down(1024→384) = 393K + 393K + 393K ≈ 1.18M
- 3 experts × 1.18M = 3.54M per MoE layer
- Shared expert: 1.18M per layer
- Total per MoE layer: 3.54 + 1.18 + router(0.002) ≈ 4.72M
- 8 MoE layers: 37.8M
- 2 dense layers (first+last): 2 × 1.18M = 2.36M
- Attention: 0.295M × 10 = 2.95M
- Norms: ~15K
- **Total: 31.1M + 37.8M + 2.36M + 2.95M + 0.015M ≈ 74.2M**

That's too big. Let me scale down:

```python
# Revised: ~50M target
BiBoConfig(
    vocab_size=81000,
    hidden_size=256,            # Smaller
    intermediate_size=768,      # 3x hidden
    num_hidden_layers=8,        # 8 layers
    num_attention_heads=4,
    num_key_value_heads=2,      # GQA 2:1
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — baseline
    polyglu_expert_multiplier=1,
    special_expert_pairs=0,
    num_experts_per_tok=2,
    use_shared_expert=True,
    shared_expert_type="mlp",
    # Router
    router_type="mlp",
    gate_type="sigmoid",
    load_balance_strategy="bias",
    tie_word_embeddings=True,
)
```

**Revised param count:**
- Embedding: 81000 × 256 = 20.7M (tied)
- SwiGLU expert: gate(256→768) + up(256→768) + down(768→256) = 196K + 196K + 196K ≈ 590K
- 3 experts × 590K = 1.77M per MoE layer
- Shared expert: 590K per layer
- Router: 256→5 = 1.3K
- Per MoE layer: 1.77 + 0.59 + 0.001 ≈ 2.36M
- 6 MoE layers (layers 1-6): 14.16M
- 2 dense layers (0, 7): 2 × 0.59M = 1.18M
- Attention per layer: Q(256→256) + K(256→128) + V(256→128) + O(256→256) = 65K + 32K + 32K + 65K ≈ 196K
- 8 layers attention: 1.57M
- Norms: ~4K
- **Total: 20.7M + 14.16M + 1.18M + 1.57M + 0.004M ≈ 37.6M**

Still under. Let me try hidden=320, intermediate=896, 10 layers:

```python
BiBoConfig(
    vocab_size=81000,
    hidden_size=320,
    intermediate_size=896,      # ~2.8x hidden
    num_hidden_layers=10,
    num_attention_heads=5,      # hidden/64
    num_key_value_heads=1,      # GQA 5:1 (aggressive)
    max_position_embeddings=2048,
    use_ssmax=True,
    polyglu_expert_multiplier=1,
    special_expert_pairs=0,
    num_experts_per_tok=2,
    use_shared_expert=True,
    shared_expert_type="mlp",
    router_type="mlp",
    gate_type="sigmoid",
    load_balance_strategy="bias",
    tie_word_embeddings=True,
)
```

**Param count:**
- Embedding: 81000 × 320 = 25.9M (tied)
- SwiGLU expert: 2×(320→896) + (896→320) = 2×286K + 286K ≈ 860K
- Per MoE layer: 3×860K + 860K(shared) + 0.2K(router) = 3.44M
- 8 MoE layers: 27.5M
- 2 dense layers: 2×860K = 1.72M
- Attention per layer: Q(320→320)+K(320→64)+V(320→64)+O(320→320) = 102K+20K+20K+102K ≈ 245K
- 10 layers: 2.45M
- **Total: 25.9M + 27.5M + 1.72M + 2.45M ≈ 57.6M**

Close enough to 50M. Good baseline config. User can tweak from here.

---

## Dataset

**Source:** `tinycompany/Instruct-packed-2K-Context-tk-QTK-81K`
- Pre-tokenized with QTK-81K tokenizer
- Packed to 2048 token sequences
- No re-tokenization needed — just load and truncate to 1024

**Loading:**
```python
from datasets import load_dataset
ds = load_dataset("tinycompany/Instruct-packed-2K-Context-tk-QTK-81K", split="train")
# Each item: {"input_ids": [int, ...], "attention_mask": [int, ...], "labels": [int, ...]}
# Truncate to 1024: input_ids[:1024], labels[1:1025]  (shift for next-token prediction)
```

**Train/Val split:** 95% train, 5% val (deterministic split from HF dataset).

---

## Tokenizer

**Source:** `fhai50032/QTK-81K`
- 81K vocab BPE tokenizer
- NOT used during training (data is pre-tokenized)
- Used ONLY for:
  1. Inference: encode prompts, decode model output
  2. Progress samples: generate text at checkpoints to measure coherence

**Loading:**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("fhai50032/QTK-81K", use_fast=True)
```

---

## Optimizer

Import from Keller Jordan's ecosystem (modded-nanogpt) or equivalent industry-standard optimizers:

| Optimizer | Source | Use Case |
|-----------|--------|----------|
| **Muon** | `modded-nanogpt` / `kellerjordan` | Primary — momentum-orthogonalized, proven on NanoGPT speedruns |
| **AdamW** | `torch.optim` | Fallback / comparison baseline |
| **SOAP** | `kellerjordan` | Alternative — Shampoo-like, good for MoE |
| **ScheduleFree** | `kellerjordan` | LR schedule-free variant |

**Plan:** Use Muon for embedding/lm_head + AdamW for everything else (standard modded-nanogpt split). If Muon import fails, fall back to pure AdamW.

---

## Training Loop

```
┌─────────────────────────────────────┐
│           Training Loop             │
├─────────────────────────────────────┤
│ 1. Load dataset (HF datasets)      │
│ 2. Create DataLoader (packed seqs)  │
│ 3. Init model (BiBoForCausalLM)     │
│ 4. torch.compile(model)             │
│ 5. Wrap with FSDP2 (2×T4)          │
│ 6. Optimizer: Muon + AdamW          │
│ 7. Training loop:                   │
│    ├─ Forward pass                  │
│    ├─ Compute loss (cross-entropy)  │
│    ├─ Backward pass                 │
│    ├─ Optimizer step                │
│    ├─ LR scheduler step             │
│    ├─ Log to WandB (loss, lr, etc) │
│    └─ Every N steps: eval + sample  │
│ 8. Final eval + full WandB summary  │
└─────────────────────────────────────┘
```

---

## WandB Logging

**Metrics logged per step:**
- `train/loss` — cross-entropy loss
- `train/learning_rate` — current LR
- `train/tokens_per_sec` — throughput
- `train/gpu_memory_allocated` — VRAM usage
- `train/gpu_memory_reserved` — VRAM reserved
- `train/step_time` — seconds per step
- `train/grad_norm` — gradient norm

**Metrics logged per eval (every N steps):**
- `val/loss` — validation loss
- `val/perplexity` — exp(val_loss)

**Progress samples (every M steps):**
- `samples/text` — generated text from fixed prompts
- `samples/token_count` — tokens generated
- Logged as WandB Table for easy comparison

---

## torch.compile Strategy

```python
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
```

- `mode="reduce-overhead"`: maximizes CUDA graph reuse (best for fixed-shape inputs)
- `fullgraph=False`: required for MoE (data-dependent routing in experts)
- Warmup: run 3-5 forward passes before timing to trigger compilation
- On T4: compile uses Triton backend (no inductor issues with fp16)

---

## Multi-GPU Strategy

**FSDP2** (Fully Sharded Data Parallel v2):
- Shards model parameters across 2 GPUs
- Each GPU holds a shard — reduces per-GPU memory
- Compatible with torch.compile
- Better than DDP for MoE models (DDP replicates full model)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# or torch 2.x FSDP2:
from torch.distributed._composable.fsdp import fully_shard
```

**Alternative: DDP** — simpler, but each GPU holds full model copy. Fine for 50M params (fits in 16GB easily). Use DDP if FSDP2 causes issues.

---

## Edge Cases

| Case | Handling |
|------|----------|
| OOM on T4 | Reduce batch size, enable gradient checkpointing |
| torch.compile failure | Fall back to eager mode, log warning |
| Muon import failure | Fall back to pure AdamW |
| Dataset loading failure | Retry with exponential backoff (HF hub flaky) |
| NaN loss | Skip step, log warning, reduce LR |
| WandB failure | Continue training, log to local CSV backup |
| Single GPU fallback | Detect GPU count, use DDP for 2, single for 1 |

---

## File Structure

```
bench/
├── train.py           # Main training script
├── config.py          # BiBo benchmark config (50M baseline)
├── data.py            # Dataset loading + preprocessing
├── optim.py           # Optimizer setup (Muon + AdamW)
├── eval.py            # Eval loop + sample generation
├── utils.py           # Logging, checkpointing, helpers
├── run.sh             # Kaggle entry point
└── README.md          # How to run
```
