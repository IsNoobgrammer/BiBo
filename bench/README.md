# BiBo Benchmark — Language Model Training

## Quick Start

```bash
# Single GPU (local dev)
python bench/train.py

# Multi-GPU (Kaggle 2×T4)
torchrun --nproc_per_node=2 bench/train.py

# Or use the Kaggle entry point
bash bench/run.sh
```

## What This Does

Trains a ~50M parameter baseline BiMo (MLP router, shared expert, uniform SwiGLU experts) on language modeling using the QTK-81K tokenizer dataset. Target: < 2.8 val loss.

## Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `--batch_size` | 32 | Per-GPU batch size |
| `--total_steps` | 50000 | Total training steps |
| `--warmup_steps` | 1000 | Linear warmup steps |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--muon_lr` | 0.02 | Learning rate (Muon) |
| `--seq_len` | 1024 | Sequence length (truncated from 2048) |
| `--eval_every` | 500 | Steps between validation |
| `--sample_every` | 1000 | Steps between sample generation |
| `--no_compile` | false | Skip torch.compile |
| `--no_wandb` | false | Disable WandB logging |

## Dependencies

```
torch>=2.6.0
transformers>=4.50.0
datasets>=3.0.0
wandb>=0.18.0
hf_transfer>=0.1.0
```

Optional: `modded-nanogpt` for Muon optimizer (falls back to AdamW)

## Expected Results

- **Val loss target:** < 2.8
- **Time:** ~3-4 hours on Kaggle 2×T4
- **WandB dashboard:** loss curves, perplexity, throughput, generated samples

## Architecture

- ~50M params: hidden=320, 10 layers, 5 attention heads (GQA 5:1)
- 3 uniform SwiGLU routed experts + 1 shared SwiGLU expert per MoE layer
- MLP router with sigmoid gating (DeepSeek-V3 style)
- SSMax attention scaling
- First and last layers are dense MLP (not MoE)

## File Structure

```
bench/
├── config.py      # Model config (BiBoConfig ~50M)
├── data.py        # Dataset loading + truncation
├── optim.py       # Muon + AdamW optimizer
├── eval.py        # Eval + sample generation
├── utils.py       # WandB, checkpointing, helpers
├── train.py       # Main training loop
├── run.sh         # Kaggle entry point
└── README.md      # This file
```
