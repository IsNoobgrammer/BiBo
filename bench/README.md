# BiBo Benchmark — Unified Training Suite

Trains BiBo and Qwen3MoE with identical data pipelines for fair comparison.

## Quick Start

```bash
# BiBo baseline
bash bench/run.sh bench/configs/bibo.yaml

# Qwen3MoE baseline
bash bench/run.sh bench/configs/qwen3moe.yaml

# Ablation: no SSMax
bash bench/run.sh bench/configs/bibo_no_ssmax.yaml

# Dual-GPU comparison (1 GPU per model)
CUDA_VISIBLE_DEVICES=0 bash bench/run.sh bench/configs/bibo.yaml &
CUDA_VISIBLE_DEVICES=1 bash bench/run.sh bench/configs/qwen3moe.yaml &
wait
```

## Configs

| Config | Model | Params | Description |
|--------|-------|--------|-------------|
| `bibo.yaml` | BiBo | ~95M | PolyGLU (6 weighted + 2 special), GQA 4:2, SSMax |
| `bibo_no_ssmax.yaml` | BiBo | ~95M | Same but without SSMax attention |
| `qwen3moe.yaml` | Qwen3MoE | ~115M | 8 homogeneous SwiGLU experts, GQA 4:2 |

## What Gets Applied

Both models receive identical optimizations:
- **Liger RMSNorm** (8-9x faster)
- **Liger RoPE** (2-3x faster)
- **Liger Fused CrossEntropy** (at training loop level)
- **Dense MLP Triton** (fused SwiGLU forward + backward)
- **BiBo-only**: MoE fused GLU + weight scatter kernels

## Metrics Logged (WandB)

- **Training**: loss, lr, grad_norm, tokens/sec, step_time, GPU memory
- **Validation**: val_loss, perplexity
- **Benchmarks**: HellaSwag accuracy, ARC-Challenge accuracy
- **MoE internals**: router entropy, expert utilization, hidden norms
- **MFU**: Model FLOPs Utilization estimate

## Deterministic Training

Same seed + same data = same training curves. Required for fair comparison.
Uses `CUBLAS_WORKSPACE_CONFIG=:4096:8` and seeded data loaders.

## Dependencies

```
torch>=2.6.0, transformers>=4.50.0, datasets>=3.0.0
wandb>=0.18.0, hf_transfer>=0.1.0, bitsandbytes>=0.44.0
pyyaml>=6.0, liger-kernel, einops
```
