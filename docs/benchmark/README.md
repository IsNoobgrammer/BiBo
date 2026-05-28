# BiBo Benchmark Documentation

This folder contains all benchmarking-related documentation for BiBo.

## Contents

| File | Description |
|------|-------------|
| [benchmarking.md](benchmarking.md) | How to run benchmarks: smoke tests, throughput, Kaggle training |
| [kernel_benchmarks_moe.md](kernel_benchmarks_moe.md) | MoE Triton kernel results (GLU fusion, 1.51-1.65x speedup) |
| [kernel_benchmarks_conv.md](kernel_benchmarks_conv.md) | Conv Triton kernel results (permute+act+gate fusion, 1.34-1.41x training) |

## Quick Reference

### Combined Triton Kernel Stack

| Kernel | Source | Speedup | Memory |
|--------|--------|---------|--------|
| RMSNorm | Liger-Kernel | 8-9x (isolated) | In-place |
| RoPE | Liger-Kernel | 2-3x (isolated) | Eliminates intermediate |
| MoE GLU Activation | Custom Triton | 1.51x (full model) | -57 MB (14%) |
| Conv Permute+Act+Gate | Custom Triton | 1.41x (training) | -144 MB (8%) |

### Full Model Training Speedup (all kernels combined)

| Tokens | MLP Router + MLP Shared | Conv Router + Conv Shared |
|--------|------------------------|--------------------------|
| 128 | 1.25x | 1.34x |
| 512 | 1.01x | 0.99x |
| 1024 | 1.65x | 1.37x |
| 2048 | — | 1.41x |

### How to Enable

```python
from src.kernels import (
    patch_bibo_with_triton,          # RMSNorm + RoPE
    patch_moe_with_triton,           # MoE GLU activation
    patch_conv_router_with_triton,   # Conv router
    patch_conv_expert_with_triton,   # Conv shared expert
)

model = BiBoForCausalLM(config).cuda()
patch_bibo_with_triton(model)
patch_moe_with_triton(model)
patch_conv_router_with_triton(model)
patch_conv_expert_with_triton(model)
```

Or just run training (auto-enabled):
```bash
python bench/train.py --batch_size 8 --total_steps 1000
# Disable with: --no_triton
```
