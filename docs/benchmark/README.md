# BiBo Benchmark Documentation

This folder contains all benchmarking-related documentation for BiBo.

## Contents

| File | Description |
|------|-------------|
| [benchmarking.md](benchmarking.md) | How to run benchmarks: smoke tests, throughput, Kaggle training |
| [../kernel_benchmark_report.md](../kernel_benchmark_report.md) | **All current kernel results** (fp16, `do_bench`): MoE, dense MLP, conv, fused-CE, RMSNorm, XSA, E2E matrix, checkpointing |

> Earlier per-kernel result docs (`kernel_benchmarks_moe.md`, `kernel_benchmarks_conv.md`) were
> removed — their numbers came from a broken fp32 3-sample harness on a non-standard config. Use
> the report above; all numbers there are fp16 via `triton.testing.do_bench`.

## Quick Reference — current results (fp16, `triton.testing.do_bench`)

| Kernel | Source | Speedup vs eager | Notes |
|--------|--------|------------------|-------|
| RMSNorm | Liger | dim320 7.8–8.5× fwd / 2.7× fwd+bwd; QK dim64 2.3× fwd / ~1.0× fwd+bwd | upcasts fp32 |
| RoPE | Liger | fused | eliminates intermediate |
| MoE experts | Custom Triton (`patch_moe_auto`) | 1.42× fwd / 1.40× fwd+bwd | grouped path ~2–2.5× fwd at 4k–8k tok |
| Dense MLP | Liger SwiGLU | 1.19× fwd / 1.23× fwd+bwd | — |
| Conv **router** | Custom Triton (`patch_conv_router_with_triton`) | ~2.5× fwd+bwd (large batch) | conv **expert** kernel NOT used (0.41× slower) |
| Fused-linear CE | Custom Triton (default) | 2.1× fwd; enabling at 16k tok | mem flat ~0.5 GB |
| XSA | Custom Triton (`patch_xsa_with_triton`) | 2.2–2.6× fwd / 2.5–3.2× fwd+bwd | mem 0.68–0.76×; in-kernel GQA broadcast |

### Full-model E2E (BiBo, fp16, RTX 3050, ≤2048 tok — clean regime)

| Tokens | baseline (ms) | all-kernels (ms) | all-kernels speedup |
|--------|---------------|------------------|---------------------|
| 1024 | 254 | 197 | 1.29× |
| 2048 | 394 | 310 | 1.27× |

`all + fused-CE` is the recommended training config (kernels ~1.3×; fused-CE flattens step memory to
0.66–0.74× and is the only path that runs the 16k-token step). >2048 tok swaps on the 4 GB card →
validate on T4.

### How to Enable

```python
from src.kernels import (
    patch_bibo_with_liger,           # RMSNorm + RoPE (Liger; alias: patch_bibo_with_triton)
    patch_moe_auto,                  # MoE experts (per-expert / grouped auto-dispatch)
    patch_conv_router_with_triton,   # Conv router (only if router_type="conv")
    patch_xsa_with_triton,           # XSA rejection
)

model = BiBoForCausalLM(config).cuda()
patch_bibo_with_liger(model)
patch_moe_auto(model)
patch_xsa_with_triton()
# patch_conv_router_with_triton(model)   # only when router_type="conv"
# NOTE: the conv shared-expert kernel is intentionally NOT applied (slower than PyTorch ops).
```

Or just run training (kernels + fused-CE auto-enabled):
```bash
python bench/train.py --batch_size 8 --total_steps 1000
# Disable with: --no_triton
```
