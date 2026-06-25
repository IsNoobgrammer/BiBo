# Kaggle Guide — Training & Benchmarking BiBo vs Qwen3MoE

Run the `bench/` suite on Kaggle's free **2×T4** to train BiBo and Qwen3MoE on identical data and
compare them. **Loss is the primary metric** (MFU / tokens-sec are secondary). Qwen is deliberately
sized *larger* so a BiBo loss win is conservative.

> CLI/local reference for the same suite: [`bench/README.md`](../bench/README.md).
> Kernel details: [`triton_kernels.md`](triton_kernels.md).

---

## TL;DR — fastest path

1. New Kaggle notebook → **Settings → Accelerator → GPU T4 ×2** → **Internet ON**.
2. Add your WandB key under **Add-ons → Secrets** as `WANDB_API_KEY` (or skip — runs offline).
3. Paste these cells and run:

```python
# Cell 1 — clone
!git clone https://github.com/IsNoobgrammer/BiBo.git
%cd BiBo
```
```python
# Cell 2 — WandB key (optional; skip to run offline)
import os
os.environ["WANDB_API_KEY"] = "your_key_here"   # or leave unset → WANDB_MODE=disabled
```
```bash
%%bash
# Cell 3 — train BiBo and Qwen SIMULTANEOUSLY, one model per T4.
# Must be a single %%bash cell: in Jupyter each `!line` is its own shell, so
# backgrounding with `!... &` then `!wait` on separate lines does NOT work.
CUDA_VISIBLE_DEVICES=0 bash bench/run.sh bench/configs/bibo.yaml     > bibo.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 bash bench/run.sh bench/configs/qwen3moe.yaml > qwen.log 2>&1 &
wait
```

`run.sh` installs deps, logs into WandB, detects the (single visible) GPU, and launches
`bench/train.py`. The cell blocks until both finish — watch live progress in WandB, or open a second
notebook and `!tail -f bibo.log`. `os.environ["WANDB_API_KEY"]` from Cell 2 is inherited by the
`%%bash` subshell.

> **Prefer a GUI?** Open `bench/notebook.ipynb` ("BiBo Benchmark Studio") on Kaggle — collapsible
> config widgets + live console for dual-GPU launch. Same engine, point-and-click.

---

## What it runs

- **Two models, identical pipeline** (same data, seed, optimizer, LR schedule, kernels) for a fair
  comparison. Determinism via `CUBLAS_WORKSPACE_CONFIG=:4096:8` + seeded loaders.
- **BiBo** → `cuda:0`, **Qwen3MoE** → `cuda:1` (one model per T4, trained at the same time).
- **Primary metric = loss** (val loss / perplexity). Secondary: HellaSwag & ARC-Challenge accuracy,
  MFU, tokens/sec.
- **Qwen is handicapped *larger*** (~85.9M total / 44.6M active vs BiBo ~72M / 39M) so a BiBo loss
  win can't be explained by size. A Qwen win, by contrast, is size-confounded — give the larger Qwen
  enough steps to converge or it's just undertrained.
- Both models get the **fused kernels + fused cross-entropy** (the only path that fits the loss at
  this vocab). `torch.compile` is **on for T4** (it's broken on the local dev GPU) — confirm both
  runs log `torch.compile OK`.

---

## Configs

| Config | Model | Size | Notes |
|--------|-------|------|-------|
| `bench/configs/bibo.yaml` | BiBo | ~72M / 39M active | 10L, H320, PolyGLU 6+2 special, GQA 5:1, SSMax, shared MLP expert |
| `bench/configs/qwen3moe.yaml` | Qwen3MoE | ~85.9M / 44.6M active | 8 homogeneous SwiGLU experts, widened to handicap-larger |
| `bench/configs/bibo_no_ssmax.yaml` | BiBo | ~72M | ablation: SSMax off |

Defaults (both): `batch_size 16`, `seq_len 1024`, `total_steps 50000`, `lr 5e-4`, `warmup 100`,
`adamw8bit`, `compile/triton/fsdp on`. Edit the YAML, or override per-run with CLI flags (below) —
no need to rewrite the file.

---

## Common overrides (CLI flags)

Append to either `run.sh` invocation — e.g. `bash bench/run.sh bench/configs/bibo.yaml --total_steps 3000 --no_wandb`:

| Flag | Effect |
|------|--------|
| `--total_steps N` | shorter run (e.g. 3000 for a smoke test) |
| `--batch_size N` / `--grad_accum N` | fit memory / change effective batch |
| `--seq_len N` | sequence length |
| `--lr F` / `--muon_lr F` | learning rate(s) |
| `--eval_every N` | benchmark cadence |
| `--no_compile` | disable `torch.compile` (use if a T4 compile error appears) |
| `--no_triton` | eager baseline (no fused kernels) — for kernel A/B |
| `--no_wandb` | disable logging |
| `--grad_checkpoint` | trade ~1.2–1.4× time for 0.34–0.65× memory (large batch/seq) |
| `--eval_only --resume PATH` | evaluate a checkpoint without training |

---

## Reading results

- **WandB** (`bibo-bench` project) is the main view: `loss`, `val_loss`, `perplexity`, `lr`,
  `grad_norm`, `tokens/sec`, `step_time`, GPU mem, MFU, and `hellaswag` / `arc_challenge` accuracy.
  Put BiBo and Qwen on the same chart — the loss curves are the headline.
- **Checkpoints / metrics** are written under the run's output dir (see `bench/train.py`); download
  via the Kaggle output panel or `shutil.make_archive`.
- A converged comparison needs the larger Qwen to actually finish converging — short runs flatter
  BiBo unfairly. Use the full `total_steps` (or enough that both val-loss curves have plateaued)
  before drawing conclusions.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Only 1 GPU detected | Settings → Accelerator → **GPU T4 ×2** |
| `torch.compile` error on T4 | add `--no_compile` (kernels still apply); report it — compile should work on T4 |
| CUDA OOM | lower `--batch_size` (16→8) and raise `--grad_accum`, or add `--grad_checkpoint` |
| WandB login prompt / blocks | set `WANDB_API_KEY` (Secrets) or run with `--no_wandb` |
| `liger-kernel` / dep missing | `run.sh` pip-installs deps; if it fails, `!pip install liger-kernel transformers datasets einops bitsandbytes` |
| One model far behind | expected mid-run — Qwen is larger; let it converge before comparing |
| Want a single GPU | run one config without `CUDA_VISIBLE_DEVICES`; `run.sh` falls back to single-GPU |

---

## Notes

- Don't compare a half-converged larger Qwen to a converged BiBo — the only clean conclusion is the
  **BiBo-wins** direction (smaller model, lower loss). Frame results accordingly.
- For the *kernel* benchmarks (speed/memory of individual Triton kernels, the E2E kernel matrix),
  see [`kernel_benchmark_report.md`](kernel_benchmark_report.md) and
  [`benchmark/benchmarking.md`](benchmark/benchmarking.md) — that's a different axis from this
  model-quality comparison.
