# Kaggle Ablation Guide — BiBo vs Qwen3MoE

> Run BiBo and Qwen3MoE side-by-side on 2xT4 GPUs. Compare routing behavior, expert utilization, and model quality on sorting and arithmetic tasks.

---

## Overview

The `misc/kaggle/multi_gpu/` directory is a self-contained ablation framework:

- **Parallel training** — BiBo on `cuda:0`, Qwen3MoE on `cuda:1` (one model per T4)
- **Two tasks** — sorting (ordinal) and arithmetic (chain-of-thought)
- **NTIL loss** — Numerical Token Integrity Loss (arXiv:2505.13077) with CCE + EMD + sequence-level components
- **Router analysis** — 18 publication-quality plots comparing routing behavior
- **Model quality analysis** — loss, accuracy, confidence calibration, length generalization

### File Inventory

| File | Purpose |
|------|---------|
| `config.yaml` | All hyperparameters (training, BiBo, Qwen3MoE) |
| `data.py` | Generate sorting task data |
| `data_arithmetic.py` | Generate arithmetic task data (multi-phase CoT) |
| `model_utils.py` | Shared config loading, model loading, routing extraction |
| `datasets.py` | `SequenceDataset`, `ArithmeticDataset`, bucketed/curriculum loaders |
| `losses.py` | `NTILLoss` — CCE + EMD + sequence-level |
| `trainer.py` | Training loop (NTIL, wandb, validation, OOD eval) |
| `train.py` | Orchestrator — spawns parallel training processes |
| `analyze_model.py` | Dispatcher — routes to sorting or arithmetic analyzer |
| `analyze_model_sorting.py` | Model quality evaluation on sorting task |
| `analyze_router.py` | Comprehensive router behavior analysis |
| `router_metrics.py` | Gini, entropy, CV, co-selection, specialization |
| `router_plots.py` | 18 router visualization functions |
| `sorting_metrics.py` | Sorting evaluation metrics |
| `sorting_plots.py` | Sorting quality plots |
| `plot_utils.py` | Shared plotting infrastructure (colors, style, save) |

---

## Kaggle Environment Setup

### Step 1 — Create the Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Create a new notebook
3. Enable **GPU** in Settings > Accelerator > **GPU T4 x2**
4. Turn **Internet ON** (needed for `pip install` and wandb)

### Step 2 — Clone the Repo

```python
!git clone https://github.com/IsNoobgrammer/BiBo.git
%cd BiBo
```

### Step 3 — Install Dependencies

```python
!pip install -qU transformers einops wandb bitsandbytes pyyaml seaborn
```

---

## Editing `config.yaml` on Kaggle

Kaggle notebooks don't let you open and edit files with a GUI. You have two options:

### Option A — `%%writefile` Magic (Recommended)

Use the `%%writefile` cell magic to overwrite `config.yaml` directly from a notebook cell. This is the simplest approach — just paste the entire YAML into a cell.

```python
%%writefile misc/kaggle/multi_gpu/config.yaml
# ============================================================
# Training Config — Multi-GPU (2xT4)
# ============================================================
training:
  seed: 48
  epochs: 1
  batch_size: 64
  vocab_size: 512
  pad_token_id: 0
  lr: 3.0e-4
  warmup_ratio: 0.10
  weight_decay: 0.01
  grad_clip: 1.0
  train_samples: 300000
  val_samples: 3000
  val_every_n_steps: 300
  optimizer: adamw_8bit
  num_workers: 2
  log_every_n_steps: 10
  wandb_project: "bibo-ablation"
  wandb_run_name: "bibo-vs-qwen3moe-sort-v1"

  # Task selection: "sort" or "arithmetic"
  task: "sort"

  # NTIL loss weights
  alpha_emd: 0.3
  alpha_seq: 0.1

  # Curriculum learning (disabled — using mixed dataset)
  curriculum: false
  curriculum_stages: [2, 64, 256]

  # Arithmetic task config (only used when task: "arithmetic")
  arithmetic:
    max_num: 500
    operators: ["+", "-", "*", "//"]
    buckets:
      - [3, 7]
      - [9, 16]
      - [19, 30]
      - [35, 50]
    ood_buckets:
      - [8, 8]
      - [17, 18]
      - [31, 33]
      - [52, 55]
      - [60, 60]

# ============================================================
# BiBo Config (~10M) — PolyGLU layout
# ============================================================
bibo:
  device: "cuda:0"
  vocab_size: 512
  hidden_size: 256
  intermediate_size: 512
  num_hidden_layers: 6
  num_attention_heads: 4
  num_key_value_heads: 4
  polyglu_expert_multiplier: 2
  special_expert_pairs: 1
  num_experts_per_tok: 3
  moe_intermediate_size: 256
  max_position_embeddings: 256
  use_ssmax: true
  shared_expert_type: "mlp"  # "mlp" (SwiGLU, like Qwen) or "conv" (CausalConv1D)
  router_type: "mlp"
  router_lambda: 1.0
  router_noise: 0
  bias_update_threshold: 2400
  mlp_only_layers: [0]
  tie_word_embeddings: true

# ============================================================
# Qwen3MoE Config (~10M)
# ============================================================
qwen3moe:
  device: "cuda:1"
  vocab_size: 512
  hidden_size: 256
  intermediate_size: 512
  num_hidden_layers: 6
  num_attention_heads: 4
  num_key_value_heads: 4
  num_experts: 8
  num_experts_per_tok: 3
  moe_intermediate_size: 256
  max_position_embeddings: 256
  decoder_sparse_step: 1
  shared_expert_intermediate_size: 256
  mlp_only_layers: [0]
  tie_word_embeddings: true
```

**To modify a parameter:** edit the value in the cell and re-run it. The file gets overwritten instantly.

### Option B — Python Script to Write YAML

If you prefer to modify individual values programmatically (e.g., from a form or slider), write a Python cell that loads, patches, and saves the YAML:

```python
import yaml

cfg_path = "misc/kaggle/multi_gpu/config.yaml"

# Load existing config
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Modify values
cfg['training']['task'] = 'arithmetic'
cfg['training']['lr'] = 1e-4
cfg['training']['epochs'] = 2
cfg['bibo']['router_lambda'] = 2.0
cfg['bibo']['router_type'] = 'conv'

# Save back
with open(cfg_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("Config updated.")
```

This approach is better when you're sweeping parameters or building a parameter grid.

### Option C — Direct `yaml.dump` from Dict

For a clean-slate config written entirely in Python:

```python
import yaml

config = {
    'training': {
        'seed': 42,
        'epochs': 2,
        'batch_size': 64,
        'vocab_size': 512,
        'task': 'arithmetic',
        # ... rest of params
    },
    'bibo': {
        'device': 'cuda:0',
        'hidden_size': 256,
        # ... rest of params
    },
    'qwen3moe': {
        'device': 'cuda:1',
        'hidden_size': 256,
        # ... rest of params
    },
}

with open("misc/kaggle/multi_gpu/config.yaml", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

> **Tip:** After writing the config, verify it loaded correctly:
> ```python
> !cat misc/kaggle/multi_gpu/config.yaml
> ```

---

## Quickstart — Sorting Task

The sorting task trains models to sort sequences of tokens. Format: `[unsorted] [SEP] [sorted]`.

```python
# 1. Generate data
!python misc/kaggle/multi_gpu/data.py

# 2. Train both models in parallel
!python misc/kaggle/multi_gpu/train.py

# 3. Analyze model quality
!python misc/kaggle/multi_gpu/analyze_model.py

# 4. Analyze router behavior
!python misc/kaggle/multi_gpu/analyze_router.py
```

Outputs:
- Checkpoints: `misc/kaggle/multi_gpu/checkpoints/{bibo,qwen3moe}.pt`
- Metrics: `misc/kaggle/multi_gpu/metrics/`
- Plots: `misc/kaggle/multi_gpu/plots/`

---

## Quickstart — Arithmetic Task

The arithmetic task trains models on multi-phase chain-of-thought arithmetic. Format: `[expression] [SEP] [precedence-resolved] [SEP] [answer]`.

```python
# 1. Make sure config has task: "arithmetic"
#    (use %%writefile or Python yaml patch from above)

# 2. Generate data
!python misc/kaggle/multi_gpu/data_arithmetic.py

# 3. Train
!python misc/kaggle/multi_gpu/train.py

# 4. Analyze
!python misc/kaggle/multi_gpu/analyze_model.py
!python misc/kaggle/multi_gpu/analyze_router.py
```

### Arithmetic Data Format

Each sample has 3 phases:

| Phase | Content | Example |
|-------|---------|---------|
| Phase 1 | Raw expression | `3 * 4 + 2 - 6 // 3` |
| Phase 2 | Precedence resolved | `12 + 2 - 2` |
| Phase 3 | Final answer | `12` |

Token encoding:
- Numbers `1..500` map to tokens `1..500`
- Token `0` = PAD
- Token `511` = SEP (phase separator)
- Token `510` = ADD (+)
- Token `509` = SUB (-)
- Token `508` = MUL (*)
- Token `507` = DIV (//)

Labels are masked before the first SEP — the model only predicts phases 2 and 3.

---

## Config Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 48 | Random seed for reproducibility |
| `epochs` | 1 | Number of training epochs |
| `batch_size` | 64 | Batch size per model |
| `vocab_size` | 512 | Vocabulary size |
| `lr` | 3e-4 | Learning rate |
| `warmup_ratio` | 0.10 | Fraction of steps for linear warmup |
| `weight_decay` | 0.01 | AdamW weight decay |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `train_samples` | 300000 | Total training samples (split across buckets) |
| `val_samples` | 3000 | Total validation samples |
| `val_every_n_steps` | 300 | Validate every N steps |
| `optimizer` | adamw_8bit | Uses bitsandbytes 8-bit AdamW if available |
| `task` | "sort" | Task type: `"sort"` or `"arithmetic"` |
| `alpha_emd` | 0.3 | NTIL EMD weight (set 0 for arithmetic) |
| `alpha_seq` | 0.1 | NTIL sequence-level weight |
| `curriculum` | false | Enable curriculum learning (short to long) |
| `curriculum_stages` | [2, 64, 256] | Sequence lengths for curriculum buckets |

### BiBo Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | cuda:0 | GPU device |
| `hidden_size` | 256 | Hidden dimension |
| `intermediate_size` | 512 | Dense MLP intermediate size |
| `num_hidden_layers` | 6 | Transformer layers |
| `num_attention_heads` | 4 | Query heads |
| `num_key_value_heads` | 4 | KV heads (GQA if < query heads) |
| `polyglu_expert_multiplier` | 2 | Groups of 3 GLU experts (SiLU, ReLU^2, Tanh) |
| `special_expert_pairs` | 1 | Pairs of (Identity, Zero) experts |
| `num_experts_per_tok` | 3 | Top-K routing |
| `moe_intermediate_size` | 256 | Per-expert intermediate size |
| `use_ssmax` | true | SSMax query scaling |
| `shared_expert_type` | "mlp" | Shared expert type: `"mlp"` (SwiGLU, like Qwen) or `"conv"` (CausalConv1D) |
| `router_type` | "mlp" | Router architecture: `"mlp"` or `"conv"` |
| `router_lambda` | 1.0 | Logit normalization scaling |
| `router_noise` | 0 | Exploration noise (0 = disabled) |
| `bias_update_threshold` | 2400 | Tokens between heuristic bias updates |
| `mlp_only_layers` | [0] | Layers using dense MLP instead of MoE |

**Expert count formula:**
```
routed_experts = polyglu_expert_multiplier * 3 + special_expert_pairs * 2
# Default: 2 * 3 + 1 * 2 = 8
```

### Qwen3MoE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | cuda:1 | GPU device |
| `num_experts` | 8 | Total homogeneous MLP experts |
| `num_experts_per_tok` | 3 | Top-K routing |
| `moe_intermediate_size` | 256 | Per-expert intermediate size |
| `shared_expert_intermediate_size` | 256 | Shared expert size |
| `decoder_sparse_step` | 1 | MoE every N layers (1 = every layer) |

---

## NTIL Loss

**Numerical Token Integrity Loss** (arXiv:2505.13077) combines three components:

```
total = CCE + alpha_emd * EMD + alpha_seq * seq_loss
```

| Component | What It Does | When Useful |
|-----------|-------------|-------------|
| **CCE** | Standard cross-entropy | Always (primary gradient) |
| **EMD** | Wasserstein-1 on predicted distribution vs one-hot target. Penalizes "off by 1" less than "off by 10". | Ordinal tokens (sorting). Disabled for arithmetic (`alpha_emd=0`). |
| **Sequence-level** | L1 between expected predicted values and target values. Captures "whole sequence" correctness. | Both tasks. |

**For arithmetic:** EMD is disabled because operator tokens (ADD, SUB, MUL, DIV) have no ordinal relationship. Only CCE + sequence-level loss is used.

---

## Pipeline Architecture

```
config.yaml
    |
    v
data.py / data_arithmetic.py    --> data/*.npy
    |
    v
train.py (orchestrator)
    |--- trainer.py (train_worker 'bibo')   --> cuda:0
    |--- trainer.py (train_worker 'qwen3moe') --> cuda:1
    |
    v
checkpoints/{bibo,qwen3moe}.pt
metrics/{bibo,qwen3moe}_metrics.json
    |
    v
analyze_model.py (dispatcher)
    |--- analyze_model_sorting.py   (if task=sort)
    |--- analyze_model_arithmetic.py (if task=arithmetic)
    |
    v
analyze_router.py
    |--- router_metrics.py (Gini, entropy, CV)
    |--- router_plots.py (18 plot types)
    |
    v
plots/*.png
metrics/router_analysis.json
```

---

## Analysis Scripts

### `analyze_model.py` — Model Quality

Dispatches to task-specific analysis based on `config.yaml` task field.

**Sorting analysis** (`analyze_model_sorting.py`):
- Evaluates on seq_lens [8, 32, 64, 96, 128, 192]
- Token accuracy, full sequence accuracy, position-wise accuracy
- Confidence calibration (correct vs wrong predictions)
- Length generalization (red = unseen lengths)
- Detailed token-by-token prediction display

### `analyze_router.py` — Router Behavior

Generates 18 plot types comparing BiBo vs Qwen3MoE routing:

| Plot | What It Shows |
|------|---------------|
| Expert usage sweep | Usage across batch sizes and seq_lens |
| Comparative usage | Side-by-side BiBo vs Qwen expert selection |
| Confidence distribution | Violin plots of top-1 weight and weight spread |
| Co-selection matrix | Which expert pairs are selected together |
| Position type routing | Routing for unsorted vs sorted token regions |
| Specialization radar | Per-expert position specialization |
| Token-expert heatmap | x=token position, y=layer, color=expert |
| Confidence evolution | Smoothed top-1 weight over positions |
| Entropy evolution | Routing entropy over positions |
| Load balance summary | Gini, entropy, CV comparison |
| Weight rank distribution | Weight by expert rank (1st, 2nd, 3rd) |
| Routing diversity | Effective expert count and utilization |
| Routing stability | Jaccard similarity across samples |
| Expert type analysis | PolyGLU deep dive (SiLU vs ReLU^2 vs Tanh) |
| Special expert analysis | Identity + Zero usage |
| Expert switching rate | How often top-1 changes between tokens |
| Grand summary | Single-page dashboard |
| Per-layer weight KDE | Weight distribution per layer |

---

## Using wandb

Training logs to [Weights & Biases](https://wandb.ai) by default. Set your API key:

```python
import os
os.environ['WANDB_API_KEY'] = 'your_key_here'
```

Or disable wandb entirely:

```python
os.environ['WANDB_MODE'] = 'disabled'
```

Logged metrics (per model):
- `{model}/loss`, `{model}/cce`, `{model}/emd_raw`, `{model}/seq_raw`
- `{model}/emd_scaled`, `{model}/seq_scaled`, `{model}/lr`
- `{model}/val_loss`, `{model}/val_acc`
- `{model}/ood_{bucket}_loss`, `{model}/ood_{bucket}_acc` (arithmetic only)

---

## Seed Override

Override the config seed from the command line:

```python
!python misc/kaggle/multi_gpu/train.py --seed 69
!python misc/kaggle/multi_gpu/data.py --seed 69
```

This is useful for running multiple ablations with different random initializations.

---

## Kaggle Notebook Tips

### Saving Outputs

Kaggle notebooks have persistent storage at `/kaggle/working/`. The repo clones to `/kaggle/working/BiBo/`, so all outputs (checkpoints, metrics, plots) persist across session restarts within the same notebook version.

### Viewing Plots

After analysis, plots are saved to `misc/kaggle/multi_gpu/plots/`. Display them inline:

```python
from IPython.display import Image, display
import glob

for path in sorted(glob.glob("misc/kaggle/multi_gpu/plots/*.png")):
    display(Image(filename=path))
```

### Downloading Results

To download checkpoints or plots:

```python
import shutil
shutil.make_archive('bibo_results', 'zip', 'misc/kaggle/multi_gpu')
```

Then download from the notebook output panel.

### Memory Management

T4 GPUs have 16GB VRAM each. If you hit OOM:
- Reduce `batch_size` (64 -> 32)
- Reduce `hidden_size` (256 -> 128)
- Reduce `train_samples` for faster iteration

### Session Recovery

Kaggle sessions can disconnect. To resume:
1. Re-clone the repo (or check if `/kaggle/working/BiBo` still exists)
2. Re-run the `%%writefile` cell to restore your config
3. Data generation is idempotent — re-running `data.py` overwrites existing files
4. Checkpoints in `checkpoints/` persist within the notebook version

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: transformers` | Missing dependency | `!pip install transformers einops` |
| `RuntimeError: CUDA out of memory` | Batch too large | Reduce `batch_size` in config |
| `FileNotFoundError: data/train_len_64.npy` | Data not generated | Run `data.py` or `data_arithmetic.py` first |
| `bitsandbytes` import warning | Not installed | `!pip install bitsandbytes` or ignore (falls back to AdamW) |
| Only 1 GPU detected | Accelerator not set | Settings > Accelerator > GPU T4 x2 |
| wandb login prompt | No API key | Set `WANDB_API_KEY` env var or `os.environ['WANDB_MODE'] = 'disabled'` |
| Config changes not taking effect | Old config cached | Re-run the `%%writefile` cell before training |
| `analyze_model_arithmetic.py` not found | File doesn't exist yet | Use sorting task, or implement the arithmetic analyzer |
