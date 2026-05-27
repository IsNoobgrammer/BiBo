# BiBo Benchmark — Implementation Plan

## Goal

Build a training benchmark for BiBo MoE transformer. Target: < 2.8 val loss on language modeling with ~50M parameter baseline BiBo (MLP router, shared expert, uniform SwiGLU experts). Kaggle 2×T4, WandB logging, torch.compile, QTK-81K tokenizer for inference/samples.

---

## Phase 0: Setup & Infrastructure [S]

**Goal:** Project scaffolding, dependencies, environment verification.

### Tasks

- [ ] **0.1** Create `bench/` directory structure in BiBo repo
  - Files: `train.py`, `config.py`, `data.py`, `optim.py`, `eval.py`, `utils.py`, `run.sh`, `README.md`
  - Acceptance: all files exist with proper imports and docstrings

- [ ] **0.2** Write `run.sh` — Kaggle entry point
  ```bash
  #!/bin/bash
  pip install hf_transfer wandb datasets transformers
  pip install -e .  # Install BiBo from repo root
  cd bench && python train.py "$@"
  ```
  - Acceptance: runs without error on fresh Kaggle notebook

- [ ] **0.3** Verify BiBo model loads from `src/`
  ```python
  from src.modeling_bibo import BiBoForCausalLM
  from src.configuration_bibo import BiBoConfig
  model = BiBoForCausalLM(BiBoConfig(hidden_size=64, num_hidden_layers=2))
  print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
  ```
  - Acceptance: model initializes, param count prints

- [ ] **0.4** Verify QTK-81K tokenizer loads
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("fhai50032/QTK-81K")
  print(tokenizer.encode("Hello world"))
  ```
  - Acceptance: tokenizer loads, encode/decode works

- [ ] **0.5** Verify dataset loads
  ```python
  from datasets import load_dataset
  ds = load_dataset("tinycompany/Instruct-packed-2K-Context-tk-QTK-81K", split="train")
  print(ds[0].keys())
  ```
  - Acceptance: dataset loads, first item has `input_ids` field

---

## Phase 1: Config & Data Pipeline [M]

**Goal:** BiBoConfig for ~50M baseline, dataset loading with truncation to 1024.

### Tasks

- [ ] **1.1** Write `config.py` — BiBo baseline config
  ```python
  from src.configuration_bibo import BiBoConfig

  BIBO_50M_BASELINE = BiBoConfig(
      vocab_size=81000,
      hidden_size=320,
      intermediate_size=896,
      num_hidden_layers=10,
      num_attention_heads=5,
      num_key_value_heads=1,
      max_position_embeddings=2048,
      use_ssmax=True,
      # Baseline MoE (no PolyGLU)
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
  - Acceptance: `model = BiBoForCausalLM(config)` works, ~50M params
  - **NOTE:** Exact numbers may need tuning. Verify param count with `sum(p.numel())`.

- [ ] **1.2** Write `config.py` param count validator
  ```python
  def validate_param_count(config, target_m=50, tolerance=0.15):
      model = BiBoForCausalLM(config)
      actual = sum(p.numel() for p in model.parameters()) / 1e6
      assert abs(actual - target_m) / target_m < tolerance, f"{actual:.1f}M != {target_m}M"
  ```
  - Acceptance: prints exact param count, errors if >15% off target

- [ ] **1.3** Write `data.py` — dataset loading + truncation
  ```python
  def load_benchmark_data(seq_len=1024, val_split=0.05):
      ds = load_dataset("tinycompany/Instruct-packed-2K-Context-tk-QTK-81K", split="train")
      # Truncate each item to seq_len
      def truncate(example):
          ids = example["input_ids"][:seq_len]
          return {"input_ids": ids, "labels": ids[1:] + [-100]}  # shift for NLP
      ds = ds.map(truncate, remove_columns=ds.column_names)
      # Split train/val
      split = ds.train_test_split(test_size=val_split, seed=42)
      return split["train"], split["test"]
  ```
  - Acceptance: returns train/val datasets, each item is length 1024

- [ ] **1.4** Write `data.py` — DataLoader with proper collation
  ```python
  def create_dataloader(dataset, batch_size, num_workers=2):
      return DataLoader(
          dataset, batch_size=batch_size, shuffle=True,
          num_workers=num_workers, pin_memory=True,
          collate_fn=default_collate,  # or custom if needed
      )
  ```
  - Acceptance: DataLoader yields [B, 1024] tensors

---

## Phase 2: Optimizer & Scheduler [S]

**Goal:** Muon + AdamW hybrid optimizer with cosine LR schedule.

### Tasks

- [ ] **2.1** Write `optim.py` — Muon import with fallback
  ```python
  try:
      from modded_nanogpt.muon import Muon
      HAS_MUON = True
  except ImportError:
      HAS_MUON = False

  def create_optimizer(model, lr=3e-4, muon_lr=0.02):
      if HAS_MUON:
          # Muon for embedding + lm_head, AdamW for the rest
          embed_params = list(model.model.embed_tokens.parameters())
          lm_head_params = list(model.lm_head.parameters())
          other_params = [p for n, p in model.named_parameters()
                          if "embed_tokens" not in n and "lm_head" not in n]
          optimizer = Muon(
              lr=muon_lr,
              params=[
                  {"params": embed_params, "lr": muon_lr},
                  {"params": lm_head_params, "lr": muon_lr},
                  {"params": other_params, "lr": lr},
              ],
          )
      else:
          optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
      return optimizer
  ```
  - Acceptance: optimizer creates without error, Muon fallback works

- [ ] **2.2** Write LR scheduler — cosine with warmup
  ```python
  def create_scheduler(optimizer, warmup_steps, total_steps):
      def lr_lambda(step):
          if step < warmup_steps:
              return step / warmup_steps
          progress = (step - warmup_steps) / (total_steps - warmup_steps)
          return 0.5 * (1 + math.cos(math.pi * progress))
      return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
  ```
  - Acceptance: scheduler returns correct LR at step 0, warmup_steps, total_steps

---

## Phase 3: Training Loop [L]

**Goal:** Main training script with FSDP2, torch.compile, WandB logging.

### Tasks

- [ ] **3.1** Write `utils.py` — WandB initialization
  ```python
  def init_wandb(config, project="bibo-bench", name="baseline-50m"):
      wandb.init(project=project, name=name, config=vars(config))
      wandb.define_metric("train/step")
      wandb.define_metric("train/*", step_metric="train/step")
      wandb.define_metric("val/step")
      wandb.define_metric("val/*", step_metric="val/step")
  ```

- [ ] **3.2** Write `utils.py` — checkpointing
  ```python
  def save_checkpoint(model, optimizer, step, path):
      torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)

  def load_checkpoint(model, optimizer, path):
      ckpt = torch.load(path)
      model.load_state_dict(ckpt["model"])
      optimizer.load_state_dict(ckpt["optimizer"])
      return ckpt["step"]
  ```

- [ ] **3.3** Write `train.py` — main training loop
  ```python
  def train(args):
      # 1. Init distributed (if multi-GPU)
      dist.init_process_group("nccl")
      local_rank = int(os.environ["LOCAL_RANK"])
      torch.cuda.set_device(local_rank)

      # 2. Load config + model
      config = BIBO_50M_BASELINE
      model = BiBoForCausalLM(config).to(local_rank)

      # 3. FSDP2 wrap
      for layer in model.model.layers:
          fully_shard(layer)
      fully_shard(model)

      # 4. torch.compile
      model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

      # 5. Optimizer + scheduler
      optimizer = create_optimizer(model)
      scheduler = create_scheduler(optimizer, args.warmup_steps, args.total_steps)

      # 6. Data
      train_ds, val_ds = load_benchmark_data(seq_len=1024)
      train_loader = create_dataloader(train_ds, args.batch_size)

      # 7. WandB
      init_wandb(args)

      # 8. Training loop
      for step, batch in enumerate(train_loader):
          input_ids = batch["input_ids"].to(local_rank)
          labels = batch["labels"].to(local_rank)

          with torch.autocast("cuda", dtype=torch.float16):
              outputs = model(input_ids=input_ids, labels=labels)
              loss = outputs.loss

          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()

          # Log
          wandb.log({"train/loss": loss.item(), "train/step": step, ...})

          # Eval + samples every M steps
          if step % args.eval_every == 0:
              val_loss = evaluate(model, val_ds)
              wandb.log({"val/loss": val_loss, "val/step": step})

          if step % args.sample_every == 0:
              samples = generate_samples(model, tokenizer)
              wandb.log({"samples/text": wandb.Table(...), "samples/step": step})
  ```

- [ ] **3.4** Write `train.py` — arg parsing
  ```python
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--total_steps", type=int, default=50000)
  parser.add_argument("--warmup_steps", type=int, default=1000)
  parser.add_argument("--eval_every", type=int, default=500)
  parser.add_argument("--sample_every", type=int, default=1000)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--muon_lr", type=float, default=0.02)
  parser.add_argument("--seed", type=int, default=42)
  ```

---

## Phase 4: Eval & Sample Generation [M]

**Goal:** Validation loss computation + coherent text generation using QTK-81K tokenizer.

### Tasks

- [ ] **4.1** Write `eval.py` — validation loss
  ```python
  @torch.no_grad()
  def evaluate(model, val_ds, batch_size=32):
      model.eval()
      total_loss = 0.0
      n_batches = 0
      loader = create_dataloader(val_ds, batch_size, shuffle=False)
      for batch in loader:
          outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
          total_loss += outputs.loss.item()
          n_batches += 1
      model.train()
      return total_loss / n_batches
  ```

- [ ] **4.2** Write `eval.py` — sample generation
  ```python
  @torch.no_grad()
  def generate_samples(model, tokenizer, prompts=None, max_new_tokens=100):
      if prompts is None:
          prompts = ["The meaning of life is", "Once upon a time", "In 2026,"]
      model.eval()
      results = []
      for prompt in prompts:
          input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
          output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9)
          text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
          results.append({"prompt": prompt, "generated": text})
      model.train()
      return results
  ```
  - Acceptance: generates coherent(ish) text, logged to WandB

- [ ] **4.3** Write WandB sample logging
  ```python
  def log_samples_to_wandb(samples, step):
      table = wandb.Table(columns=["prompt", "generated"])
      for s in samples:
          table.add_data(s["prompt"], s["generated"])
      wandb.log({"samples/table": table, "samples/step": step})
  ```

---

## Phase 5: Integration & Kaggle Run [M]

**Goal:** End-to-end run on Kaggle 2×T4, verify < 2.8 val loss.

### Tasks

- [ ] **5.1** Write `run.sh` — full Kaggle setup
  ```bash
  #!/bin/bash
  set -e
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  pip install hf_transfer wandb datasets transformers
  pip install -e .

  # Login to WandB (set WANDB_API_KEY in Kaggle secrets)
  wandb login

  # Launch with torchrun for multi-GPU
  torchrun --nproc_per_node=2 bench/train.py \
      --batch_size 16 \
      --total_steps 50000 \
      --warmup_steps 1000 \
      --eval_every 500 \
      --sample_every 1000 \
      --lr 3e-4
  ```

- [ ] **5.2** Test single-GPU mode (for local dev)
  ```bash
  python bench/train.py --batch_size 8 --total_steps 1000
  ```

- [ ] **5.3** Test multi-GPU mode (Kaggle)
  ```bash
  torchrun --nproc_per_node=2 bench/train.py --batch_size 16
  ```

- [ ] **5.4** Verify WandB dashboard shows all metrics
  - train/loss curve decreasing
  - val/loss < 2.8 at convergence
  - samples/table shows improving coherence
  - tokens_per_sec stable
  - GPU memory < 16GB per T4

- [ ] **5.5** Write `README.md` with:
  - How to run (single GPU, multi-GPU, Kaggle)
  - Config options (batch_size, lr, total_steps, etc.)
  - Expected results (val loss target, time estimate)
  - Troubleshooting (OOM, compile errors, Muon fallback)

---

## Complexity Summary

| Phase | Tasks | Complexity | Estimated Time |
|-------|-------|------------|----------------|
| Phase 0: Setup | 5 | S | 30 min |
| Phase 1: Config + Data | 4 | M | 1 hour |
| Phase 2: Optimizer | 2 | S | 30 min |
| Phase 3: Training Loop | 4 | L | 2 hours |
| Phase 4: Eval + Samples | 3 | M | 1 hour |
| Phase 5: Integration | 5 | M | 1.5 hours |
| **Total** | **23** | — | **~6 hours** |

---

## Dependencies

```
torch>=2.6.0
transformers>=4.50.0
datasets>=3.0.0
wandb>=0.18.0
hf_transfer>=0.1.0
numpy
```

Optional:
```
modded-nanogpt  # For Muon optimizer (fallback: pure AdamW)
```

---

## Success Criteria

- [ ] BiBo ~50M model trains on 2×T4
- [ ] Val loss < 2.8 on language modeling
- [ ] torch.compile works (no errors, measurable speedup)
- [ ] WandB dashboard shows all metrics
- [ ] Sample generation shows improving coherence over training
- [ ] Multi-GPU scaling works (FSDP2 or DDP)
- [ ] Full run completes in < 4 hours on Kaggle
