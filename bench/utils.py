"""
BiBo Benchmark — Utilities

WandB logging, checkpointing, metrics, helpers.
"""

import os
import time
import math
import torch
import wandb


# ─────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────

def init_wandb(config, project="bibo-bench", name="baseline-50m", notes=""):
    """Initialize WandB run with proper metric definitions."""
    run = wandb.init(
        project=project,
        name=name,
        notes=notes,
        config=config,
    )
    # Define step metrics so WandB plots correctly
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("samples/step")
    wandb.define_metric("samples/*", step_metric="samples/step")
    return run


def log_train_metrics(step, loss, lr, grad_norm, tokens_per_sec, step_time):
    """Log training metrics to WandB."""
    wandb.log({
        "train/step": step,
        "train/loss": loss,
        "train/learning_rate": lr,
        "train/grad_norm": grad_norm,
        "train/tokens_per_sec": tokens_per_sec,
        "train/step_time": step_time,
        "train/gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
        "train/gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
    })


def log_val_metrics(step, val_loss, val_ppl):
    """Log validation metrics to WandB."""
    wandb.log({
        "val/step": step,
        "val/loss": val_loss,
        "val/perplexity": val_ppl,
    })


def log_samples(step, samples):
    """Log generated samples as WandB Table."""
    table = wandb.Table(columns=["prompt", "generated"])
    for s in samples:
        table.add_data(s["prompt"], s["generated"])
    wandb.log({
        "samples/step": step,
        "samples/table": table,
    })


# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Handle compiled models
    state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
    }, path)
    print(f"[ckpt] Saved checkpoint at step {step} -> {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint. Returns step number."""
    if not os.path.exists(path):
        print(f"[ckpt] No checkpoint found at {path}, starting from scratch")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_load.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt.get("step", 0)
    print(f"[ckpt] Loaded checkpoint from step {step}")
    return step


# ─────────────────────────────────────────────────────────────
# Timing & Metrics
# ─────────────────────────────────────────────────────────────

class ThroughputMeter:
    """Track tokens/sec throughput."""

    def __init__(self, warmup=3):
        self.warmup = warmup
        self.count = 0
        self.total_tokens = 0
        self.start_time = None

    def reset(self):
        self.total_tokens = 0
        self.start_time = time.time()

    def update(self, n_tokens):
        self.count += 1
        if self.count <= self.warmup:
            self.reset()
            return
        if self.start_time is None:
            self.start_time = time.time()
        self.total_tokens += n_tokens

    def tokens_per_sec(self):
        if self.start_time is None or self.total_tokens == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.total_tokens / elapsed


def format_time(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
