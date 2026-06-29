"""
BiBo Benchmark — Metrics & WandB Logging

MetricsCollector for model-internal metrics + WandB integration.
Captures router entropy, expert utilization, attention entropy,
per-layer norms, gradient norms, and MFU.
"""

import os
import time
import torch
import wandb


# ─────────────────────────────────────────────────────────────
# MetricsCollector — Forward hooks for model-internal metrics
# ─────────────────────────────────────────────────────────────

class MetricsCollector:
    """
    Collects model-internal metrics during forward/backward pass.

    Usage:
        collector = MetricsCollector(model)
        with collector.track():
            loss = model(input_ids, labels).loss
            loss.backward()
        metrics = collector.compute()
    """

    def __init__(self, model, enabled=True):
        self.enabled = enabled
        self._router_entropies = {}
        self._attn_entropies = {}
        self._hidden_norms = {}
        self._hooks = []
        self._active = False

        if not enabled:
            return

        self._register_hooks(model)

    def _register_hooks(self, model):
        """Register forward hooks on router, attention, and norm layers."""
        try:
            from src.modeling.ffn.router import BiBoMoERouter
            for name, module in model.named_modules():
                if isinstance(module, BiBoMoERouter):
                    h = module.register_forward_hook(self._make_router_hook(name))
                    self._hooks.append(h)
        except ImportError:
            pass

        try:
            from src.modeling.attn.base import BiBoAttention
            for name, module in model.named_modules():
                if isinstance(module, BiBoAttention):
                    h = module.register_forward_hook(self._make_attn_hook(name))
                    self._hooks.append(h)
        except ImportError:
            pass

    def _make_router_hook(self, layer_name):
        def hook(module, input, output):
            if not self._active:
                return
            try:
                if isinstance(output, tuple) and len(output) >= 2:
                    indices, weights = output[0], output[1]
                    n_experts = getattr(module, 'num_routed_experts', 8)
                    full_weights = torch.zeros(
                        weights.shape[0], weights.shape[1], n_experts,
                        device=weights.device
                    )
                    full_weights.scatter_(2, indices, weights)
                    entropy = -(full_weights * (full_weights + 1e-10).log()).sum(-1).mean()
                    self._router_entropies[layer_name] = entropy.item()
            except Exception:
                pass
        return hook

    def _make_attn_hook(self, layer_name):
        def hook(module, input, output):
            if not self._active:
                return
            try:
                if isinstance(output, tuple) and len(output) >= 1:
                    attn_output = output[0]
                    if attn_output is not None:
                        self._hidden_norms[layer_name] = attn_output.norm(dim=-1).mean().item()
            except Exception:
                pass
        return hook

    def track(self):
        """Context manager to enable metric collection."""
        class _Tracker:
            def __init__(self, collector):
                self.collector = collector
            def __enter__(self):
                self.collector._active = True
                self.collector.reset()
                return self
            def __exit__(self, *args):
                self.collector._active = False
        return _Tracker(self)

    def compute(self) -> dict:
        """Compute collected metrics."""
        metrics = {}
        if self._router_entropies:
            vals = list(self._router_entropies.values())
            metrics["router_entropy_mean"] = sum(vals) / len(vals)
            metrics["router_entropy_min"] = min(vals)
            metrics["router_entropy_max"] = max(vals)
        if self._attn_entropies:
            vals = list(self._attn_entropies.values())
            metrics["attn_entropy_mean"] = sum(vals) / len(vals)
        if self._hidden_norms:
            vals = list(self._hidden_norms.values())
            metrics["hidden_norm_mean"] = sum(vals) / len(vals)
        return metrics

    def reset(self):
        """Reset all collected metrics."""
        self._router_entropies.clear()
        self._attn_entropies.clear()
        self._hidden_norms.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────
# WandB Integration
# ─────────────────────────────────────────────────────────────

def init_wandb(config, project="bibo-bench", name="baseline", notes=""):
    """Initialize WandB run with proper metric definitions."""
    run = wandb.init(
        project=project,
        name=name,
        notes=notes,
        config=config,
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("bench/*", step_metric="val/step")
    wandb.define_metric("samples/step")
    wandb.define_metric("samples/*", step_metric="samples/step")
    return run


def log_train_metrics(step, loss, lr, grad_norm, tokens_per_sec, step_time, extra=None):
    """Log training metrics to WandB."""
    d = {
        "train/step": step,
        "train/loss": loss,
        "train/learning_rate": lr,
        "train/grad_norm": grad_norm,
        "train/tokens_per_sec": tokens_per_sec,
        "train/step_time": step_time,
        "train/gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
        "train/gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
    }
    if extra:
        for k, v in extra.items():
            d[f"train/{k}"] = v
    wandb.log(d)


def log_eval_metrics(step, eval_results, tokens=None):
    """Log eval results (val_loss, val_ppl, benchmarks). `tokens` = tokens_trained so far,
    logged as val/tokens_trained for the loss-vs-tokens (sample-efficiency) curve (#2)."""
    d = {
        "val/step": step,
        "val/loss": eval_results.get("val_loss", 0),
        "val/perplexity": eval_results.get("val_ppl", 0),
    }
    if tokens is not None:
        d["val/tokens_trained"] = tokens
    if eval_results.get("val_bpb") is not None:
        d["val/bits_per_byte"] = eval_results["val_bpb"]
    if "hellaswag" in eval_results:
        hs = eval_results["hellaswag"]
        d["bench/hellaswag_acc"] = hs["accuracy"]
        d["bench/hellaswag_acc_norm"] = hs.get("acc_norm", 0)
        d["bench/hellaswag_margin"] = hs.get("margin", 0)   # continuous signal (>0 prefers gold)
        d["bench/hellaswag_auc"] = hs.get("auc", 0)         # ranking AUC, 0.5 = chance
    # length-extrapolation bpb (#3): bench/bpb_L{len} + the degradation ratio vs train length
    if "length_extrap" in eval_results:
        le = eval_results["length_extrap"]
        for L, m in le.get("per_len", {}).items():
            if m.get("bpb") is not None:
                d[f"val/bpb_L{L}"] = m["bpb"]
            if m.get("loss") is not None:
                d[f"val/loss_L{L}"] = m["loss"]
        d["val/length_extrap_ratio"] = le.get("ratio", 0)
    wandb.log(d)


def log_samples(step, samples):
    """Log generated samples as WandB Table."""
    table = wandb.Table(columns=["prompt", "generated", "top1_prob", "entropy", "first_token"])
    for s in samples:
        table.add_data(s["prompt"], s["generated"][:200],
                       s.get("top1_prob"), s.get("entropy"), s.get("first_token"))
    log = {"samples/step": step, "samples/table": table}
    if samples and "top1_prob" in samples[0]:
        # scalar trend lines (table values aren't plottable over steps)
        log["samples/top1_prob"] = sum(s["top1_prob"] for s in samples) / len(samples)
        log["samples/entropy"] = sum(s["entropy"] for s in samples) / len(samples)
    wandb.log(log)


# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step, path, save_optimizer=False):
    """Save a checkpoint.

    MODEL-ONLY by default: compare.py and eval only load weights, so the optimizer state
    (fused fp32 Adam + Muon momentum) is dead weight that ~doubled the file (1.23GB -> ~0.44GB
    here) and helped fill /kaggle/working -> the torch.save iostream/ENOSPC crash. Pass
    save_optimizer=True only when you need to RESUME training.

    Atomic (write .tmp then os.replace) so a crash never leaves a half-written .pt, and
    NON-FATAL (a full disk warns + continues training instead of killing a long run)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Unwrap torch.compile (_orig_mod) AND DDP (.module) in any nesting order, else keys get
        # a `module.`/`_orig_mod.` prefix and a strict=False load silently leaves random init.
        inner = model
        for _ in range(3):
            nxt = getattr(inner, "_orig_mod", None) or getattr(inner, "module", None)
            if nxt is None:
                break
            inner = nxt
        blob = {"model": inner.state_dict(), "step": step}
        if save_optimizer:
            blob["optimizer"] = optimizer.state_dict()
            blob["scheduler"] = scheduler.state_dict() if scheduler else None
        tmp = path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)   # atomic; never leaves a partial .pt on crash
        print(f"    Checkpoint saved at step {step} ({'model+optim' if save_optimizer else 'model-only'})")
    except Exception as e:
        print(f"    [WARN] checkpoint save failed at step {step}: {e} — continuing training")
        try:
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
        except OSError:
            pass


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint. Returns step number."""
    if not os.path.exists(path):
        print(f"    No checkpoint at {path}, starting fresh")
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_to_load.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    step = ckpt.get("step", 0)
    print(f"    Loaded checkpoint from step {step}")
    return step


# ─────────────────────────────────────────────────────────────
# MFU Estimation
# ─────────────────────────────────────────────────────────────

def estimate_mfu(active_params, tokens_per_sec, device):
    """
    Estimate Model FLOPs Utilization.

    MFU = (tokens_per_sec * 6 * active_params) / (peak_tflops * 1e12)
    """
    flops_per_token = 6 * active_params
    achieved_flops = tokens_per_sec * flops_per_token

    # fp16 tensor-core peak (training runs in fp16 autocast). Using the fp32
    # peak here would understate MFU by ~8x on Turing/Ampere.
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() and "cuda" in str(device) else "cpu"
    if "T4" in gpu_name:
        peak_tflops = 65.0       # T4 fp16 tensor cores
    elif "3050" in gpu_name:
        peak_tflops = 18.0       # RTX 3050 laptop fp16 tensor cores
    elif "A100" in gpu_name:
        peak_tflops = 312.0      # A100 fp16 tensor cores
    elif "4090" in gpu_name:
        peak_tflops = 330.0      # RTX 4090 fp16 tensor cores
    else:
        peak_tflops = 65.0

    return achieved_flops / (peak_tflops * 1e12)


# ─────────────────────────────────────────────────────────────
# Throughput Meter
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
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_count(n):
    """Format a token count with K/M/B suffix (e.g. 1234567 -> '1.23M')."""
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{int(n)}"
