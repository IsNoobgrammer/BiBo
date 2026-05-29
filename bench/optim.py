"""
BiBo Benchmark — Optimizer Setup

AdamW (8-bit if available) for all parameters.
Proper weight decay grouping: 2D weights get decay, 1D params (norms, biases) don't.

Install: pip install bitsandbytes
"""

import math
import torch
import torch.optim as optim


# ─────────────────────────────────────────────────────────────
# Try importing bitsandbytes 8-bit AdamW
# ─────────────────────────────────────────────────────────────

HAS_BNB = False
AdamW8bit = None

try:
    from bitsandbytes.optim import AdamW8bit
    HAS_BNB = True
except ImportError:
    pass


def create_optimizer(model, lr=3e-4, weight_decay=0.1, **kwargs):
    """
    Create optimizer with correct parameter grouping:
      - AdamW (lr=3e-4, wd=0.1)  → 2D weights (attn, MLP, experts, embed, head)
      - AdamW (lr=3e-4, wd=0.0)  → 1D params (RMSNorm, biases, scalars)

    Uses 8-bit AdamW if bitsandbytes available, else pure AdamW.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if HAS_BNB:
        print("[optim] Using 8-bit AdamW (bitsandbytes)")
        optimizer = AdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
    else:
        print("[optim] Using pure AdamW")
        optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    return optimizer


def create_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == "__main__":
    print(f"bitsandbytes available: {HAS_BNB}")
    if HAS_BNB:
        print("  -> Will use 8-bit AdamW (saves ~40% optimizer VRAM)")
    else:
        print("  -> Will use pure AdamW fallback")
