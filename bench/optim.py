"""
BiBo Benchmark — Optimizer Setup

Muon (Keller Jordan / modded-nanogpt) for embeddings + AdamW for the rest.
Fallback to pure AdamW if Muon is not available.
"""

import math
import torch
import torch.optim as optim


# ─────────────────────────────────────────────────────────────
# Try importing Muon from modded-nanogpt
# ─────────────────────────────────────────────────────────────

HAS_MUON = False
Muon = None

try:
    from modded_nanogpt.muon import Muon
    HAS_MUON = True
except ImportError:
    pass

if not HAS_MUON:
    try:
        # Alternative import path
        import sys, os
        for p in ["modded-nanogpt", "../modded-nanogpt"]:
            if os.path.isdir(p):
                sys.path.insert(0, p)
        from muon import Muon
        HAS_MUON = True
    except ImportError:
        pass


def create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1):
    """
    Create optimizer: Muon for embeddings + lm_head, AdamW for everything else.
    Falls back to pure AdamW if Muon is not available.
    """
    if HAS_MUON:
        print("[optim] Using Muon + AdamW hybrid")
        # Separate param groups
        embed_params = []
        lm_head_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed_tokens" in name:
                embed_params.append(param)
            elif "lm_head" in name:
                lm_head_params.append(param)
            else:
                other_params.append(param)

        # Muon for 1D params (embeddings, lm_head)
        # AdamW for the rest (attention weights, MoE experts, norms)
        optimizer = Muon(
            lr=muon_lr,
            params=[
                {"params": embed_params, "lr": muon_lr},
                {"params": lm_head_params, "lr": muon_lr},
                {"params": other_params, "lr": lr, "betas": (0.9, 0.95), "weight_decay": weight_decay},
            ],
        )
    else:
        print("[optim] Muon not found, using pure AdamW")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

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
    print(f"Muon available: {HAS_MUON}")
    if HAS_MUON:
        print("  -> Will use Muon + AdamW hybrid")
    else:
        print("  -> Will use pure AdamW fallback")
