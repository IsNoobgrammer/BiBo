"""
BiBo Benchmark — Optimizer Setup

Muon for 2D hidden layer weights (orthogonalized updates via Newton-Schulz).
AdamW (8-bit if available) for embeddings, LM head, and all 1D params.

Muon is ONLY for 2D hidden matrices (attn projections, MLP weights, expert weights).
Embeddings, LM head, biases, norms → AdamW. This matches the recipe used by:
  - Kimi K2 (MuonClip, arXiv:2507.20534)
  - Moonlight (arXiv:2502.16982)
  - modded-nanogpt (KellerJordan/Muon)
  - NVIDIA NeMo Emerging Optimizers

Install: pip install muon-optimizer bitsandbytes
"""

import math
import torch
import torch.optim as optim
import torch.distributed as dist


# ─────────────────────────────────────────────────────────────
# Try importing Muon
# ─────────────────────────────────────────────────────────────

HAS_MUON = False

try:
    from muon import SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
    HAS_MUON = True
except ImportError:
    pass


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


def create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1):
    """
    Create optimizer with correct parameter grouping:
      - Muon (lr=0.02, wd=0.01) → 2D hidden weights (attn Q/K/V/O, MLP, experts, router)
      - AdamW (lr=3e-4, wd=0.1)  → embeddings, lm_head (2D but input/output layers)
      - AdamW (lr=3e-4, wd=0.0)  → 1D params (RMSNorm, biases, scalars)

    Muon orthogonalizes updates via Newton-Schulz for 2D matrices only.
    Embeddings/lm_head have different optimization dynamics (modular norm theory).
    Falls back to 8-bit AdamW → pure AdamW if Muon unavailable.

    Uses SingleDeviceMuonWithAuxAdam for single-GPU, MuonWithAuxAdam for distributed.
    """
    if HAS_MUON:
        is_distributed = dist.is_initialized()
        variant = "distributed" if is_distributed else "single-device"
        print(f"[optim] Using Muon ({variant}) + AdamW hybrid")

        muon_params = []        # 2D hidden layer weights → Muon
        adamw_decay = []        # embed, lm_head, other 2D non-hidden → AdamW with decay
        adamw_no_decay = []     # 1D params (norms, biases, scalars) → AdamW no decay

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Embeddings and LM head → AdamW (even though 2D)
            if "embed_tokens" in name or "lm_head" in name:
                adamw_decay.append(param)
            # 1D params: norms, biases, scalars → AdamW no decay
            elif param.ndim < 2:
                adamw_no_decay.append(param)
            # Muon only works on 2D matrices with reasonable size
            # Skip: 3D+ tensors (fused experts), tiny params (ssmax_scale, router)
            elif param.ndim == 2 and min(param.shape) >= 64:
                muon_params.append(param)
            # Everything else 2D but small, or 3D+ → AdamW with decay
            else:
                adamw_decay.append(param)

        print(f"[optim]   Muon params: {len(muon_params)} tensors")
        print(f"[optim]   AdamW decay params: {len(adamw_decay)} tensors (embed/head)")
        print(f"[optim]   AdamW no-decay params: {len(adamw_no_decay)} tensors (norms/biases)")

        # Build param groups with use_muon flag
        param_groups = [
            # Group 0: 2D hidden weights → Muon
            {"params": muon_params, "lr": muon_lr, "momentum": 0.95,
             "weight_decay": 0.01, "use_muon": True},
            # Group 1: embed + lm_head → AdamW with decay
            {"params": adamw_decay, "lr": lr, "betas": (0.9, 0.95),
             "eps": 1e-10, "weight_decay": weight_decay, "use_muon": False},
            # Group 2: 1D params → AdamW no decay
            {"params": adamw_no_decay, "lr": lr, "betas": (0.9, 0.95),
             "eps": 1e-10, "weight_decay": 0.0, "use_muon": False},
        ]

        if is_distributed:
            optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    elif HAS_BNB:
        print("[optim] Using 8-bit AdamW (bitsandbytes)")
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW8bit(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )
    else:
        print("[optim] Using pure AdamW")
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
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
    print(f"bitsandbytes available: {HAS_BNB}")
    if HAS_MUON:
        print("  -> Will use Muon (2D hidden weights) + AdamW (embed/head/1D)")
    elif HAS_BNB:
        print("  -> Will use 8-bit AdamW (saves ~40% optimizer VRAM)")
    else:
        print("  -> Will use pure AdamW fallback")
