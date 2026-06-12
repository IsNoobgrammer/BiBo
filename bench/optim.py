"""
BiBo Benchmark — Optimizer Setup

Muon + AdamW-8bit split optimizer:
- Muon: Nesterov momentum with Newton-Schulz orthogonalized gradients for 2D weights
- AdamW-8bit: For 1D params (norms, biases) + embeddings + lm_head

Falls back to pure AdamW if bitsandbytes unavailable.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────
# Newton-Schulz orthogonalization
# ─────────────────────────────────────────────────────────────

def newton_schulz_iteration(G, num_iters=5):
    """
    Approximate matrix sign function for orthogonalization.
    From Keller Jordan / modded-nanogpt.
    """
    assert G.ndim == 2
    X = G / (G.norm() + 1e-7)
    for _ in range(num_iters):
        A = X @ X.T
        X = (3 * X - A @ X) / 2
    return X


# ─────────────────────────────────────────────────────────────
# Muon Optimizer
# ─────────────────────────────────────────────────────────────

class Muon(optim.Optimizer):
    """
    Muon optimizer — Nesterov momentum with orthogonalized gradients.

    For 2D weight matrices:
      - Compute gradient
      - Orthogonalize via Newton-Schulz iteration
      - Apply Nesterov momentum update

    For 1D params: falls back to AdamW.

    Reference: Keller Jordan, modded-nanogpt
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.0, adamw_params=None, adamw_lr=3e-4,
                 adamw_betas=(0.9, 0.95), adamw_wd=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.adamw_params = list(adamw_params) if adamw_params is not None else []
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_wd = adamw_wd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
            if not params:
                continue

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in params:
                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                g_orth = newton_schulz_iteration(grad)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g_orth)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g_orth)

                if group["nesterov"]:
                    update = g_orth + momentum * buf
                else:
                    update = buf

                p.add_(update, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────
# AdamW 8-bit fallback
# ─────────────────────────────────────────────────────────────

HAS_BNB = False
AdamW8bit = None

try:
    from bitsandbytes.optim import AdamW8bit as _AdamW8bit
    AdamW8bit = _AdamW8bit
    HAS_BNB = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# Optimizer factory
# ─────────────────────────────────────────────────────────────

def create_optimizer(model, cfg):
    """
    Create optimizer from config.

    Returns (optimizer, name_str) where name_str is for logging.
    """
    train_cfg = cfg["training"]
    optim_type = train_cfg.get("optimizer", "muon_adamw8bit")
    lr = train_cfg.get("lr", 3e-4)
    muon_lr = train_cfg.get("muon_lr", 0.02)
    weight_decay = train_decay if (train_decay := train_cfg.get("weight_decay", 0.1)) else 0.1

    # Separate 2D vs 1D params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Embeddings + lm_head always use AdamW (not Muon)
    embed_params = []
    other_2d_params = []
    for p in decay_params:
        if p.ndim == 2 and p.numel() > 10000:
            other_2d_params.append(p)
        else:
            embed_params.append(p)

    if optim_type == "muon_adamw8bit":
        if HAS_BNB:
            adamw = AdamW8bit(
                [{"params": no_decay_params, "weight_decay": 0.0},
                 {"params": embed_params, "weight_decay": weight_decay}],
                lr=lr, betas=(0.9, 0.95),
            )
        else:
            adamw = optim.AdamW(
                [{"params": no_decay_params, "weight_decay": 0.0},
                 {"params": embed_params, "weight_decay": weight_decay}],
                lr=lr, betas=(0.9, 0.95),
            )

        muon = Muon(
            other_2d_params,
            lr=muon_lr,
            momentum=0.95,
            weight_decay=weight_decay,
        )

        # Combine into a single optimizer-like object
        optimizer = _CombinedOptimizer(muon, adamw)
        name = "Muon+AdamW8bit" if HAS_BNB else "Muon+AdamW"

    elif optim_type == "adamw8bit":
        if HAS_BNB:
            optimizer = AdamW8bit(
                [{"params": decay_params, "weight_decay": weight_decay},
                 {"params": no_decay_params, "weight_decay": 0.0}],
                lr=lr, betas=(0.9, 0.95),
            )
            name = "AdamW8bit"
        else:
            optimizer = optim.AdamW(
                [{"params": decay_params, "weight_decay": weight_decay},
                 {"params": no_decay_params, "weight_decay": 0.0}],
                lr=lr, betas=(0.9, 0.95),
            )
            name = "AdamW"

    else:
        optimizer = optim.AdamW(
            [{"params": decay_params, "weight_decay": weight_decay},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95),
        )
        name = "AdamW"

    print(f"  Optimizer: {name}")
    return optimizer, name


class _CombinedOptimizer(optim.Optimizer):
    """Wraps Muon + AdamW as a single optimizer for gradient accumulation."""

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        # Build param_groups from both sub-optimizers
        all_groups = muon.param_groups + adamw.param_groups
        super().__init__(all_groups, {})

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

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
