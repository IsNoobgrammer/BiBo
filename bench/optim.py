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
    Orthogonalize G (drive its singular values toward 1) via the TUNED QUINTIC Newton-Schulz
    from Keller Jordan / modded-nanogpt — the same one Moonlight/Kimi use. The quintic coeffs
    (3.4445, -4.7750, 2.0315) push SVs into ~[0.7, 1.3] in 5 iters; a plain cubic (3X-AX)/2
    UNDER-orthogonalizes (SVs ~0.4) so the downstream 0.2·√max RMS scaling lands at ~0.085
    instead of 0.2 and the LR is mis-attributed. Compute in fp32 for stability.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)                     # (1, A, B) — unify 2D + batched (E, A, B) paths
    # normalize each slice by its Frobenius norm (per-expert for the batched case)
    X = X / (X.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1))
    transposed = X.size(1) > X.size(2)         # iterate on the smaller Gram matrix
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(num_iters):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(G.dtype)


# ─────────────────────────────────────────────────────────────
# Muon Optimizer
# ─────────────────────────────────────────────────────────────

class Muon(optim.Optimizer):
    """
    Muon optimizer — Moonlight/Kimi recipe (arXiv:2502.16982, 2507.20534).

    For 2D weight matrices, per step:
      1. momentum on the RAW gradient, THEN Newton-Schulz orthogonalize (canonical order)
      2. CONSISTENT-RMS scale: update *= 0.2 * sqrt(max(fan_in, fan_out)).
         The orthogonalized matrix has element-RMS ~= 1/sqrt(max(A,B)); this scaling forces
         element-RMS ~= 0.2 (AdamW's typical update RMS) for EVERY matrix shape, so a single
         AdamW-range LR (~2e-4..4e-4) is correctly attributed across all layers.
      3. decoupled weight decay (AdamW-style: p *= 1 - lr*wd), NOT folded into the grad.

    Without (2)+(3) the effective step is shape-dependent and ~6-10x hotter than this recipe —
    that's why the old lr=0.02 (raw modded-nanogpt regime) was mis-scaled. See AGENTS.md.

    Stacked MoE expert tensors (3D, (num_experts, A, B) — identical layout in BiBo and Qwen3MoE)
    are orthogonalized PER EXPERT SLICE: Newton-Schulz batches over the expert dim and the
    0.2·√max(A,B) scaling uses the slice dims. So Muon covers attention + dense MLP + every expert.

    Step is PER-PARAM (3D experts batch over the expert dim inside NS). Cross-param shape-bucketing
    (one batched NS over all same-shape matrices) was tried and REVERTED (bench/bench_muon.py, Jun 27):
    numerically exact (1e-6) but only ~1.1× on the FLOP-bound expert matmuls and 2× the memory — which
    thrashed the 4GB local GPU (22× slower with the model resident). NS is GEMM-bound (71%, profiled);
    the GEMM count (3 matmuls/iter) is the algorithmic floor — you can't go below eager, and cuBLAS bmm
    beats Triton tl.dot here. The real launch-overhead lever is `compile_ns` (torch.compile + cudagraphs,
    Kaggle-only — broken locally), which cuts launches WITHOUT the batched-transient memory blowup.

    For 1D params (and 3D conv kernels): handled by the paired AdamW (see create_optimizer).
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 weight_decay=0.0, adamw_params=None, adamw_lr=3e-4,
                 adamw_betas=(0.9, 0.95), adamw_wd=0.1, compile_ns=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Optionally torch.compile the (shape-bucketed) Newton-Schulz: keeps GEMMs on cuBLAS but
        # fuses the elementwise (a*X + B@X, b*A + c*A@A) and cuts launch overhead on the tiny bmm.
        # OFF by default — torch.compile is broken in the local env; flip on (Kaggle) via config.
        # Calling torch.compile here is lazy (inductor only imports on first invocation), so it's
        # inert when compile_ns=False. Bucketing means few distinct shapes -> bounded recompiles.
        self._ns = torch.compile(newton_schulz_iteration, dynamic=False) if compile_ns else newton_schulz_iteration

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
            # 2D weight matrices + 3D stacked MoE expert tensors (orthogonalized per slice)
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue

            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]

                # momentum on raw grad -> orthogonalize (Nesterov). For 3D experts, NS batches over
                # the expert dim internally (cheap, E~9). self._ns is plain or torch.compile'd.
                buf.mul_(momentum).add_(p.grad)
                g = p.grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = self._ns(g)

                # consistent-RMS scale (Moonlight) using the matrix dims (last two)
                a_dim, b_dim = p.shape[-2], p.shape[-1]
                g = g.mul_(0.2 * (max(a_dim, b_dim) ** 0.5))

                # decoupled weight decay, then the scaled orthogonal step
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(g, alpha=-lr)

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
    muon_lr = train_cfg.get("muon_lr", 3e-4)  # Moonlight/Kimi RMS-matched range (NOT raw 0.02)
    compile_opt = bool(cfg.get("hardware", {}).get("compile_optimizer", False))  # Kaggle-only (compile broken locally)
    weight_decay = train_decay if (train_decay := train_cfg.get("weight_decay", 0.1)) else 0.1

    # --modded-muon: swap the standard 5-iter quintic NS for Turbo-Muon (AOL + per-iter coeffs, 4 iters).
    # Lazy import (avoids circular import; optim_modded imports Muon from here). Default path unchanged.
    use_modded = bool(train_cfg.get("modded_muon", False))
    if use_modded:
        from optim_modded import ModdedMuon as MuonClass
    else:
        MuonClass = Muon

    # Separate params by name:
    #   Muon: 2D weight matrices in attention projections + MLP/expert weights (NOT embeddings/lm_head)
    #   AdamW: everything else (embeddings, lm_head, norms, biases, scalars)
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Embeddings, lm_head, norms, biases → AdamW
        is_embedding = "embed" in name or "lm_head" in name
        is_norm_bias = param.ndim <= 1 or "norm" in name or "bias" in name
        is_small = param.ndim == 2 and param.numel() <= 10000
        # 3D stacked MoE expert tensors (BiBo + Qwen: ...experts.gate_up_proj / ...experts.down_proj)
        # → Muon (orthogonalized per expert slice). Excludes 3D conv kernels (gate_conv) by name.
        is_expert_3d = (param.ndim == 3 and "experts." in name
                        and ("gate_up_proj" in name or "down_proj" in name))

        if is_embedding or is_norm_bias or is_small:
            adamw_params.append(param)
        elif param.ndim == 2 or is_expert_3d:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if optim_type == "muon_adamw8bit":
        if HAS_BNB:
            adamw = AdamW8bit(
                [{"params": adamw_params, "weight_decay": weight_decay}],
                lr=lr, betas=(0.9, 0.95),
            )
        else:
            adamw = optim.AdamW(
                [{"params": adamw_params, "weight_decay": weight_decay}],
                lr=lr, betas=(0.9, 0.95),
            )

        muon = MuonClass(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            weight_decay=weight_decay,
            compile_ns=compile_opt,
        )

        # Combine into a single optimizer-like object
        optimizer = _CombinedOptimizer(muon, adamw)
        name = "Muon+AdamW8bit" if HAS_BNB else "Muon+AdamW"

    elif optim_type == "muon_adamw":
        # Fused fp32 AdamW (single CUDA kernel, no 8-bit quant) for embeds/norms/scalars + Muon for 2D.
        # Prefer when memory isn't the constraint: more stable fp32 states, no bnb dependency.
        fused = torch.cuda.is_available()   # fused=True requires CUDA params; falls back to foreach on CPU
        adamw = optim.AdamW(
            [{"params": adamw_params, "weight_decay": weight_decay}],
            lr=lr, betas=(0.9, 0.95), fused=fused,
        )
        muon = MuonClass(muon_params, lr=muon_lr, momentum=0.95, weight_decay=weight_decay, compile_ns=compile_opt)
        optimizer = _CombinedOptimizer(muon, adamw)
        name = "Muon+AdamW(fused)" if fused else "Muon+AdamW"

    elif optim_type == "adamw":
        all_params = muon_params + adamw_params
        fused = torch.cuda.is_available()
        optimizer = optim.AdamW(
            [{"params": all_params, "weight_decay": weight_decay}],
            lr=lr, betas=(0.9, 0.95), fused=fused,
        )
        name = "AdamW(fused)" if fused else "AdamW"

    elif optim_type == "adamw8bit":
        all_params = muon_params + adamw_params
        if HAS_BNB:
            optimizer = AdamW8bit(
                [{"params": all_params, "weight_decay": weight_decay}],
                lr=lr, betas=(0.9, 0.95),
            )
            name = "AdamW8bit"
        else:
            optimizer = optim.AdamW(
                [{"params": all_params, "weight_decay": weight_decay}],
                lr=lr, betas=(0.9, 0.95),
            )
            name = "AdamW"

    else:
        all_params = muon_params + adamw_params
        optimizer = optim.AdamW(
            [{"params": all_params, "weight_decay": weight_decay}],
            lr=lr, betas=(0.9, 0.95),
        )
        name = "AdamW"

    if use_modded and "Muon" in name:
        name += " [Turbo-NS 4it]"
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

def create_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1,
                     scheduler="cosine", decay_frac=0.05):
    """LR schedule with linear warmup. scheduler="cosine" (AdamW default) or "whd" (Muon goto).

    WHD = Warmup-Hold-Decay (Mellum 2 / GLM-4.5 Muon recipe, arXiv:2605.31268): linear warmup →
    HOLD at peak → sharp LINEAR decay to ZERO over the final `decay_frac` of total_steps. Held flat
    in between because Muon's update scales 1:1 with LR — cosine's long shallow tail bleeds that
    lever away early; WHD keeps it at peak, then anneals hard at the very end.
    ⚠ decay_frac=0.05 is shorter than disclosed Muon runs (Mellum ~15%, Kimi ~35%); IMU-1 found 10%
    underfits short runs, so a 5% anneal may still leave a little on the table. Tunable.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if scheduler == "whd":
            decay_steps = max(int(decay_frac * total_steps), 1)
            decay_start = total_steps - decay_steps
            if step < decay_start:
                return 1.0                                              # hold at peak
            return max(1.0 - (step - decay_start) / decay_steps, 0.0)   # linear decay → 0
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
