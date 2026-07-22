"""Optimizer builder: bf16-safe FusedMuon (NS8, aurora-K1) for 2D/3D matrices + AdamW for the rest.
Identical for both arms. NEVER fp16 (see the fp16-divergence finding); ns_dtype defaults bf16."""
from . import _paths  # noqa: F401
import torch

_KJ, _PIN = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)
NS8 = (_KJ,) * 6 + (_PIN,) * 2


def build_optimizers(model, muon_lr=3e-4, adam_lr=3e-4, wd=0.1, momentum=0.95, ns_dtype=torch.bfloat16,
                     scale_mode="aurora", xorth_post=0.0, xorth_gate_ref=0.3, xorth_ema=0.95,
                     xorth_warmup_steps=0, xorth_where="post"):
    from kernels.sm120.muon import FusedMuon   # Blackwell: gram-NS (self-gates to symmul/cuBLAS on small mats) + 8M knee
    stacks, mats, other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in n or p.ndim not in (2, 3):
            other.append(p)                 # -> AdamW (1D norms/biases, embeddings)
        elif p.ndim == 3:
            stacks.append(p)                # 3D MoE expert stacks -> the xorth (cross-expert whitening) target
        else:
            mats.append(p)                  # plain 2D weight matrices -> never whitened
    # gram_restarts=[4,5] = the NS8-schedule fp16 autotune winner (gram only activates for dim>=2048; harmless below)
    # scale_mode = post-NS row scaling (ABLATION AXIS): aurora (default, no EMA) | normuon | aurora_ema |
    # aurora_ema_v2 (the EMA variants keep a persistent per-row 2nd-moment buffer) | polar.
    # xorth_post = cross-expert whitening MAX strength (0=off); SCOPED to the 3D expert stacks only (2D=0), so
    # whitening acts exactly on the MoE experts. xorth_gate_ref = correlation gate (full whitening at off-diag
    # RMS >= this; below it ramps to ~0 so decorrelated experts are left alone; <=0 disables gate). xorth_ema =
    # EMA decay of the persistent per-stack (E,E) gram (identity-init).
    groups = []
    if stacks:
        groups.append({"params": stacks, "xorth_post": xorth_post})
    if mats:
        groups.append({"params": mats, "xorth_post": 0.0})
    _xo = dict(xorth_post=xorth_post, xorth_gate_ref=xorth_gate_ref, xorth_ema=xorth_ema,
               xorth_warmup_steps=xorth_warmup_steps, xorth_where=xorth_where)
    muon = FusedMuon(groups, lr=muon_lr, momentum=momentum, weight_decay=wd,
                     coeffs=NS8, ns_dtype=ns_dtype, aurora_k=1, gram_restarts=[4, 5], scale_mode=scale_mode,
                     **_xo)
    adamw = torch.optim.AdamW(other, lr=adam_lr, weight_decay=wd)
    return [muon, adamw], len(stacks) + len(mats), len(other)
