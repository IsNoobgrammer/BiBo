"""Optimizer builder: bf16-safe FusedMuon (NS8, aurora-K1) for 2D/3D matrices + AdamW for the rest.
Identical for both arms. NEVER fp16 (see the fp16-divergence finding); ns_dtype defaults bf16."""
from . import _paths  # noqa: F401
import torch

_KJ, _PIN = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)
NS8 = (_KJ,) * 6 + (_PIN,) * 2


def build_optimizers(model, muon_lr=3e-4, adam_lr=3e-4, wd=0.1, momentum=0.95, ns_dtype=torch.bfloat16,
                     scale_mode="aurora"):
    from kernels.sm120.muon import FusedMuon   # Blackwell: gram-NS (self-gates to symmul/cuBLAS on small mats) + 8M knee
    matrix, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (matrix if (p.ndim in (2, 3) and "embed" not in n) else other).append(p)
    # gram_restarts=[4,5] = the NS8-schedule fp16 autotune winner (gram only activates for dim>=2048; harmless below)
    # scale_mode = post-NS row scaling (ABLATION AXIS): aurora (default, no EMA) | normuon | aurora_ema |
    # aurora_ema_v2 (the EMA variants keep a persistent per-row 2nd-moment buffer) | polar.
    muon = FusedMuon(matrix, lr=muon_lr, momentum=momentum, weight_decay=wd,
                     coeffs=NS8, ns_dtype=ns_dtype, aurora_k=1, gram_restarts=[4, 5], scale_mode=scale_mode)
    adamw = torch.optim.AdamW(other, lr=adam_lr, weight_decay=wd)
    return [muon, adamw], len(matrix), len(other)
