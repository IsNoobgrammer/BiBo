"""Optimizer builder: bf16-safe FusedMuon (NS8, aurora-K1) for 2D/3D matrices + AdamW for the rest.
Identical for both arms. NEVER fp16 (see the fp16-divergence finding); ns_dtype defaults bf16."""
from . import _paths  # noqa: F401
import torch

_KJ, _PIN = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)
NS8 = (_KJ,) * 6 + (_PIN,) * 2


def build_optimizers(model, muon_lr=3e-4, adam_lr=3e-4, wd=0.1, momentum=0.95, ns_dtype=torch.bfloat16):
    from kernels.sm75.muon import FusedMuon
    matrix, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (matrix if (p.ndim in (2, 3) and "embed" not in n) else other).append(p)
    muon = FusedMuon(matrix, lr=muon_lr, momentum=momentum, weight_decay=wd,
                     coeffs=NS8, ns_dtype=ns_dtype, aurora_k=1)
    adamw = torch.optim.AdamW(other, lr=adam_lr, weight_decay=wd)
    return [muon, adamw], len(matrix), len(other)
