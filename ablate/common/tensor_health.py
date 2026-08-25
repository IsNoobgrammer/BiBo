"""Per-tensor parameter and gradient norms.

Borrowed from the marin_moe 67B run (`params/norm/*`, `grad/norm/*`), which logs them for every
weight tensor. The router half of that idea lives in per_layer.py, which owns every per-layer
router metric so there is exactly one implementation of each.

WHY THEY EARN THEIR PLACE HERE:

`grad/norm/*` is the cheapest possible detector for an INERT PARAMETER. A tensor whose gradient
norm sits at zero while training proceeds is not being trained, and this repo has shipped that bug
more than once -- XSA's alpha was dead behind a parity test that passed, and ACT_CYCLE steered only
the patched forward while the eager path ran a hardcoded activation. Neither showed up in the loss
for days. A grad norm of 0.0 would have shown up in one step.

`params/norm/*` catches the other half: a tensor that is growing without bound, or one that never
moves from its initialisation.

Cost: one `.norm()` per tensor per logged step, grouped so the key count stays small.
"""
from . import _paths  # noqa: F401

import re

import torch

# Collapse `model.layers.7.self_attn.q_proj.weight` -> `layers.self_attn.q_proj`, so 10 layers of
# the same tensor share one series. Per-LAYER series for every weight would be 400+ keys on this
# model and would bury the handful that matter; the point is to spot a dead or exploding TENSOR,
# and a tensor that is dead in one layer is nearly always dead in all of them.
_LAYER_IDX = re.compile(r"\.(\d+)\.")


def _group(name):
    return _LAYER_IDX.sub(".", name).replace("model.", "").replace(".weight", "")


@torch.no_grad()
def tensor_norms(model, grads=True):
    """{'params/norm/<group>': float, 'grad/norm/<group>': float}.

    Groups are RMS-combined across layers (sqrt of mean of squared norms) rather than summed, so
    the number does not drift just because the model got deeper.
    """
    pn, gn = {}, {}
    for name, p in model.named_parameters():
        g = _group(name)
        pn.setdefault(g, []).append(p.detach().float().norm().item() ** 2)
        if grads and p.grad is not None:
            gn.setdefault(g, []).append(p.grad.detach().float().norm().item() ** 2)
    out = {f"params/norm/{k}": (sum(v) / len(v)) ** 0.5 for k, v in pn.items()}
    out.update({f"grad/norm/{k}": (sum(v) / len(v)) ** 0.5 for k, v in gn.items()})
    # One scalar worth watching on its own: the smallest gradient norm in the model. If a tensor
    # goes inert this drops to 0 and stays there, which is visible on a single chart without
    # reading 30 series.
    if gn:
        out["grad/norm_min_over_tensors"] = min(out[k] for k in out if k.startswith("grad/norm/"))
    return out
