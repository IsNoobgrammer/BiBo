"""Per-tensor parameter and gradient norms, plus per-layer router logit scale.

Both borrowed from the marin_moe 67B run (`params/norm/*`, `grad/norm/*`,
`train/router/layer_N/router_z_loss`), which logs them for every weight tensor and every layer.

WHY THEY EARN THEIR PLACE HERE:

`grad/norm/*` is the cheapest possible detector for an INERT PARAMETER. A tensor whose gradient
norm sits at zero while training proceeds is not being trained, and this repo has shipped that bug
more than once -- XSA's alpha was dead behind a parity test that passed, and ACT_CYCLE steered only
the patched forward while the eager path ran a hardcoded activation. Neither showed up in the loss
for days. A grad norm of 0.0 would have shown up in one step.

`params/norm/*` catches the other half: a tensor that is growing without bound, or one that never
moves from its initialisation.

`router_z_loss` is the mean squared logsumexp of the router logits -- the router's LOGIT SCALE, as
opposed to the entropy of its output distribution. We log entropy, top-1 weight and balance, all of
which are computed AFTER the sigmoid, so none of them can see the logits growing. In the marin run
this quantity climbs from 3.2 at layer 0 to 11-13 in the last third (peak 26.3), a depth profile
that is invisible in entropy -- theirs is flat at 0.984-0.993 throughout. Given the temperature
round found an interior optimum, logit scale by depth is exactly the missing variable.

NOT applied as a loss. marin logs it with `router_z_loss_coef = 0` and so do we: this is a
diagnostic, and turning it into an objective is a separate, pre-registered decision.

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


class RouterLogitScale:
    """Per-layer router z-loss = mean(logsumexp(logits)^2), the router's logit SCALE.

    Hooks `router_logits` on each router. That method exists precisely so the raw pre-activation
    logits are reachable; every other router diagnostic we log reads post-sigmoid scores and is
    therefore blind to this.
    """

    def __init__(self, model):
        self.z = {}
        self._handles = []
        i = 0
        for _, mod in model.named_modules():
            if mod.__class__.__name__ == "BiBoMoERouter":
                self._handles.append(mod.register_forward_hook(self._mk(i)))
                i += 1
        self.n_layers = i

    def _mk(self, i):
        @torch.no_grad()
        def hook(mod, args, out):
            h = args[0] if args else None
            if h is None:
                return
            lg = mod.router_logits(h)
            self.z[i] = (torch.logsumexp(lg.float(), dim=-1) ** 2).mean().item()
        return hook

    def stats(self, reset=True):
        out = {f"train/router/layer_{i}/router_z_loss": v for i, v in self.z.items()}
        if self.z:
            out["train/router/z_loss_mean"] = sum(self.z.values()) / len(self.z)
        if reset:
            self.z = {}
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
