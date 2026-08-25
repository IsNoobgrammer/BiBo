"""Per-layer, per-step diagnostics: expert load, balance, boundary gap, XSA alpha, radial p.

Modelled on the marin_moe 67B run, which logs `train/router/layer_N/{routing_hist, ...}` for every
layer and every step. W&B renders a per-step histogram series as a heatmap with step on x and the
bin on y, so a 64-expert load distribution over 2000 steps becomes one picture -- and a collapse,
a dead expert, or a slow drift into bimodality is visible at a glance in a way that max-load and
entropy cannot show. Both of those are summary statistics of exactly the distribution being hidden.

WHAT IS LOGGED PER LAYER

  routing_hist        expert load, one bin per expert. The picture.
  load_balancing_loss the Switch aux quantity, normalised so 1.0 == perfectly uniform.
                      DIAGNOSTIC ONLY -- never added to the objective. Balance here comes from
                      the router bias update, and that is the arrangement being measured.
  boundary_gap        score(k-th) - score(k+1-th): how close the routing decision was to flipping.
                      We already log this averaged over the model; per layer is the useful form,
                      since a model-wide mean cannot say WHICH layer is undecided.
  router_z_loss       logsumexp(logits)^2, the router's LOGIT scale (see tensor_health).

  xsa_hist            tanh(alpha) per head, one bin per head. Only num_heads values (4 on the
                      board config), so the histogram is coarse by construction -- but as a
                      per-step heatmap it shows a head flipping sign, which the model-wide
                      min/mean/max cannot localise. Per-head scalars are logged too; 4 heads x 10
                      layers is 40 series, which is affordable, and a scalar is what you plot when
                      you already know which head you care about.
  radial_p_hist       sigmoid(radial_theta) per GLU expert, one bin per expert. The depth ramp was
                      only ever visible per layer; the interp round then found the L0-L2 profile
                      is NOT stable across runs while L3+ is, and a per-step histogram is how you
                      would have seen that during training rather than from two checkpoints.

NOT COMPARABLE TO MARIN'S NUMBERS. Their router is softmax over 256 experts; ours is sigmoid over
64 with a bias, so the load-balancing scale differs. Compare our layers to each other and our runs
to each other, never our absolute value to theirs.

Cost: one extra router forward per logged step (a hidden x E linear) plus a few histograms. The
hooks are installed once and only fire while `enabled` is set, so the untraced steps pay nothing.
"""
from . import _paths  # noqa: F401

import math

import torch


def _hist(values, n_bins):
    """A wandb.Histogram binned by INDEX (expert id, head id), not by value.

    wandb.Histogram(values) would bin by magnitude and lose which expert a bar belongs to, which
    is the entire point here. np_histogram takes (counts, edges) directly, so bin i is item i.
    Returns a plain list when wandb is unavailable, so nothing here can break a run.
    """
    try:
        import wandb
        import numpy as np
        return wandb.Histogram(np_histogram=(list(values), list(np.arange(n_bins + 1))))
    except Exception:
        return None


class PerLayerRouter:
    """Per-layer router diagnostics, accumulated over the steps between logs."""

    def __init__(self, model, num_experts, top_k):
        self.E, self.k = num_experts, top_k
        self.enabled = False
        self.acc = {}
        self._handles = []
        self.n = 0
        for _, mod in model.named_modules():
            if mod.__class__.__name__ == "BiBoMoERouter":
                self._handles.append(mod.register_forward_hook(self._mk(self.n)))
                self.n += 1

    def _mk(self, i):
        @torch.no_grad()
        def hook(mod, args, out):
            if not self.enabled or not args:
                return
            scores = torch.sigmoid(mod.router_logits(args[0]))          # (N, E), pre-bias
            idx = out[0].reshape(-1, out[0].shape[-1])                  # (N, k) chosen experts
            a = self.acc.setdefault(i, {"counts": torch.zeros(self.E, dtype=torch.float64),
                                        "p": torch.zeros(self.E, dtype=torch.float64),
                                        "gap": 0.0, "z": 0.0, "n": 0})
            a["counts"] += torch.bincount(idx.reshape(-1).to("cpu"),
                                          minlength=self.E).double()
            a["p"] += scores.float().mean(0).to("cpu").double()
            if self.k < self.E:
                tk = scores.topk(self.k + 1, dim=-1).values
                a["gap"] += (tk[..., self.k - 1] - tk[..., self.k]).mean().item()
            lg = mod.router_logits(args[0]).float()
            a["z"] += (torch.logsumexp(lg, dim=-1) ** 2).mean().item()
            a["n"] += 1
        return hook

    def flush(self):
        """Metrics for the interval, then reset. Empty dict if nothing was traced."""
        out = {}
        for i, a in sorted(self.acc.items()):
            if not a["n"]:
                continue
            pre = f"train/router/layer_{i}"
            counts = a["counts"]
            tot = counts.sum().clamp_min(1)
            f = counts / tot                                   # fraction of assignments per expert
            p = a["p"] / a["n"]
            p = p / p.sum().clamp_min(1e-12)                   # normalised mean router mass
            # Switch aux quantity, scaled so a uniform router reads exactly 1.0 -- the raw
            # E*sum(f*P) is 1.0 at uniform only for a softmax router, and ours is sigmoid.
            out[f"{pre}/load_balancing_loss"] = float(self.E * (f * p).sum())
            ent = float(-(f.clamp_min(1e-12) * f.clamp_min(1e-12).log()).sum())
            out[f"{pre}/routing_entropy"] = ent
            out[f"{pre}/balance_entropy"] = ent / math.log(self.E)   # 1.0 = perfectly balanced
            out[f"{pre}/max_load"] = float(f.max())
            out[f"{pre}/dead_experts"] = int((counts == 0).sum())
            out[f"{pre}/boundary_gap"] = a["gap"] / a["n"]
            out[f"{pre}/router_z_loss"] = a["z"] / a["n"]
            h = _hist(counts.tolist(), self.E)
            if h is not None:
                out[f"{pre}/routing_hist"] = h
        self.acc = {}
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def per_layer_params(model):
    """Per-layer XSA alpha and radial p, as histograms plus the scalars worth plotting alone.

    Read straight off the parameters, so no hooks and no forward cost. Both are transformed into
    the units the MODEL applies -- tanh(alpha) and sigmoid(radial_theta) -- because "off" has to
    be readable at a glance: an arm that never switches XSA on is a null, not a win, and that is
    invisible if the raw logit is what gets plotted.
    """
    out = {}
    for i, layer in enumerate(getattr(getattr(model, "model", model), "layers", [])):
        a = getattr(getattr(layer, "self_attn", None), "xsa_alpha", None)
        if a is not None:
            t = torch.tanh(a.detach().float().flatten())
            pre = f"train/xsa/layer_{i}"
            h = _hist(t.tolist(), t.numel())
            if h is not None:
                out[f"{pre}/hist"] = h
            out[f"{pre}/mean"] = t.mean().item()
            out[f"{pre}/min"] = t.min().item()
            out[f"{pre}/max"] = t.max().item()
            for j, v in enumerate(t.tolist()):        # 4 heads: cheap, and directly plottable
                out[f"{pre}/h{j}"] = v
        rt = getattr(getattr(getattr(layer, "mlp", None), "experts", None), "radial_theta", None)
        if rt is not None:
            p = torch.sigmoid(rt.detach().float().flatten())
            pre = f"train/radial_p/layer_{i}"
            h = _hist(p.tolist(), p.numel())
            if h is not None:
                out[f"{pre}/hist"] = h
            out[f"{pre}/mean"] = p.mean().item()
            out[f"{pre}/min"] = p.min().item()
            out[f"{pre}/max"] = p.max().item()
            out[f"{pre}/std"] = p.std().item() if p.numel() > 1 else 0.0
    return out
