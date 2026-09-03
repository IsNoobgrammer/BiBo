"""Per-layer routing behaviour on a trained checkpoint: load balance, confidence, specialisation.

    python -m ablate.tools.router_probe <..._result.json> [--seqs 8] [--json out.json]

Answers "do the layers that only just BECAME MoE (0 and n-1 in the all-MoE arm) route like the
layers that were always MoE, or are they doing something else?" -- which the training log cannot,
because it reports balance entropy and top-1 weight averaged over the whole model.

Hooks the expert module's forward_pre_hook, whose args are (hidden, top_k_index, top_k_weights) --
the same seam RouterTrace and interp_experts use, and the one point that is identical on the fused
and eager dispatch paths, so nothing here depends on which kernel ran. Only counts and running
sums are kept, so it is safe to run alongside a training job.

Architecture comes from load_from_result, never from retyped flags: a probe that rebuilds the
wrong model reports confidently on something that was never trained.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import json
import math

import torch

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    model, cfg = load_from_result(a.result_json, device=DEV)
    model.eval()
    holdout = _val.build_holdout(cfg.dataset, a.seq_len, a.seqs, DEV)

    layers = [(i, l.mlp.experts) for i, l in enumerate(model.model.layers)
              if hasattr(getattr(l, "mlp", None), "experts")]
    E = layers[0][1].gate_up_proj.shape[0]
    stats = {i: {"load": torch.zeros(E, device=DEV), "w1": 0.0, "wsum": 0.0, "n": 0}
             for i, _ in layers}

    def mk(i):
        def hook(mod, args):
            _h, idx, wgt = args[0], args[1], args[2]
            idx = idx.reshape(-1, idx.shape[-1])
            w = wgt.reshape(-1, wgt.shape[-1]).float()
            s = stats[i]
            s["load"] += torch.bincount(idx.reshape(-1), minlength=E).float()
            s["w1"] += w.max(-1).values.sum().item()
            s["wsum"] += w.sum().item()
            s["n"] += idx.shape[0]
        return hook

    hs = [m.register_forward_pre_hook(mk(i)) for i, m in layers]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(holdout.shape[0]):
            model(holdout[b:b + 1, :-1])
    for h in hs:
        h.remove()

    dense = getattr(cfg, "mlp_only_layers", None)
    print(f"\nexperts={E}  mlp_only_layers={dense}  tokens/layer={stats[layers[0][0]]['n']}")
    print(f"{'L':<4}{'bal_entropy':>12}{'max_load%':>11}{'dead':>6}{'top1_w':>9}{'w_sum':>8}")
    out = {}
    for i, _ in layers:
        s = stats[i]
        p = (s["load"] / s["load"].sum()).clamp_min(1e-12)
        ent = float(-(p * p.log()).sum() / math.log(E))       # 1.0 = perfectly balanced
        row = dict(bal_entropy=ent, max_load=float(p.max()),
                   dead=int((s["load"] == 0).sum()),
                   top1_w=s["w1"] / max(s["n"], 1), w_sum=s["wsum"] / max(s["n"], 1))
        out[i] = row
        print(f"L{i:<3}{ent:>12.4f}{100*row['max_load']:>11.2f}{row['dead']:>6}"
              f"{row['top1_w']:>9.4f}{row['w_sum']:>8.4f}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"experts": E, "mlp_only_layers": str(dense), "layers": out}, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
