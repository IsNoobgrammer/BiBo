"""Causal test of the AttnRes carry: knock out each layer's carry and measure the loss cost.

    python -m ablate.tools.carry_ablate <..._result.json> [--seqs 8] [--json out.json]

WHY. W&B shows the learned carry scale c per layer, and in the all-MoE arms layer 0 settled at
c=0.009 under `raw` but c=0.781 under `sigmoid` -- for the same val loss (3.5829 vs 3.5903).
A learned value near zero is NOT evidence that the term is unused, and a learned value near one is
not evidence that it is load-bearing: both are consistent with a FLAT direction the optimizer is
free to wander along. The only way to tell is to intervene and measure.

WHAT IT DOES. For each layer, overwrite that layer's carry with a constant (0 = term removed,
1 = term forced to full strength), rerun the SAME frozen batches, and report delta vs baseline.
The weights are restored between layers, so the ablations are independent and the baseline is
recomputed at the end as a self-check: if it does not return to its original value, the restore
leaked and every number above it is void.

READ IT AS: cost(c=0) large  -> the carry at that layer is doing real work.
            cost(c=0) ~0     -> flat direction; the learned value there means nothing.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import json

import torch

import contextlib

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from kernels.sm120.cross_entropy import fused_linear_cross_entropy

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
    dataset = getattr(cfg, "dataset", None)
    holdout = _val.build_holdout(dataset, a.seq_len, a.seqs, DEV)
    print(f"holdout {tuple(holdout.shape)}  carry_scale={getattr(cfg,'attn_res_carry_scale','?')}"
          f"  mlp_only={getattr(cfg,'mlp_only_layers','?')}")

    layers = model.model.layers
    have = [i for i, l in enumerate(layers) if getattr(l, "attn_res_carry_theta", None) is not None]

    # same CE and autocast the run trained and validated under -- a different dtype here would
    # move the baseline and make every delta below a mix of ablation and precision.
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def loss():
        out, _flat = _val.losses(model, holdout, None, fused_linear_cross_entropy, amp, pad_id=0)
        return float(out)

    base = loss()
    print(f"\nbaseline loss {base:.4f}")
    # theta such that c == target, inverted per transform; done in theta-space so the forward
    # path is untouched and the ablation cannot diverge from how the model actually computes c.
    mode = getattr(cfg, "attn_res_carry_scale", "raw")

    def theta_for(c):
        t = torch.tensor(float(c))
        if mode in ("sigmoid", "sigmoid_rms"):
            c = min(max(c, 1e-6), 2 - 1e-6)
            return torch.logit(torch.tensor(c / 2)).item()
        if mode in ("tanh", "tanh_rms"):
            return torch.atanh(torch.tensor(min(max(c - 1, -0.999), 0.999))).item()
        return float(c)          # raw / unbounded / rms: c IS theta

    rows = []
    print(f"\n{'layer':7}{'c=0 loss':>11}{'delta':>9}   {'c=1 loss':>11}{'delta':>9}")
    for i in have:
        p = layers[i].attn_res_carry_theta
        keep = p.data.clone()
        out = {"layer": i}
        for target in (0.0, 1.0):
            p.data.fill_(theta_for(target))
            l = loss()
            out[f"loss_c{int(target)}"] = l
            out[f"delta_c{int(target)}"] = l - base
        p.data.copy_(keep)       # restore BEFORE the next layer -- ablations must be independent
        rows.append(out)
        print(f"L{i:<6}{out['loss_c0']:>11.4f}{out['delta_c0']:>+9.4f}   "
              f"{out['loss_c1']:>11.4f}{out['delta_c1']:>+9.4f}")

    after = loss()
    ok = abs(after - base) < 1e-6
    print(f"\nbaseline recheck {after:.6f} (was {base:.6f}) -> "
          f"{'restore clean' if ok else 'RESTORE LEAKED -- numbers above are void'}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"baseline": base, "recheck": after, "restore_clean": ok,
                       "carry_scale": mode, "rows": rows}, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
