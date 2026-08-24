"""Per-layer / per-head report from a trained checkpoint: AttnRes carry, XSA alpha, radial p.

    python -m ablate.tools.carry_report <checkpoint.pt> [--json out.json]

W&B logs the carry per layer but XSA and radial p only as min/mean/max over the whole model, so
the questions "which head went negative" and "how does p ramp with depth" can ONLY be answered
from a checkpoint. Aug 15 2026 a box died with its checkpoints and those tables were lost for a
whole 3-arm round -- run this ON THE BOX and keep the numbers, they are a few KB.

Reads the state dict directly, so it needs no model build and no GPU.
"""
import argparse
import json
import math

import torch

# c = f(theta) per attn_res_carry_scale. Kept in sync with exp/modeling_bibo.py _CARRY_C: a report
# that assumes the wrong transform prints plausible wrong numbers, which is worse than crashing.
CARRY_F = {
    "none": lambda t: torch.ones_like(t),
    "raw": lambda t: t,
    "unbounded": lambda t: t,
    "rms": lambda t: t,
    "sigmoid": lambda t: 2 * torch.sigmoid(t),
    "sigmoid_rms": lambda t: 2 * torch.sigmoid(t),
    "tanh": lambda t: 1 + torch.tanh(t),
    "tanh_rms": lambda t: 1 + torch.tanh(t),
    "slope05": lambda t: 1 + 0.5 * t,
    "slope15": lambda t: 1 + 1.5 * t,
}


def _stats(v):
    v = v.detach().float().flatten()
    return dict(mean=v.mean().item(), std=v.std().item() if v.numel() > 1 else 0.0,
                min=v.min().item(), max=v.max().item(),
                negfrac=(v < 0).float().mean().item(), n=v.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--json", default=None)
    ap.add_argument("--carry_scale", default=None,
                    help="override; otherwise read from the checkpoint's config")
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob.get("state_dict", blob))
    cfg = blob.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = vars(cfg)
    mode = a.carry_scale or cfg.get("attn_res_carry_scale", "raw")
    dense = cfg.get("mlp_only_layers", "?")
    print(f"checkpoint: {a.ckpt}")
    print(f"carry_scale={mode}  mlp_only_layers={dense}  tensors={len(sd)}")

    # group the interesting parameters by layer index, by name
    def collect(substr):
        out = {}
        for k, v in sd.items():
            if substr in k:
                parts = [p for p in k.split(".") if p.isdigit()]
                out.setdefault(int(parts[0]) if parts else -1, []).append((k, v))
        return out

    report = {"carry_scale": mode, "mlp_only_layers": dense, "layers": {}}

    # exact names, verified against the model source: carry c = f(attn_res_carry_theta),
    # XSA strength = tanh(xsa_alpha) per head, radial p = sigmoid(radial_theta) per GLU expert.
    carry = collect("attn_res_carry_theta")
    xsa = collect("xsa_alpha")
    radial = collect("radial_theta")

    print(f"\n{'L':<4}{'carry c':>34}   {'xsa tanh(alpha) per head':>34}   {'radial p':>22}")
    print(f"{'':4}{'mean':>9}{'std':>8}{'min':>9}{'neg%':>8}   "
          f"{'mean':>9}{'min':>9}{'max':>9}{'neg':>7}   {'mean':>9}{'min':>7}{'max':>7}")
    for L in sorted(set(carry) | set(xsa) | set(radial)):
        if L < 0:
            continue
        row = {}
        line = f"L{L:<3}"
        if L in carry:
            theta = torch.cat([v.flatten() for _, v in carry[L]])
            c = CARRY_F.get(mode, CARRY_F["raw"])(theta)
            s = _stats(c)
            row["carry"] = s
            line += f"{s['mean']:>9.4f}{s['std']:>8.4f}{s['min']:>9.4f}{100*s['negfrac']:>7.1f}%"
        else:
            line += " " * 34
        if L in xsa:
            al = torch.cat([v.flatten() for _, v in xsa[L]])
            t = torch.tanh(al.float())
            s = _stats(t)
            row["xsa_tanh_alpha"] = s
            row["xsa_per_head"] = [round(x, 4) for x in t.tolist()]
            line += f"   {s['mean']:>9.4f}{s['min']:>9.4f}{s['max']:>9.4f}{int(s['negfrac']*s['n']):>4}/{s['n']:<2}"
        else:
            line += " " * 37
        if L in radial:
            p = torch.sigmoid(torch.cat([v.flatten() for _, v in radial[L]]).float())
            s = _stats(p)
            row["radial_p"] = s
            line += f"   {s['mean']:>9.4f}{s['min']:>7.3f}{s['max']:>7.3f}"
        print(line)
        report["layers"][L] = row

    if not carry:
        print("\nNO attn_res_carry_theta PARAMS FOUND -- key names may have changed; "
              f"sample keys: {list(sd)[:5]}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
