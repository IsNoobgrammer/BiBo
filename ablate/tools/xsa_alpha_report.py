"""Per-layer XSA rejection strength from a checkpoint, grouped by attention type.

XSA removes the component of the attended output along each value direction:
    Y <- Y - a (Y . Vn) Vn,   a = tanh(xsa_alpha),  one LEARNABLE LOGIT PER HEAD
so a=1 is full rejection of the self-component, a=0 is XSA off, a<0 AMPLIFIES it.
`xsa_alpha` is (num_heads,) per layer -- per HEAD, not per hidden dimension.

Why this report exists: under swa_pattern=block3 on 10 layers the schedule is

    L0 L1 L2 L3 L4 L5 L6 L7 L8 L9
     G  S  S  G  S  S  G  S  S  G        G = global attention, S = sliding window
     D  M  M  M  M  M  M  M  M  D        D = dense FFN (mlp_only_layers), M = MoE

and the per-dim AttnRes carry `c` reproducibly dips at L6 and peaks at L7 across every run at
both 524M and 1B. The question is whether XSA's learned strength explains it -- i.e. whether L6
is doing something different with its value-component rejection than the other global layers.
CPU only, no GPU: this is a state_dict read, so it runs while another arm trains.

    python -m ablate.tools.xsa_alpha_report --ckpt runs/foo_final.pt [--ckpt other.pt]
"""
import argparse
import json
import math
import os

import torch

GLOBAL_LAYERS = {0, 3, 6, 9}      # swa_block_pattern(10) -> [0,1,1,0,1,1,0,1,1,0]
DENSE_FFN = {0, 9}                # SHARED["mlp_only_layers"]


def _kind(L):
    a = "global" if L in GLOBAL_LAYERS else "swa128"
    f = "dense" if L in DENSE_FFN else "moe"
    return f"{a}/{f}"


def _carry(sd, L):
    """per-dim AttnRes carry c for layer L, if this arm has one."""
    for k in (f"model.layers.{L}.attn_res_carry_theta",):
        if k in sd:
            t = sd[k].float()
            return t.mean().item(), t.std().item() if t.numel() > 1 else 0.0, t.numel()
    return None


def report(path):
    sd = torch.load(path, map_location="cpu")
    name = os.path.basename(path)
    alphas = {}
    for k, v in sd.items():
        if k.endswith("self_attn.xsa_alpha"):
            alphas[int(k.split(".")[2])] = v.float()
    if not alphas:
        print(f"\n{name}: no xsa_alpha in this checkpoint (use_xsa off?)")
        return None

    print(f"\n{'='*94}\n{name}\n{'='*94}")
    nh = len(next(iter(alphas.values())))
    hdr = f"{'L':>2} {'kind':<12} " + " ".join(f"{'h'+str(i):>7}" for i in range(nh))
    hdr += f" | {'mean a':>7} {'|a| mean':>8} {'carry c':>9}"
    print(hdr); print("-" * len(hdr))
    rows = {}
    for L in sorted(alphas):
        a = torch.tanh(alphas[L])                       # APPLIED strength, not the raw logit
        c = _carry(sd, L)
        rows[L] = a
        cs = f"{c[0]:9.4f}" if c else f"{'-':>9}"
        print(f"{L:>2} {_kind(L):<12} " + " ".join(f"{x:7.4f}" for x in a.tolist())
              + f" | {a.mean():7.4f} {a.abs().mean():8.4f} {cs}")

    def grp(ls):
        v = torch.cat([rows[L] for L in ls if L in rows])
        return v.mean().item(), v.abs().mean().item()

    print()
    g, s = [L for L in rows if L in GLOBAL_LAYERS], [L for L in rows if L not in GLOBAL_LAYERS]
    print(f"  global layers {g}: mean a = {grp(g)[0]:+.4f}   mean |a| = {grp(g)[1]:.4f}")
    print(f"  swa    layers {s}: mean a = {grp(s)[0]:+.4f}   mean |a| = {grp(s)[1]:.4f}")
    print()
    # the comparison the round actually asked for: the L6/L7 pair against the other G/S pairs
    print("  matched (global, next-swa) pairs -- L6/L7 is the anomalous carry pair:")
    for gl, sw in ((0, 1), (3, 4), (6, 7)):
        if gl in rows and sw in rows:
            print(f"    L{gl}(G) a={rows[gl].mean():+.4f}   L{sw}(S) a={rows[sw].mean():+.4f}"
                  f"   delta={rows[sw].mean()-rows[gl].mean():+.4f}")
    if 8 in rows and 9 in rows:
        print(f"    L8(S) a={rows[8].mean():+.4f}   L9(G) a={rows[9].mean():+.4f}"
              f"   delta={rows[9].mean()-rows[8].mean():+.4f}   (tail, order reversed)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    args = ap.parse_args()
    for p in args.ckpt:
        report(p)


if __name__ == "__main__":
    main()
