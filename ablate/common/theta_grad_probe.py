"""Is radial's p bound BINDING at the low end, or is the optimum interior?

p = sigmoid(theta) in (0,1), so p can approach normsilu but never pass it. If the model wants
gain < 1 (i.e. r^p with p < 0, shrinking high-rms tokens) it cannot say so, and switching to
p = tanh(theta) in (-1,1) would be worth an arm. If the optimum is interior, tanh buys nothing and
adds range for no reason.

SIGN CONVENTION, stated because it is easy to get backwards: descent applies theta -= lr * g.
So a persistently POSITIVE d(loss)/d(theta) drives theta DOWN, i.e. the model is still trying to
lower p. That -- on experts already at low p -- is the binding signature. g ~ 0 means interior.

A single batch's gradient is noise, so this averages over several and reports SIGN CONSISTENCY:
a real constraint shows the same direction batch after batch, not just a large mean.

    python -m ablate.common.theta_grad_probe --ckpt runs/..._k8_cos_final.pt --experts 64 --top_k 8
"""
from . import _paths  # noqa: F401
import argparse

import torch

from .models import build_arm
from . import patches as patchmod
from .data import token_batches
from kernels.sm120.cross_entropy import fused_linear_cross_entropy

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--special_pairs", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)      # small: a training run may share the GPU
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--dataset", default="/home/marimo/work/data/bip2")
    ap.add_argument("--low_p", type=float, default=0.2)  # "low p" = pressed toward the p->0 bound
    a = ap.parse_args()

    model, cfg = build_arm("bibo_min", device=DEV, dtype=torch.float32, attn_impl="sdpa",
                           num_experts=a.experts, special_pairs=a.special_pairs, top_k=a.top_k)
    patchmod.apply(["liger_norm", "liger_rope", "moe"])
    sd = torch.load(a.ckpt, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        raise SystemExit("checkpoint/config mismatch -- refusing to report gradients for a "
                         f"partly-random model ({len(missing)} missing, {len(unexpected)} unexpected)")
    model.train()

    thetas = [(n, p) for n, p in model.named_parameters() if n.endswith("radial_theta")]
    assert thetas, "no radial_theta found"
    print(f"[probe] {len(thetas)} radial_theta tensors, {thetas[0][1].numel()} experts each", flush=True)

    gen = token_batches(a.batch, a.seq_len, DEV, dataset=a.dataset, seed=1234)
    grads = []                                          # per batch: list of (L, E) grads
    for b in range(a.batches):
        ids = next(gen)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.model(input_ids=ids[:, :-1], use_cache=False)
            h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        loss = fused_linear_cross_entropy(h.float(), model.lm_head.weight.float(),
                                          ids[:, 1:].reshape(-1))
        loss.backward()
        grads.append(torch.stack([p.grad.detach().float().clone() for _, p in thetas]))
        print(f"  batch {b+1}/{a.batches}  loss={loss.item():.4f}", flush=True)

    G = torch.stack(grads)                              # (batches, layers, experts)
    p = torch.sigmoid(torch.stack([t.detach().float() for _, t in thetas]))   # (layers, experts)
    gmean, gstd = G.mean(0), G.std(0)
    pos_frac = (G > 0).float().mean(0)                  # fraction of batches pushing theta DOWN

    print("\n  d(loss)/d(theta): POSITIVE mean + high sign-agreement on LOW-p experts would mean")
    print("  the model is still trying to push p below the sigmoid floor (bound BINDS).\n")
    print("LAYER  n_low   <g|low p>   sign-agree   <g|high p>  sign-agree   |g|low / |g|high")
    for L in range(p.shape[0]):
        lo = p[L] < a.low_p
        hi = ~lo
        if lo.sum() == 0:
            print(f"{L:5d} {0:6d}   (no experts below p={a.low_p})")
            continue
        gl, gh = gmean[L][lo], gmean[L][hi]
        al = torch.maximum(pos_frac[L][lo], 1 - pos_frac[L][lo]).mean().item()
        ah = (torch.maximum(pos_frac[L][hi], 1 - pos_frac[L][hi]).mean().item()
              if hi.sum() else float("nan"))
        ratio = (gl.abs().mean() / gh.abs().mean()).item() if hi.sum() and gh.abs().mean() > 0 else float("nan")
        print(f"{L:5d} {int(lo.sum()):6d} {gl.mean().item():+11.3e} {al:12.2f} "
              f"{(gh.mean().item() if hi.sum() else float('nan')):+12.3e} {ah:11.2f} {ratio:16.2f}")

    lo_all = p < a.low_p
    if lo_all.any():
        g_lo = gmean[lo_all]
        agree = torch.maximum(pos_frac[lo_all], 1 - pos_frac[lo_all]).mean().item()
        push_down = (g_lo > 0).float().mean().item()
        print(f"\n  ALL low-p experts (n={int(lo_all.sum())}):")
        print(f"    mean d(loss)/d(theta) = {g_lo.mean().item():+.4e}   (>0 pushes p DOWN)")
        print(f"    fraction with g>0     = {push_down:.2f}   (1.0 = all want lower p)")
        print(f"    mean sign-agreement   = {agree:.2f}   (0.5 = pure noise, 1.0 = consistent)")
        # scale it: how far would theta actually move over a real run at this gradient?
        travel = g_lo.abs().mean().item() * 0.01 * 4000
        print(f"    implied theta travel at lr=0.01 over 4000 steps = {travel:.3f}")
        print(f"    (theta already sits at {torch.log(p[lo_all]/(1-p[lo_all])).mean().item():+.2f}; "
              f"a travel much smaller than 1 means the optimum is INTERIOR, not clipped)")


if __name__ == "__main__":
    main()
