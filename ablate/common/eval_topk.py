"""Score ONE trained checkpoint at several inference-time top_k values.

The interp says effective-k is ~5.4 at top_k=8 (exp(H) of the routing weights) and w_min ~0.02, so
the 8th expert moves the output ~2% for a full expert's compute. That is suggestive, NOT proof that
k=6 is free: dropping k RENORMALIZES the surviving weights (w/sum w), so every remaining expert's
contribution changes -- it is not "delete a 2% term". This measures it instead of assuming it.

This is the CHEAP half of the k question and answers only the inference-time one. Whether a model
TRAINED at k=6 matches is a separate experiment; a k=8-trained router has learned a distribution
over 8 slots and may lean on the tail in ways a k=6-trained one would not.

    python -m ablate.common.eval_topk --ckpt runs/..._final.pt --experts 64 --ks 8,7,6,5,4
"""
from . import _paths  # noqa: F401
import argparse
import json

import torch

from .models import build_arm
from . import patches as patchmod
from .evaluate import evaluate, Tok

DEV = "cuda"


def set_top_k(model, k):
    """Retarget every router. Returns how many it touched -- 0 means the sweep is measuring nothing."""
    import src.modeling.ffn.router as rmod
    n = 0
    for m in model.modules():
        if isinstance(m, rmod.BiBoMoERouter):
            m.top_k = k
            n += 1
    model.config.num_experts_per_tok = k
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--special_pairs", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=8, help="the k the checkpoint was TRAINED at")
    ap.add_argument("--ks", default="8,7,6,5,4")
    ap.add_argument("--bpb_n", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--patches", default="liger_norm,liger_rope,moe")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.precision]
    model, cfg = build_arm("bibo_min", device=DEV, dtype=torch.float32, attn_impl="sdpa",
                           num_experts=a.experts, special_pairs=a.special_pairs, top_k=a.top_k)
    patchmod.apply([p for p in a.patches.split(",") if p.strip() and p.strip() != "ce"])

    sd = torch.load(a.ckpt, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    # strict=False on a config that does not match the checkpoint loads almost NOTHING and then
    # happily evaluates a randomly-initialised model. Refuse rather than report noise as a result.
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint does not match the built model: {len(missing)} missing / "
            f"{len(unexpected)} unexpected tensors. First missing: {missing[:4]}. "
            f"Check --experts/--special_pairs/--top_k against the run that produced it.")
    model.eval()

    tok = Tok()
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    rows = []
    for k in ks:
        touched = set_top_k(model, k)
        assert touched, "no BiBoMoERouter found -- top_k override did nothing"
        res, _ = evaluate(model, tok, seq_len=a.seq_len, mcq_n=0, bpb_n=a.bpb_n,
                          extrap_lengths=None, do_probes=False, do_samples=False,
                          do_icl=False, device=DEV, dtype=dt)
        b = res["bpb"]
        rows.append(dict(k=k, routers=touched, bpb=b["overall"],
                         **{f"bpb_{lang}": v for lang, v in b["per_language"].items()}))
        print(f"  k={k}  routers={touched}  bpb={b['overall']:.5f}  "
              + "  ".join(f"{lang}={v:.5f}" for lang, v in sorted(b["per_language"].items())),
              flush=True)

    base = next((r for r in rows if r["k"] == a.top_k), rows[0])
    FLOOR = 0.00037          # same-seed bpb spread at this config; see bibo-noise-floor
    print(f"\n  k   bpb        vs k={a.top_k}    xFLOOR   expert-FLOPs")
    for r in rows:
        d = r["bpb"] - base["bpb"]
        print(f"  {r['k']}   {r['bpb']:.5f}  {d:+.5f}   {abs(d)/FLOOR:6.2f}x   "
              f"{100.0*r['k']/a.top_k:5.1f}%")
    print(f"\n  floor = {FLOOR}; a delta under ~1x is not a real difference.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"ckpt": a.ckpt, "trained_top_k": a.top_k, "rows": rows}, f, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
