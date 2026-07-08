"""STANDALONE re-eval of a saved checkpoint (the primary eval path is periodic, inside train.py).
Use this to re-score an old checkpoint or run the full suite (incl length-extrap) on demand.

  python -m ablate.common.run_eval --arm bibo_min --ckpt runs/bibo_min_seed0_final.pt --wandb
"""
from . import _paths  # noqa: F401
import os
import json
import argparse
import torch
from .models import build_arm
from . import patches as patchmod
from .evaluate import evaluate, Tok, summarize

DEV = "cuda"
_DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["qwen", "bibo_min"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--attn", choices=["sdpa", "flash_attention_4"], default="sdpa")
    ap.add_argument("--patches", default="liger_norm,liger_rope,moe")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--mcq_n", type=int, default=500)
    ap.add_argument("--extrap_lengths", default="1024,2048,4096")
    ap.add_argument("--no_probes", action="store_true")
    ap.add_argument("--with_global_mmlu", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="bibo-qwen-ablate")
    args = ap.parse_args()

    dt = _DT[args.precision]
    model, cfg = build_arm(args.arm, device=DEV, dtype=torch.float32, attn_impl=args.attn)
    patchmod.apply([p for p in args.patches.split(",") if p.strip() and p.strip() != "ce"])
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV), strict=False)
    tok = Tok()
    lengths = tuple(int(x) for x in args.extrap_lengths.split(",") if x.strip()) or None

    print(f"[eval {args.arm}] ckpt={args.ckpt} full suite (en+hi)...", flush=True)
    res, flat = evaluate(model, tok, seq_len=args.seq_len, mcq_n=args.mcq_n, extrap_lengths=lengths,
                         do_probes=not args.no_probes, with_global_mmlu=args.with_global_mmlu,
                         device=DEV, dtype=dt)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    out_path = os.path.join(out_dir, f"{args.arm}_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arm": args.arm, "ckpt": args.ckpt, **res}, f, indent=2, ensure_ascii=False)

    print("=== bits-per-byte (lower=better) ===", flush=True)
    for lang in sorted(res["bpb"]["per_language"]):
        print(f"  bpb[{lang}] = {res['bpb']['per_language'][lang]:.4f}", flush=True)
    if "length_extrap" in res:
        print("=== length-extrapolation (degradation >1 = worse at long ctx) ===", flush=True)
        for lang in sorted(res["length_extrap"]):
            d = res["length_extrap"][lang]
            print(f"  {lang}: " + "  ".join(f"L{L}={d[L]:.3f}" for L in lengths)
                  + f"  degradation={d['degradation']:.3f}", flush=True)
    print("=== LL-MCQ accuracy (higher=better) ===", flush=True)
    for lang in sorted(res["mcq"]["per_language"]):
        print(f"  acc[{lang}] = {res['mcq']['per_language'][lang]:.4f}", flush=True)
    if res.get("probes"):
        print("=== capability probes (en+hi) ===", flush=True)
        for lang in sorted(res["probes"]["per_language"]):
            print(f"  probe_acc[{lang}] = {res['probes']['per_language'][lang]:.4f}", flush=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f"{args.arm}_eval", config=vars(args))
        wandb.log(flat)
        wandb.finish()
    print(f"\n[eval {args.arm}] {summarize(flat)}  ->  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
