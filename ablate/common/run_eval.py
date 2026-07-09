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
    ap.add_argument("--icl", action="store_true", help="also run the separate ICL-slope metric (eval/icl_*)")
    ap.add_argument("--icl_n", type=int, default=100)
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
                         do_icl=args.icl, icl_n=args.icl_n, device=DEV, dtype=dt)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    out_path = os.path.join(out_dir, f"{args.arm}_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arm": args.arm, "ckpt": args.ckpt, **res}, f, indent=2, ensure_ascii=False)

    print("=== bits-per-byte (lower=better; se from per-text variance) ===", flush=True)
    for name, d in res["bpb"]["per_source"].items():
        print(f"  {name:12s} bpb={d['bpb']:.4f} +-{d['se']:.4f}  ({d['n_texts']} texts, {d['tokens']} tok)", flush=True)
    for lang in sorted(res["bpb"]["per_language"]):
        print(f"  bpb[{lang}] = {res['bpb']['per_language'][lang]:.4f}", flush=True)
    if "length_extrap" in res:
        print("=== length-extrapolation (degradation >1 = worse at long ctx) ===", flush=True)
        for lang in sorted(res["length_extrap"]):
            d = res["length_extrap"][lang]
            print(f"  {lang}: " + "  ".join(f"L{L}={d[L]:.3f}" for L in lengths)
                  + f"  degradation={d['degradation']:.3f}", flush=True)
    print("=== LL-MCQ accuracy (acc [95% CI] vs chance; z = sigmas above chance) ===", flush=True)
    for lang in sorted(res["mcq"]["per_language"]):
        d = res["mcq"]["per_language"][lang]
        print(f"  {lang}: acc={d['acc']:.3f} CI{d['ci95']} n={d['n']} chance={d['chance']:.3f} z={d['z_vs_chance']}", flush=True)
    for name, d in res["mcq"]["per_source"].items():
        print(f"    {name:12s} acc={d['acc']:.3f} CI{d['ci95']} ({d['n']}x{d['n_options']}-way, z={d['z_vs_chance']})", flush=True)
    it = res["interp"]
    print("=== MoE interp (expert utilization + router confidence) ===", flush=True)
    print(f"  balance_entropy={it['balance_entropy']} (1=balanced,0=collapsed)  load_cov={it['load_cov']}"
          f"  max/min load={it['max_expert_load']}/{it['min_expert_load']}", flush=True)
    print(f"  router: top1_weight={it['router_top1_weight']} entropy={it['router_entropy']} "
          f"frac(top1>0.5)={it['router_frac_top1_gt_0.5']}  | expert_load={it['expert_load']}", flush=True)
    if res.get("probes"):
        print("=== capability probes (en+hi) ===", flush=True)
        for lang in sorted(res["probes"]["per_language"]):
            print(f"  probe_acc[{lang}] = {res['probes']['per_language'][lang]['acc']:.4f}", flush=True)
    if res.get("icl"):
        print("=== ICL curve (SEPARATE metric; acc/nll vs shots; chance="
              f"{res['icl']['chance']}) ===", flush=True)
        for lang in sorted(res["icl"]["per_language"]):
            d = res["icl"]["per_language"][lang]
            accs = "  ".join(f"k{k}={d['acc'][k]:.3f}" for k in res["icl"]["shots"])
            nlls = "  ".join(f"k{k}={d['nll'][k]:.2f}" for k in res["icl"]["shots"])
            print(f"  [{lang}] acc: {accs}", flush=True)
            print(f"  [{lang}] nll: {nlls}", flush=True)
            print(f"  [{lang}] jump_acc(0->1)={d['jump_acc']}  slope_acc={d['slope_acc']}  "
                  f"slope_nll={d['slope_nll']}", flush=True)
    if res.get("samples"):
        print("=== samples (2 en + 2 hi, KV-cache decode) ===", flush=True)
        for s in res["samples"]:
            print(f"  [{s['lang']}] {s['prompt']} -> {s['completion']}", flush=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=f"{args.arm}_eval", config=vars(args))
        wandb.log(flat)
        wandb.finish()
    print(f"\n[eval {args.arm}] {summarize(flat)}  ->  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
