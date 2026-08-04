"""STANDALONE re-eval of a saved checkpoint (the primary eval path is periodic, inside train.py).
Use this to re-score an old checkpoint or run the full suite (incl length-extrap) on demand.

  python -m ablate.common.run_eval --arm bibo_min --ckpt runs/bibo_min_seed0_final.pt --wandb

ARCHITECTURE COMES FROM THE CHECKPOINT'S OWN _result.json, not from flags here. This used to call
build_arm() with no architecture kwargs at all, so a 64-expert SWA+XSA checkpoint was loaded into a
default 6-expert global model -- and because the load was strict=False, every mismatched key was
dropped in silence and the eval reported confident nonsense. train.py already writes
`{"config": vars(args)}` into <run>_result.json next to the .pt, so the run is reproducible from
disk; this just reads it. The load is strict by default now, which turns that whole failure class
into an exception instead of a plausible number.
"""
from . import _paths  # noqa: F401
import os
import json
import argparse
import torch
from .models import build_arm
from .configs import SHARED, resolve_swa
from . import patches as patchmod
from .evaluate import evaluate, Tok, summarize

DEV = "cuda"
_DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _sidecar(ckpt):
    """<run>_final.pt / <run>_step900.pt -> <run>_result.json, or None."""
    base = os.path.basename(ckpt)
    for suf in ("_final.pt", ".pt"):
        if base.endswith(suf):
            stem = base[: -len(suf)]
            break
    stem = stem.split("_step")[0]
    p = os.path.join(os.path.dirname(os.path.abspath(ckpt)), f"{stem}_result.json")
    return p if os.path.exists(p) else None


def _arch_kwargs(c):
    """build_arm(**kwargs) reproducing the architecture a training run built from `c` = its saved
    vars(args). Every key defaulted, so a result.json written before an axis existed still loads."""
    pattern, window = resolve_swa(c.get("swa_pattern", "none"),
                                  c.get("sliding_window", 128),
                                  SHARED["num_hidden_layers"])
    bias_factor = c.get("bias_update_factor", -1)
    return dict(
        num_experts=c.get("experts") or None,
        special_pairs=c.get("special_pairs", 0),
        use_xsa=c.get("use_xsa", False),
        xsa_alpha_init=c.get("xsa_alpha_init", 0.0),
        pos_identity_expert=c.get("pos_identity_expert", True),
        neg_identity_expert=c.get("neg_identity_expert", True),
        top_k=(c.get("top_k") or None),
        moe_intermediate_size=(c.get("moe_inter") or None),
        num_shared_experts=c.get("n_shared", 0),
        hybrid_layer_pattern=pattern,
        sliding_window=window,
        swa_qk_norm=c.get("swa_qk_norm", True),
        attn_res=c.get("attn_res", "off"),
        attn_res_sites=c.get("attn_res_sites", 2),
        attn_res_carry=c.get("attn_res_carry", False),
        attn_res_fp32_stream=c.get("attn_res_fp32_stream", False),
        attn_res_carry_scale=c.get("attn_res_carry_scale", "none"),
        attn_res_emb_term=c.get("attn_res_emb_term", False),
        attn_res_emb_scale=c.get("attn_res_emb_scale", "none"),
        attn_res_emb_site=c.get("attn_res_emb_site", "mlp"),
        attn_res_emb_gain=c.get("attn_res_emb_gain", False),
        bias_update_threshold=c.get("bias_update_threshold", 10240),
        bias_update_factor=(None if bias_factor is not None and bias_factor < 0 else bias_factor),
        aux_coef=c.get("aux_coef", 0.001),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["qwen", "bibo_min"], required=True)
    ap.add_argument("--ckpt", required=True)
    # Architecture source. Defaults to the <run>_result.json sitting next to the checkpoint.
    ap.add_argument("--result_json", default=None,
                    help="training run's _result.json; auto-discovered from --ckpt if omitted")
    ap.add_argument("--allow_default_arch", action="store_true",
                    help="build the stock architecture when no _result.json exists (unsafe)")
    ap.add_argument("--loose_load", action="store_true",
                    help="strict=False state_dict load; hides architecture mismatches")
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

    # Architecture + activation from the run's own saved args. The activation lives on patchmod
    # (module-level, read at kernel-dispatch time), so it MUST be set before the first forward or
    # a silu checkpoint gets evaluated with radial experts.
    res_path = args.result_json or _sidecar(args.ckpt)
    kw = {}
    if res_path:
        saved = json.load(open(res_path))["config"]
        kw = _arch_kwargs(saved)
        patchmod.RADIAL_P = saved.get("radial_p", "sigmoid")
        patchmod.EXPERT_ACT = saved.get("act", "radial")
        print(f"[eval] architecture from {os.path.basename(res_path)}: "
              f"act={patchmod.EXPERT_ACT} experts={kw['num_experts']} top_k={kw['top_k']} "
              f"xsa={kw['use_xsa']} swa={kw['hybrid_layer_pattern']} window={kw['sliding_window']} "
              f"qk_norm={kw['swa_qk_norm']}", flush=True)
    elif args.allow_default_arch:
        print("[eval] WARNING: no _result.json found; building the DEFAULT architecture. This is "
              "only correct for a checkpoint trained with stock settings.", flush=True)
    else:
        raise SystemExit(
            f"no _result.json next to {args.ckpt} and --result_json not given. The architecture "
            f"cannot be inferred from the state dict, and guessing it produced silent nonsense "
            f"before this check existed. Pass --result_json, or --allow_default_arch if you are "
            f"certain the checkpoint is stock.")

    model, cfg = build_arm(args.arm, device=DEV, dtype=torch.float32, attn_impl=args.attn, **kw)
    patchmod.apply([p for p in args.patches.split(",") if p.strip() and p.strip() != "ce"])
    # strict by default: a shape/key mismatch means the rebuilt architecture is wrong, and a
    # silently partial load is indistinguishable from a real result downstream.
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV), strict=not args.loose_load)
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
