"""Ablation trainer — one arm, one seed, one run (~6h on RTX 6000). W&B for graphs/logs.

  python -m ablate.common.train --arm bibo_min --seed 0 --tokens 1_000_000_000 \
      --batch 40 --seq_len 1024 --precision bf16 --patches liger_norm,liger_rope,ce,moe --wandb

Local smoke:  --data synthetic --max_steps 5 --batch 2 --seq_len 128  (no --wandb)
No seed aggregation / variance machinery by design: pass --seed, read W&B; compare seeds there.
"""
from . import _paths  # noqa: F401
import os
import json
import time
import math
import argparse
import contextlib
import torch
from .models import build_arm, count_params
from . import patches as patchmod
from .optim import build_optimizers
from .schedule import make_wsd
from .data import token_batches, TRAIN_DATASET
from .evaluate import evaluate, Tok, summarize
from .eval.sample import generate_samples
from kernels.sm120.cross_entropy import fused_linear_cross_entropy   # sm120 (Blackwell); CE byte-identical to sm75

DEV = "cuda"
_DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


class _QwenAuxCollector:
    """Hooks Qwen's routers to grab per-layer gate logits (the vendored Qwen3MoeModel doesn't return them),
    so we can add the Switch-style load-balancing aux loss ourselves (Qwen's native balancing = fair vs
    BiBo's bias balancing). No-op for BiBo."""
    def __init__(self, model):
        self.logits = []
        self._handles = [m.register_forward_hook(self._hook)
                         for _, m in model.named_modules() if m.__class__.__name__ == "Qwen3MoeTopKRouter"]

    def _hook(self, module, inp, out):
        self.logits.append(out[0])           # (num_tokens, num_experts)

    def reset(self):
        self.logits = []


def _ce(model, ids, use_fused, aux=None, aux_coef=0.0, num_experts=9, top_k=2):
    inp, tgt = ids[:, :-1], ids[:, 1:].reshape(-1)
    if aux is not None:
        aux.reset()
    out = model.model(input_ids=inp, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    sh = h.reshape(-1, h.shape[-1])
    loss = (fused_linear_cross_entropy(sh, model.lm_head.weight, tgt) if use_fused
            else torch.nn.functional.cross_entropy(model.lm_head(sh), tgt))
    if aux is not None and aux_coef > 0 and aux.logits:      # Qwen aux load-balancing loss
        from baseline.qwen3moe.modeling import load_balancing_loss_func
        loss = loss + aux_coef * load_balancing_loss_func(tuple(aux.logits), num_experts, top_k)
    return loss


def _measure_peak_tflops(device, dtype, n=8192, iters=30):
    """Self-calibrating MFU denominator: achievable dense matmul TFLOPS on THIS gpu/dtype."""
    if device != "cuda":
        return 0.0
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    return (2 * n ** 3 * iters) / (time.time() - t) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["qwen", "bibo_min"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=1_000_000_000)
    ap.add_argument("--batch", type=int, default=40)             # per micro-step
    ap.add_argument("--grad_accum", type=int, default=1)          # global batch = batch * grad_accum
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")  # NEVER fp16
    ap.add_argument("--attn", choices=["sdpa", "flash_attention_4"], default="sdpa")
    ap.add_argument("--load_balance", choices=["none", "bias"], default="bias")   # BiBo: bias=DeepSeek sigmoid+balance; none=softmax
    ap.add_argument("--aux_coef", type=float, default=0.001)                      # Qwen aux load-balancing loss coef (0=off; paper 0.001)
    ap.add_argument("--polyglu_mult", type=int, default=3)                        # BiBo GLU experts = polyglu_mult*3 (= Qwen num_experts)
    ap.add_argument("--special_pairs", type=int, default=0)                       # BiBo param-free Identity/Zero pairs (specials = *2)
    ap.add_argument("--router_type", choices=["mlp", "conv"], default="mlp")       # BiBo router; conv -> sm120 fused-Triton conv kernel
    ap.add_argument("--kernel_size", type=int, default=3)                         # conv-router kernel width (only used when router_type=conv)
    ap.add_argument("--use_ssmax", action="store_true")                           # ablation axis: SSMax scalable softmax (default OFF)
    ap.add_argument("--use_xsa", action="store_true")                             # ablation axis: XSA exclusive self-attention (default OFF)
    ap.add_argument("--balance_exclude_specials", action="store_true")            # ablation axis: bias balancer ignores Identity/Zero experts (freezes their bias at 0; router learns special usage) — only matters with special_pairs>0
    ap.add_argument("--bias_update_threshold", type=int, default=10240)           # tokens between bias updates (if bias)
    ap.add_argument("--bias_update_factor", type=float, default=-1.0)             # <0 = auto Hill (~0.175 for 9 experts)
    ap.add_argument("--compile", action="store_true")           # torch.compile the transformer body
    ap.add_argument("--peak_tflops", type=float, default=0.0)   # MFU denominator: 0=auto-measure achievable GEMM;
    #                                                             else theoretical, e.g. 480 (dense bf16) / 960 (sparse)
    ap.add_argument("--patches", default="liger_norm,liger_rope,ce,moe")
    ap.add_argument("--muon_lr", type=float, default=3e-4)
    ap.add_argument("--adam_lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--decay_frac", type=float, default=0.20)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--dataset", default=TRAIN_DATASET)          # QTK-81K packed instruct corpus (HF id)
    ap.add_argument("--max_steps", type=int, default=0)      # >0 overrides token budget (smoke)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    # in-training eval -> W&B curves (this is the point; not a post-hoc-only eval)
    ap.add_argument("--eval_every", type=int, default=500)       # 0 disables; try 200/500/1000
    ap.add_argument("--sample_every", type=int, default=0)       # 0 = same as eval_every; steps between 2en+2hi samples
    ap.add_argument("--eval_mcq_n", type=int, default=200)       # cheap periodic MCQ sample
    ap.add_argument("--eval_bpb_n", type=int, default=200)       # cheap periodic bpb sample/source
    ap.add_argument("--eval_extrap", default="")                 # periodic length-extrap (default off; e.g. 1024,2048,4096)
    ap.add_argument("--final_mcq_n", type=int, default=500)      # full final eval
    ap.add_argument("--final_extrap", default="1024,2048,4096")
    ap.add_argument("--no_eval_icl", action="store_true")        # ICL-slope metric is ON by default (periodic + final)
    ap.add_argument("--eval_icl_n", type=int, default=50)        # periodic ICL items/lang/shot (final uses 100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="bibo-qwen-ablate")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dt = _DT[args.precision]
    patch_list = [p.strip() for p in args.patches.split(",") if p.strip()]
    use_fused_ce = "ce" in patch_list
    if args.router_type == "conv" and "router" not in patch_list:     # conv router -> use the fused sm120 kernel
        patch_list.append("router")

    model, cfg = build_arm(args.arm, device=DEV, dtype=torch.float32, attn_impl=args.attn,  # fp32 master weights
                           load_balance=args.load_balance, bias_update_threshold=args.bias_update_threshold,
                           bias_update_factor=(None if args.bias_update_factor < 0 else args.bias_update_factor),
                           aux_coef=args.aux_coef, polyglu_mult=args.polyglu_mult, special_pairs=args.special_pairs,
                           router_type=args.router_type, kernel_size=args.kernel_size,
                           use_ssmax=args.use_ssmax, use_xsa=args.use_xsa,
                           balance_exclude_specials=args.balance_exclude_specials)
    aux_collector = _QwenAuxCollector(model) if (args.arm == "qwen" and args.aux_coef > 0) else None
    total, trainable, active = count_params(model)
    patchmod.apply([p for p in patch_list if p != "ce"])              # ce handled in _ce()
    opts, n_mat, n_oth = build_optimizers(model, args.muon_lr, args.adam_lr, args.wd, ns_dtype=dt)
    if args.compile:                                            # compile the transformer body only; the
        model.model = torch.compile(model.model)               # triton/liger kernels stay eager (compiler.disable)
        print(f"[{args.arm}_seed{args.seed}] torch.compile(model.model) on; fused CE + liger/moe/flash kernels stay eager",
              flush=True)

    tok_per_step = args.batch * args.seq_len * args.grad_accum   # global batch
    total_steps = args.max_steps or (args.tokens // tok_per_step)
    scheds = make_wsd(opts, total_steps, args.warmup_frac, args.decay_frac)
    amp = contextlib.nullcontext() if args.precision == "fp32" else torch.autocast("cuda", dtype=dt)
    # include special_pairs + conv kernel so SE / conv-router variants don't collide on ckpt/log/run
    # names (they otherwise share arm+seed): e.g. bibo_min_seed2307_se1_conv5
    run_name = (f"{args.arm}_seed{args.seed}"
                + (f"_se{args.special_pairs}" if args.special_pairs else "")
                + ("_xsp" if args.balance_exclude_specials else "")
                + (f"_conv{args.kernel_size}" if args.router_type == "conv" else ""))
    out_dir = args.out or os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(out_dir, exist_ok=True)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=run_name,
                        config={**vars(args), "total_steps": total_steps,
                                "params_total": total, "params_active": active})

    # in-training eval (needs the real corpus/tokenizer + benchmark datasets)
    do_eval = args.eval_every > 0 and args.data == "real"
    tok = Tok() if do_eval else None
    if args.eval_every > 0 and not do_eval:
        print("[eval] disabled: --data synthetic (benchmark eval needs the real corpus + downloads)", flush=True)
    ev_extrap = tuple(int(x) for x in args.eval_extrap.split(",") if x.strip()) or None
    sample_every = args.sample_every if args.sample_every > 0 else args.eval_every   # default: sample when we eval

    print(f"[{run_name}] params total={total/1e6:.2f}M active={active/1e6:.2f}M | steps={total_steps} "
          f"tok/step={tok_per_step} patches={patch_list} {args.precision} attn={args.attn} "
          f"muon_mats={n_mat} eval_every={args.eval_every if do_eval else 'off'}", flush=True)

    gen = token_batches(args.batch, args.seq_len, DEV, dataset=args.dataset,
                        synthetic=(args.data == "synthetic"), vocab=cfg.vocab_size, seed=args.seed)
    # MFU denominator: measured achievable GEMM peak, or --peak_tflops (theoretical). FLOPs/token = 6N + attn.
    measured_peak = _measure_peak_tflops(DEV, dt)
    peak_tflops = args.peak_tflops if args.peak_tflops > 0 else measured_peak
    flops_per_token = 6 * active + 12 * cfg.num_hidden_layers * cfg.hidden_size * args.seq_len
    print(f"[{run_name}] MFU peak={peak_tflops:.0f} TFLOPS "
          f"({'set' if args.peak_tflops > 0 else 'measured GEMM'}); measured GEMM={measured_peak:.0f} | "
          f"flops/token ~{flops_per_token/1e9:.2f} GFLOP", flush=True)
    model.train()
    t0 = time.time(); _last_t = t0; _last_tok = 0; _last_step = 0
    for step in range(total_steps):
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(args.grad_accum):                 # gradient accumulation -> global batch
            ids = next(gen)
            with amp:
                loss = _ce(model, ids, use_fused_ce, aux_collector, args.aux_coef,
                           getattr(cfg, "num_experts", 9), cfg.num_experts_per_tok) / args.grad_accum
            loss.backward()
            loss_val += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else \
            torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        if step % args.log_every == 0 or step == total_steps - 1:
            lv, gn = loss_val, float(gnorm)
            lr = opts[0].param_groups[0]["lr"]
            toks = (step + 1) * tok_per_step
            _now = time.time(); _dt = _now - _last_t
            _steps_since = max(step - _last_step, 1)
            ms_per_step = 1000.0 * _dt / _steps_since                          # wall time per step
            tps = (toks - _last_tok) / _dt if _dt > 0 else 0.0                 # tokens/sec this interval
            mfu = 100.0 * flops_per_token * tps / (peak_tflops * 1e12) if peak_tflops > 0 else 0.0
            _last_t, _last_tok, _last_step = _now, toks, step
            mem = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
            elapsed = _now - t0                                                # total wall time so far
            eta = (total_steps - step - 1) * elapsed / max(step + 1, 1)        # est. time remaining
            fin = math.isfinite(lv)
            print(f"  step {step}/{total_steps} loss={lv:.4f} |g|={gn:.3f} lr={lr:.2e} tok={toks/1e6:.1f}M "
                  f"ms/step={ms_per_step:.0f} tps={tps/1e3:.1f}k mfu={mfu:.1f}% mem={mem:.1f}G "
                  f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                  f"{'' if fin else '  <<NON-FINITE>>'}", flush=True)
            if wb:
                wb.log({"train/loss": lv, "train/grad_norm": gn, "train/lr": lr, "train/ms_per_step": ms_per_step,
                        "train/tps": tps, "train/mfu": mfu, "train/mem_gb": mem, "train/elapsed_s": elapsed,
                        "tokens": toks}, step=step)
        if do_eval and step % args.eval_every == 0:            # periodic eval -> W&B curves
            _, flat = evaluate(model, tok, seq_len=args.seq_len, mcq_n=args.eval_mcq_n, bpb_n=args.eval_bpb_n,
                               extrap_lengths=ev_extrap, do_samples=False,
                               do_icl=not args.no_eval_icl, icl_n=args.eval_icl_n, device=DEV, dtype=dt)
            if wb:
                wb.log(flat, step=step)
            print(f"  [eval @{step}] {summarize(flat)}", flush=True)
        if do_eval and sample_every > 0 and step % sample_every == 0:   # samples on their own cadence (default = eval_every)
            for s in generate_samples(model, tok, device=DEV, dtype=dt):
                print(f"    [sample {s['lang']}] {s['prompt']} -> {s['completion']}", flush=True)
        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"{run_name}_step{step}.pt"))

    ckpt = os.path.join(out_dir, f"{run_name}_final.pt")
    torch.save(model.state_dict(), ckpt)
    final_eval = None
    if do_eval:
        fe = tuple(int(x) for x in args.final_extrap.split(",") if x.strip()) or None
        final_eval, full_flat = evaluate(model, tok, seq_len=args.seq_len, mcq_n=args.final_mcq_n,
                                         extrap_lengths=fe, do_icl=not args.no_eval_icl, icl_n=100,
                                         device=DEV, dtype=dt)
        if wb:
            wb.log(full_flat, step=total_steps)
        print(f"  [final eval] {summarize(full_flat)}", flush=True)
        for s in final_eval.get("samples", []):
            print(f"    [sample {s['lang']}] {s['prompt']} -> {s['completion']}", flush=True)
    res = {"arm": args.arm, "seed": args.seed, "steps": total_steps, "tokens": total_steps * tok_per_step,
           "final_loss": loss_val, "params_total": total, "params_active": active,
           "ckpt": ckpt, "wall_s": time.time() - t0, "eval": final_eval, "config": vars(args)}
    with open(os.path.join(out_dir, f"{run_name}_result.json"), "w") as f:
        json.dump(res, f, indent=2)
    if wb:
        wb.finish()
    print(f"[{run_name}] DONE final_loss={loss_val:.4f} ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
