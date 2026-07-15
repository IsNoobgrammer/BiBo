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


@torch.no_grad()
def _expert_corr(model):
    """Mean cross-expert off-diagonal |cosine| over the 3D MoE expert stacks (0 = orthogonal experts,
    1 = identical). Diagnostic for xorth: does whitening actually decorrelate the experts over training?"""
    vals = []
    for n, p in model.named_parameters():
        if p.ndim == 3 and ("expert" in n or "gate_up_proj" in n or "down_proj" in n) and p.shape[0] > 1:
            e = p.shape[0]
            x = p.detach().reshape(e, -1).float()
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            m = x @ x.t()
            vals.append((m - torch.diag(torch.diagonal(m))).abs().sum().item() / (e * e - e))
    return sum(vals) / len(vals) if vals else 0.0


def _save_hf_ckpt(model, tokenizer, out_dir):
    """Write a reload-ready bf16 HF checkpoint (config.json + safetensors + tokenizer) to out_dir. Runs on
    the MAIN thread between steps (fast). Casts WEIGHTS to bf16 in a fresh state-dict COPY — the live fp32
    master weights are untouched (casting them in place would corrupt training) — and leaves buffers (RoPE
    inv_freq, router bias) at full precision. Unwraps torch.compile so the state-dict keys are clean
    (compiled model.model has `_orig_mod.` prefixes that would make the checkpoint unloadable)."""
    os.makedirs(out_dir, exist_ok=True)
    compiled = getattr(model, "model", None)
    orig = getattr(compiled, "_orig_mod", None)   # set iff model.model was torch.compile'd
    if orig is not None:
        model.model = orig
    prev_dtype = getattr(model.config, "torch_dtype", None)
    try:
        _params = set(n for n, _ in model.named_parameters())        # cast weights only, not buffers
        sd = {k: (v.to(torch.bfloat16) if (k in _params and v.is_floating_point()) else v)
              for k, v in model.state_dict().items()}
        model.config.torch_dtype = torch.bfloat16                    # so from_pretrained loads as bf16
        model.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
    finally:
        model.config.torch_dtype = prev_dtype
        if orig is not None:
            model.model = compiled
    tokenizer.save_pretrained(out_dir)
    return out_dir


def _push_hf_async(api, repo, local_dir, path_in_repo, tag):
    """Fire-and-forget upload (rayon-style): upload_folder runs in a background thread (run_as_future), so
    training continues immediately. On completion the local dir is reclaimed. Returns the Future to drain
    at process exit — a detached nohup process would otherwise kill the upload thread on exit."""
    fut = api.upload_folder(folder_path=local_dir, path_in_repo=path_in_repo, repo_id=repo,
                            commit_message=f"checkpoint {tag}", run_as_future=True)

    def _done(f, _d=local_dir, _t=tag):
        try:
            f.result()
            print(f"  [hf] pushed {_t} -> done", flush=True)
        except Exception as e:
            print(f"  [hf] push {_t} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
        finally:
            import shutil
            shutil.rmtree(_d, ignore_errors=True)   # reclaim disk once uploaded
    fut.add_done_callback(_done)
    return fut


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
    ap.add_argument("--special_pairs", type=int, default=0)                       # BiBo param-free special experts, per-type count
    ap.add_argument("--no_identity_expert", dest="identity_expert", action="store_false")  # drop Identity (code 3); test Zero alone
    ap.add_argument("--no_zero_expert", dest="zero_expert", action="store_false")          # drop Zero (code 4); test Identity alone
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
    ap.add_argument("--optimizer", choices=["muon", "manas"], default="muon")  # manas = Muon + long-memory lookahead probe
    ap.add_argument("--probe_gamma", type=float, default=0.08)     # Manas probe step (couples to LR/geometry)
    ap.add_argument("--probe_rho", type=float, default=0.98)       # Manas probe memory (SAMPLES; re-derive for batch size)
    ap.add_argument("--manas_rank", type=int, default=8)           # Manas low-rank probe rank
    ap.add_argument("--probe_warmup_steps", type=int, default=0)   # skip probe (d) accumulation for first N steps
    ap.add_argument("--manas_comp", type=float, default=0.0)       # Manas U buffer strength in units of gamma (0=off; +1=extend, toy champ)
    ap.add_argument("--rgd_tau", type=float, default=0.0)          # RGD loss-weighted probe voting (0=off/equal votes; e.g. 3.0)
    ap.add_argument("--probe_norm", choices=["global", "perparam"], default="global")  # probe grad-norm: global scalar | per-matrix (muP-ish)
    ap.add_argument("--cos_beta", type=float, default=0.0)         # cos(g,d) vote weight: 0=off/equal; +sharpen agreeing, -favor novelty
    ap.add_argument("--micro_vote", action="store_true")           # manas: one probe vote per MICRO-batch (rho binds to micro batch); needs low-rank probe
    ap.add_argument("--nexus_gamma", type=float, default=0.0)      # manas: common-basin walker strength (needs --micro_vote); ceiling = nexus_gamma*grad_accum
    ap.add_argument("--probe_rho_step", type=float, default=None)   # manas two-clock (needs --micro_vote): probe_rho=within-step vote decay, this=per-step-boundary block decay; reach=gamma/(1-rho_step)
    ap.add_argument("--probe_gamma_intra", type=float, default=None) # manas fresh-block (needs --probe_rho_step): per-vote weight of THIS step's fresh block; folds to history at probe_gamma. None/=gamma -> plain two-clock
    ap.add_argument("--probe_refresh", type=int, default=None)      # manas basis rebuild cadence (probe updates); None -> auto 2/(1-rho_step)
    ap.add_argument("--probe_sketch_rho", type=float, default=None) # manas EMA-sketch window (needs low-rank): Q aims at where grads live (memory 1/(1-rho_q)); None -> one-sample snapshot
    ap.add_argument("--probe_sketch_votes", action="store_true")    # manas two-clock window (needs --probe_sketch_rho --micro_vote): unit-normalized per-vote sketch covers vote distribution vs collapsed mean
    ap.add_argument("--probe_sketch_min_votes", type=int, default=2) # manas GA1 gate: min votes/step for sketch aim (else snapshot); default 2, set 1 to force sketch at ga1
    ap.add_argument("--sketch_gate", type=int, default=None)        # manas: instance override of _sketch_gate (min votes/step for sketch aim); set 1 to force union window at ga1
    ap.add_argument("--gamma_decay", action="store_true")           # manas: probe_gamma tracks the muon LR schedule (warmup+decay); default OFF = gamma constant over the run
    ap.add_argument("--muon_scale_mode", choices=["polar", "normuon", "aurora", "aurora_ema", "aurora_ema_v2"],
                    default="aurora")  # post-NS row scaling; EMA variants: normuon / aurora_ema / aurora_ema_v2
    ap.add_argument("--xorth_post", type=float, default=0.0)       # cross-expert whitening MAX strength (0=off), scoped to MoE expert stacks
    ap.add_argument("--xorth_gate_ref", type=float, default=0.3)   # correlation gate: full whitening at off-diag RMS>=this; below it ramps to ~0; <=0 disables gate
    ap.add_argument("--xorth_ema", type=float, default=0.95)       # EMA decay of the persistent per-stack (E,E) gram
    ap.add_argument("--xorth_warmup_steps", type=int, default=0)   # gate xorth OFF until step > this (0 = active from step 1)
    ap.add_argument("--xorth_where", choices=["pre", "post"], default="post")  # whiten momentum PRE-NS or orthogonalized update POST-NS
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
    ap.add_argument("--hf_repo", default="")     # if set, push model+tokenizer to this HF repo every --ckpt_every steps (async, non-blocking)
    ap.add_argument("--hf_token", default="")    # HF WRITE token; falls back to $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN
    ap.add_argument("--hf_private", action="store_true")  # create the repo private
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
                           balance_exclude_specials=args.balance_exclude_specials,
                           identity_expert=args.identity_expert, zero_expert=args.zero_expert)
    aux_collector = _QwenAuxCollector(model) if (args.arm == "qwen" and args.aux_coef > 0) else None
    total, trainable, active = count_params(model)
    patchmod.apply([p for p in patch_list if p != "ce"])              # ce handled in _ce()
    opts, n_mat, n_oth = build_optimizers(model, args.muon_lr, args.adam_lr, args.wd, ns_dtype=dt,
                                          scale_mode=args.muon_scale_mode, xorth_post=args.xorth_post,
                                          xorth_gate_ref=args.xorth_gate_ref, xorth_ema=args.xorth_ema,
                                          xorth_warmup_steps=args.xorth_warmup_steps, xorth_where=args.xorth_where,
                                          optimizer=args.optimizer, probe_gamma=args.probe_gamma,
                                          probe_rho=args.probe_rho,
                                          manas_rank=(args.manas_rank if args.manas_rank > 0 else None),  # 0 -> full rank (probe_rank=None)
                                          probe_warmup_steps=args.probe_warmup_steps, manas_comp=args.manas_comp,
                                          rgd_tau=(args.rgd_tau or None), probe_norm=args.probe_norm,
                                          cos_beta=args.cos_beta, micro_vote=args.micro_vote,
                                          nexus_gamma=args.nexus_gamma, probe_rho_step=args.probe_rho_step,
                                          probe_gamma_intra=args.probe_gamma_intra,
                                          probe_refresh=args.probe_refresh,
                                          probe_sketch_rho=args.probe_sketch_rho,
                                          probe_sketch_votes=args.probe_sketch_votes,
                                          probe_sketch_min_votes=args.probe_sketch_min_votes)
    manas = opts[0] if args.optimizer == "manas" else None   # needs probe() around fwd/bwd (see loop)
    if manas is not None and args.sketch_gate is not None:
        manas._sketch_gate = args.sketch_gate                 # instance override of the ga1 sketch gate
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
                + (("_idonly" if not args.zero_expert else "") if args.special_pairs else "")
                + (("_zeroonly" if not args.identity_expert else "") if args.special_pairs else "")
                + ("_xsp" if args.balance_exclude_specials else "")
                + (f"_{args.muon_scale_mode}" if args.muon_scale_mode != "aurora" else "")
                + (f"_manas_g{args.probe_gamma:g}r{args.probe_rho:g}" + (f"w{args.probe_warmup_steps}" if args.probe_warmup_steps else "")
                   + (f"c{args.manas_comp:g}" if args.manas_comp else "")
                   + (f"rgd{args.rgd_tau:g}" if args.rgd_tau else "")
                   + ("pp" if args.probe_norm == "perparam" else "")
                   + (f"cb{args.cos_beta:g}" if args.cos_beta else "")
                   + ("mv" if args.micro_vote else "")
                   + (f"nx{args.nexus_gamma:g}" if args.nexus_gamma else "")
                   + (f"rs{args.probe_rho_step:g}" if args.probe_rho_step else "")
                   + (f"gi{args.probe_gamma_intra:g}" if args.probe_gamma_intra else "")
                   + (f"rf{args.probe_refresh:g}" if args.probe_refresh else "")
                   + (f"sk{args.probe_sketch_rho:g}" if args.probe_sketch_rho else "")
                   + ("skv" if args.probe_sketch_votes else "")
                   + (f"mv{args.probe_sketch_min_votes}" if args.probe_sketch_min_votes != 2 else "")
                   + (f"sg{args.sketch_gate}" if args.sketch_gate is not None else "")
                   if args.optimizer == "manas" else "")
                + (f"_xo{args.xorth_post:g}{args.xorth_where}" if args.xorth_post > 0 else "")
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

    # async HF checkpoint push: save_pretrained locally (main thread), then upload_folder in the background.
    hf_api = hf_tok = None
    hf_futures = []
    if args.hf_repo:
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        from .evaluate import TOKENIZER
        _hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        hf_api = HfApi(token=_hf_token)
        hf_api.create_repo(args.hf_repo, private=args.hf_private, exist_ok=True)
        hf_tok = tok._t if tok is not None else AutoTokenizer.from_pretrained(TOKENIZER)   # reuse the eval tokenizer
        print(f"[{run_name}] HF push -> {args.hf_repo} every {args.ckpt_every} steps (async, non-blocking); "
              f"periodic -> step<N>/, final -> repo root", flush=True)

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
        if manas is not None and args.gamma_decay:       # probe dose tracks the muon LR schedule (else constant)
            manas.probe_gamma = args.probe_gamma * (opts[0].param_groups[0]["lr"] / args.muon_lr)
        loss_val = 0.0
        # Manas: run each micro-batch's fwd/bwd at theta+d (probe), then vote() OUTSIDE the probe so d
        # (and the nexus walker) advance per micro-batch; step() below applies the Muon update on the
        # accumulated probe-point grads. vote() is a no-op when micro_vote=False -> the same loop runs
        # both modes, and step-voting is numerically unchanged (d is constant within a step, so per-micro
        # apply/remove == one probe around all micros). Plain Muon uses nullcontext + no vote.
        for _ in range(args.grad_accum):                     # gradient accumulation -> global batch
            ids = next(gen)
            with (manas.probe() if manas is not None else contextlib.nullcontext()):
                with amp:
                    loss = _ce(model, ids, use_fused_ce, aux_collector, args.aux_coef,
                               getattr(cfg, "num_experts", 9), cfg.num_experts_per_tok) / args.grad_accum
                loss.backward()
            loss_val += loss.item()
            if manas is not None:
                manas.vote()                                 # no-op unless micro_vote=True
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else \
            torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters() if p.grad is not None))
        for o in opts:
            o.step(probe_loss=loss_val) if o is manas else o.step()   # probe_loss needed iff rgd_tau set; ignored otherwise
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
            ecorr = _expert_corr(model)                                    # cross-expert redundancy — logged every run
            print(f"  step {step}/{total_steps} loss={lv:.4f} |g|={gn:.3f} lr={lr:.2e} tok={toks/1e6:.1f}M "
                  f"ms/step={ms_per_step:.0f} tps={tps/1e3:.1f}k mfu={mfu:.1f}% mem={mem:.1f}G "
                  f"xcorr={ecorr:.4f} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                  f"{'' if fin else '  <<NON-FINITE>>'}", flush=True)
            if wb:
                wb.log({"train/loss": lv, "train/grad_norm": gn, "train/lr": lr, "train/ms_per_step": ms_per_step,
                        "train/tps": tps, "train/mfu": mfu, "train/mem_gb": mem, "train/elapsed_s": elapsed,
                        "train/expert_corr": ecorr, "tokens": toks}, step=step)
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
            if hf_api is not None:
                _dir = _save_hf_ckpt(model, hf_tok, os.path.join(out_dir, f"{run_name}_step{step}"))
                hf_futures.append(_push_hf_async(hf_api, args.hf_repo, _dir, f"step{step}",
                                                 f"{run_name} step{step}"))
            else:
                torch.save(model.state_dict(), os.path.join(out_dir, f"{run_name}_step{step}.pt"))

    ckpt = os.path.join(out_dir, f"{run_name}_final.pt")
    torch.save(model.state_dict(), ckpt)
    if hf_api is not None:                                  # final -> repo root so `from_pretrained(repo)` just works
        _dir = _save_hf_ckpt(model, hf_tok, os.path.join(out_dir, f"{run_name}_final"))
        hf_futures.append(_push_hf_async(hf_api, args.hf_repo, _dir, "final", f"{run_name} final"))
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
    if hf_futures:                                         # drain: block until every background upload lands,
        print(f"[{run_name}] waiting on {len(hf_futures)} HF upload(s) before exit...", flush=True)
        for f in hf_futures:                               # else this detached process exits and kills the threads
            try:
                f.result()
            except Exception:
                pass
    if wb:
        wb.finish()
    print(f"[{run_name}] DONE final_loss={loss_val:.4f} ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
