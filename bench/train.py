"""
BiBo Benchmark — Unified Training Script

Supports BiBo, BiBo variants, and Qwen3MoE via YAML config.
Both models get Liger kernels (RMSNorm, RoPE, SiLUMul, FusedCE) + dense MLP Triton.

Usage:
    python bench/train.py --config bench/configs/bibo.yaml
    python bench/train.py --config bench/configs/qwen3moe.yaml --batch_size 8
    torchrun --nproc_per_node=2 bench/train.py --config bench/configs/bibo.yaml
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import math
import time
import random
import argparse
import logging

import yaml
import numpy as np
import torch
import torch._dynamo
import torch._logging
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard

torch._dynamo.config.verbose = False
torch._dynamo.config.cache_size_limit = 64
torch._logging.set_logs(dynamo=logging.ERROR, inductor=logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BENCH_DIR)

from models import build_model_from_config, count_params, apply_triton_kernels, resize_embeddings
from data import load_benchmark_data, create_dataloader
from optim import create_optimizer, create_scheduler, HAS_BNB
from eval import evaluate, generate_samples, get_tokenizer, run_all_evals, evaluate_length_extrapolation
from metrics import (
    init_wandb, log_train_metrics, log_eval_metrics, log_samples,
    save_checkpoint, load_checkpoint, ThroughputMeter, format_time, format_count,
    MetricsCollector, estimate_mfu,
)


# Training precision. T4 (Turing) has no bf16, so fp16 is the only AMP option
# on the target hardware — kept explicit so a run never silently drifts to fp32/bf16.
AMP_DTYPE = torch.float16


def set_deterministic(seed, strict=False):
    """Make training reproducible: identical config + seed -> identical loss curve.

    strict=False (default): warn on ops with no deterministic CUDA impl but keep
      running. Curves are reproducible except for custom Triton atomic kernels.
    strict=True: hard-error on any non-deterministic op (use to hunt what breaks
      bit-exact reproducibility; will crash on custom atomic kernels).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 is non-deterministic vs fp32 and varies run-to-run on Ampere; off for
    # bit-stable fp32 master-weight/optimizer math (no effect on the fp16 matmuls).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def parse_args():
    p = argparse.ArgumentParser(description="BiBo Benchmark Training")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--muon_lr", type=float, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=None,
                   help="Cap HellaSwag/ARC examples per eval (default 500 via config). Keeps eval fast.")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--no_triton", action="store_true")
    p.add_argument("--no_fused_ce", action="store_true",
                   help="Keep Triton body kernels but use standard (compiled) CE instead of the "
                        "fused-linear-CE kernel. Faster when the (N,V) logits fit in memory.")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    # BiBo feature ablations (override the config; ignored for qwen). Accept hyphen or underscore.
    p.add_argument("--no-xsa", "--no_xsa", dest="no_xsa", action="store_true",
                   help="BiBo: disable Exclusive Self Attention (use_xsa=False)")
    p.add_argument("--no-ssmax", "--no_ssmax", dest="no_ssmax", action="store_true",
                   help="BiBo: disable SSMax query scaling (use_ssmax=False)")
    p.add_argument("--no-partial-rope", "--no_partial_rope", dest="no_partial_rope", action="store_true",
                   help="BiBo: disable partial RoPE / NoPE heads (rope_nope_ratio=0.0 → all-RoPE)")
    p.add_argument("--no-conv-router", "--no_conv_router", dest="no_conv_router", action="store_true",
                   help="BiBo: use the MLP router instead of the conv router (router_type='mlp')")
    return p.parse_args()


def load_config(args):
    """Load YAML config, apply CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t = cfg["training"]
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        t["grad_accum"] = args.grad_accum
    if args.total_steps is not None:
        t["total_steps"] = args.total_steps
    if args.lr is not None:
        t["lr"] = args.lr
    if args.muon_lr is not None:
        t["muon_lr"] = args.muon_lr
    if args.seq_len is not None:
        t["seq_len"] = args.seq_len
    if args.eval_every is not None:
        cfg["eval"]["eval_every"] = args.eval_every
    if args.max_eval_examples is not None:
        cfg["eval"]["max_eval_examples"] = args.max_eval_examples
    if args.wandb_name is not None:
        cfg["logging"]["wandb_name"] = args.wandb_name

    # BiBo feature ablations (override model config; the qwen builder ignores these keys)
    m = cfg["model"]
    ablated = []
    if args.no_xsa:          m["use_xsa"] = False;        ablated.append("xsa")
    if args.no_ssmax:        m["use_ssmax"] = False;      ablated.append("ssmax")
    if args.no_partial_rope: m["rope_nope_ratio"] = 0.0;  ablated.append("partial-rope")
    if args.no_conv_router:  m["router_type"] = "mlp";    ablated.append("conv-router")
    if ablated:
        cfg["_ablated"] = ablated   # surfaced in the startup banner

    return cfg


def setup_distributed():
    """Setup distributed training if available."""
    if not dist.is_available() or not dist.is_nccl_available():
        return False, 0, 1, "cpu"

    if "RANK" not in os.environ:
        if torch.cuda.is_available():
            return False, 0, 1, "cuda:0"
        return False, 0, 1, "cpu"

    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, f"cuda:{local_rank}"


def wrap_fsdp(model, device):
    """Wrap model with FSDP2 for multi-GPU."""
    for layer in model.model.layers:
        fully_shard(layer)
    fully_shard(model)
    model = model.to(device)
    return model


def compile_model(model):
    """Apply torch.compile — mode=default for Triton inductor fusion (same as old bench)."""
    return torch.compile(model, mode="default", fullgraph=False)


def train(args):
    TAG = "[bench]"

    is_distributed, rank, world_size, device = setup_distributed()
    is_main = rank == 0

    cfg = load_config(args)
    train_cfg = cfg["training"]
    eval_cfg = cfg["eval"]
    log_cfg = cfg["logging"]
    hw_cfg = cfg["hardware"]

    if train_cfg.get("deterministic", True):
        set_deterministic(train_cfg.get("seed", 42),
                          strict=train_cfg.get("strict_deterministic", False))

    model_type = cfg["model"]["type"]

    if is_main:
        print(f"{TAG} " + "=" * 60)
        print(f"{TAG} BiBo Benchmark — {model_type.upper()}")
        print(f"{TAG} " + "=" * 60)
        print(f"{TAG}   Device: {device} (world_size={world_size})")
        print(f"{TAG}   Batch: {train_cfg['batch_size']} x {train_cfg['grad_accum']} accum")
        print(f"{TAG}   Steps: {train_cfg['total_steps']}")
        print(f"{TAG}   Optimizer: {train_cfg.get('optimizer', 'muon_adamw8bit')}")
        print(f"{TAG}   Precision: AMP {str(AMP_DTYPE).replace('torch.', '')} (fp32 master weights)")
        if cfg.get("_ablated"):
            print(f"{TAG}   ⚠ ABLATED: {', '.join(cfg['_ablated'])} (disabled via CLI)")
        if train_cfg.get("deterministic", True):
            _strict = train_cfg.get("strict_deterministic", False)
            print(f"{TAG}   Deterministic: ON (seed={train_cfg.get('seed', 42)}, "
                  f"strict={_strict}) — identical config => identical curve")
            if not _strict:
                print(f"{TAG}     note: custom Triton atomic kernels remain non-bit-exact; "
                      f"use --no_triton for a bit-exact reference curve")
        print(f"{TAG} " + "=" * 60)

    # ── Data ──────────────────────────────────────────────────
    if is_main:
        print(f"{TAG} Loading dataset...")
    train_ds, val_ds = load_benchmark_data(
        seq_len=train_cfg.get("seq_len", 1024),
        val_split=train_cfg.get("val_split", 0.05),
        seed=train_cfg.get("seed", 42),
    )

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = get_tokenizer()
    actual_vocab_size = len(tokenizer)

    # The dataset is pre-tokenized with this tokenizer, so every id < vocab.
    # Size to max(config, tokenizer); the old 1-batch peek under-sized the
    # embedding whenever a later sample held a higher id (-> CUDA index assert).
    _safe_vocab = max(actual_vocab_size, cfg["model"]["vocab_size"])

    if is_main:
        print(f"{TAG} Tokenizer vocab: {actual_vocab_size}, safe: {_safe_vocab}")

    # ── Model ─────────────────────────────────────────────────
    if is_main:
        print(f"{TAG} Building {model_type} model...")
    model, config = build_model_from_config(cfg)

    resize_embeddings(model, config, _safe_vocab)

    stats = count_params(model, config)
    if is_main:
        print(f"{TAG} Params: {stats['total_m']:.2f}M total, {stats['active_m']:.2f}M active")
        print(f"{TAG} MoE layers: {stats['num_moe_layers']}, Dense: {stats['num_dense_layers']}")

    # ── Gradient Checkpointing ────────────────────────────────
    if getattr(args, 'grad_checkpoint', False):
        if is_main:
            print(f"{TAG} Gradient checkpointing ENABLED")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
    config.use_cache = False

    model = model.to(device)

    # ── Triton Kernels (Liger + Dense MLP for BOTH models) ────
    if not args.no_triton:
        # Standard (compiled) CE is the DEFAULT — measured 2.36x faster / 14.7% vs 6.2% MFU at
        # H512/V81920/B16 (the fused tl.dot-CE forward runs at ~2% SoL vs cuBLAS ~50%; it only
        # helps when the (N,V) logits OOM). Opt in with use_fused_ce: true (config) for that regime.
        use_fused_ce = bool(train_cfg.get("use_fused_ce", False)) and not args.no_fused_ce
        if is_main:
            print(f"{TAG} Applying Triton kernels... (CE: {'fused' if use_fused_ce else 'standard/compiled'})")
        apply_triton_kernels(model, config, no_triton=False, use_fused_ce=use_fused_ce)
    else:
        if is_main:
            print(f"{TAG} Triton DISABLED (--no_triton)")

    # ── FSDP2 ─────────────────────────────────────────────────
    if is_distributed:
        if is_main:
            print(f"{TAG} Wrapping with FSDP2...")
        model = wrap_fsdp(model, device)

    # ── torch.compile ─────────────────────────────────────────
    if not args.no_compile:
        if is_main:
            print(f"{TAG} Compiling model...")
        try:
            model = compile_model(model)
            if is_main:
                print(f"{TAG} torch.compile OK")
        except Exception as e:
            if is_main:
                print(f"{TAG} torch.compile failed: {e}")

    # ── Optimizer ─────────────────────────────────────────────
    optimizer, optim_name = create_optimizer(model, cfg)
    scheduler = create_scheduler(
        optimizer,
        train_cfg.get("warmup_steps", 1000),
        train_cfg.get("total_steps", 50000),
    )

    # ── DataLoader ────────────────────────────────────────────
    train_loader = create_dataloader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        seed=train_cfg.get("seed", 42),
    )

    # ── Metrics Collector ─────────────────────────────────────
    collector = MetricsCollector(model, enabled=log_cfg.get("log_internal_metrics", True))

    # ── WandB ─────────────────────────────────────────────────
    if not args.no_wandb and is_main:
        wandb_config = {
            "model": model_type,
            "optimizer": optim_name,
            "batch_size": train_cfg["batch_size"],
            "grad_accum": train_cfg["grad_accum"],
            "total_steps": train_cfg["total_steps"],
            "lr": train_cfg["lr"],
            "seq_len": train_cfg.get("seq_len", 1024),
            "seed": train_cfg.get("seed", 42),
            "world_size": world_size,
            "total_params": stats["total"],
            "active_params": stats["active"],
            "amp_dtype": str(AMP_DTYPE).replace("torch.", ""),
            "deterministic": train_cfg.get("deterministic", True),
            "strict_deterministic": train_cfg.get("strict_deterministic", False),
            "compile": not args.no_compile,
            "triton": not args.no_triton,
            "warmup_steps": train_cfg.get("warmup_steps", 1000),
            "weight_decay": train_cfg.get("weight_decay", 0.1),
            "grad_clip": train_cfg.get("grad_clip", 1.0),
        }
        init_wandb(wandb_config, project=log_cfg["wandb_project"],
                    name=log_cfg["wandb_name"], notes=f"{model_type} training")

    # ── Resume ────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume)

    # ── Eval Only ─────────────────────────────────────────────
    if args.eval_only:
        if is_main:
            results = run_all_evals(model, tokenizer, val_ds, device,
                                    eval_cfg.get("benchmarks", []),
                                    batch_size=train_cfg["batch_size"],
                                    max_examples=eval_cfg.get("max_eval_examples", 500))
            bpb = results.get("val_bpb")
            bpb_str = f", bpb: {bpb:.4f}" if bpb is not None else ""
            print(f"{TAG} Val loss: {results['val_loss']:.4f}, PPL: {results['val_ppl']:.2f}{bpb_str}")
            if "hellaswag" in results:
                print(f"{TAG} HellaSwag: acc={results['hellaswag']['accuracy']:.4f} acc_norm={results['hellaswag'].get('acc_norm',0):.4f}")
        return

    # ── Training Loop ─────────────────────────────────────────
    if is_main:
        print(f"\n{TAG} Starting training from step {start_step}...")
        print(f"{TAG} {len(train_ds)} train, {len(val_ds)} val samples")

    scaler = torch.amp.GradScaler()
    throughput = ThroughputMeter(warmup=3)
    model.train()
    epoch = 0
    step = start_step
    data_iter = iter(train_loader)
    accum_steps = train_cfg["grad_accum"]
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    # ── Fairness: Qwen gets its Switch-Transformer aux load-balancing loss.
    #    BiBo uses its own router-bias heuristic (no loss term). We log the
    #    pure LM loss for both so the loss curves stay apples-to-apples.
    is_qwen = model_type in ("qwen3moe", "qwen3_moe")
    aux_coef = cfg["model"].get("router_aux_loss_coef", 0.001)
    fwd_kwargs = {"output_router_logits": True} if is_qwen else {}
    active_params = stats["active"]
    step_times = []      # wall step times within the current log window (stall/recompile proxy)
    tokens_seen = 0      # cumulative tokens processed (for token-budget x-axis in WandB)

    # Router-entropy / hidden-norm hooks add per-step cost, so only run them when
    # log_internal_metrics is on. compute() then returns real values each log step.
    if collector.enabled:
        collector._active = True

    total_steps = train_cfg.get("total_steps", 50000)
    log_every = log_cfg.get("log_every", 10)
    eval_every = eval_cfg.get("eval_every", 500)
    sample_every = eval_cfg.get("sample_every", 1000)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=AMP_DTYPE):
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False,
                            **fwd_kwargs)
            # Extract loss — handle tuple, scalar, or nested returns
            if hasattr(outputs, "loss"):
                loss_val = outputs.loss
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                loss_val = outputs[0]
            else:
                raise RuntimeError(f"Unexpected model output: {type(outputs)}")
            if isinstance(loss_val, tuple):
                loss_val = loss_val[0]
            # loss_val = total (LM + coef*aux for Qwen). Backprop the total so
            # Qwen actually gets load balancing; log LM-only loss separately.
            aux = getattr(outputs, "aux_loss", None)
            lm_loss_val = (loss_val - aux_coef * aux) if aux is not None else loss_val
            loss = loss_val / accum_steps

        scaler.scale(loss).backward()
        micro_step += 1

        if micro_step % accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            step_time = time.time() - t0
            step_times.append(step_time)

            n_tokens = input_ids.numel() * accum_steps
            tokens_seen += n_tokens
            throughput.update(n_tokens)
            lm_loss = lm_loss_val.item()           # pure LM loss (comparable across models)
            total_loss = loss.item() * accum_steps  # LM + aux (what's actually optimized)

            # ── Logging ───────────────────────────────────────
            if is_main and (step % log_every == 0 or step == 1):
                lrs = scheduler.get_last_lr()  # one per param group (Muon + AdamW differ)
                lr_now = lrs[0]
                # Windowed (instantaneous) tps over the current log window — NOT cumulative.
                # Cumulative avg (throughput.tokens_per_sec) gets polluted by the compile step
                # and eval wall-time, making tps/mfu "drop" after every eval (a metric artifact,
                # not a real slowdown). Window = tokens this window / wall time this window.
                win_t = sum(step_times)
                tps = (n_tokens * len(step_times) / win_t) if win_t > 0 else 0.0
                gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                # Collect internal metrics (router entropy, hidden norms)
                internal = collector.compute() if collector.enabled else {}
                collector.reset()

                # ── Perf diagnostics (answers the "why is TPS low" question) ──
                mfu = estimate_mfu(active_params, tps, device)
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                # stall/recompile proxy: steps in window far above the median
                med = sorted(step_times)[len(step_times) // 2]
                step_time_max = max(step_times)
                slow_steps = sum(1 for t in step_times if t > 3 * med)
                internal.update({
                    "mfu": mfu,
                    "peak_mem_gb": peak_mem,
                    "step_time_max": step_time_max,
                    "slow_steps": slow_steps,
                    "loss_total": total_loss,
                    "perplexity": math.exp(min(lm_loss, 20)),
                    "tokens_seen": tokens_seen,
                    "tokens_trained": tokens_seen,   # numeric for WandB x-axis (UI auto-scales to K/M/B)
                    "epoch": epoch,
                    "samples_per_sec": tps / train_cfg.get("seq_len", 1024),
                    "loss_scale": scaler.get_scale(),   # fp16 health: drops on overflow
                    "lr_max": max(lrs),
                    "lr_min": min(lrs),
                })
                if aux is not None:
                    internal["aux_loss"] = aux.item()

                log_train_metrics(step, lm_loss, lr_now, gn, tps, step_time, extra=internal)
                step_times.clear()
                torch.cuda.reset_peak_memory_stats()

                pct = step / total_steps * 100
                eta = (total_steps - step) * step_time
                print(
                    f"  step {step:>6d}/{total_steps} ({pct:5.1f}%) | "
                    f"loss={lm_loss:.4f} | lr={lr_now:.2e} | tok={format_count(tokens_seen)} | "
                    f"tps={tps:.0f} | mfu={mfu*100:.1f}% | mem={peak_mem:.1f}G | "
                    f"{step_time:.3f}s | grad={gn:.4f} | ETA={format_time(eta)}"
                )
                if slow_steps:
                    print(f"    [perf] {slow_steps} stall(s) this window "
                          f"(max step {step_time_max:.2f}s) "
                          f"— likely recompile/launch-bound")

            # ── Validation + Benchmarks ───────────────────────
            if is_main and step % eval_every == 0:
                try:
                    with collector.track():
                        results = run_all_evals(
                            model, tokenizer, val_ds, device,
                            eval_cfg.get("benchmarks", []),
                            batch_size=train_cfg["batch_size"],
                            max_batches=100,
                            max_examples=eval_cfg.get("max_eval_examples", 500),
                        )
                    log_eval_metrics(step, results, tokens=tokens_seen)
                    _bpb = results.get("val_bpb")
                    _bpb_s = f" bpb={_bpb:.4f}" if _bpb is not None else ""
                    print(f"    [VAL] loss={results['val_loss']:.4f} ppl={results['val_ppl']:.2f}{_bpb_s}")
                    if "hellaswag" in results:
                        hs = results["hellaswag"]
                        print(f"    [BENCH] HellaSwag acc_norm={hs.get('acc_norm',0):.4f} "
                              f"margin={hs.get('margin',0):+.3f} auc={hs.get('auc',0):.3f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"    [VAL] OOM — skipping")
                    torch.cuda.empty_cache()

            # ── Sample Generation ─────────────────────────────
            if is_main and step % sample_every == 0:
                try:
                    samples = generate_samples(model, device=device,
                                               prompts=["The meaning of life is"],
                                               max_new_tokens=50)
                    log_samples(step, samples)
                    print(f"    [SAMPLE] {samples[0]['generated'][:100]}...")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # ── Checkpointing ─────────────────────────────────
            if is_main and step % (eval_every * 5) == 0:
                ckpt_path = os.path.join(REPO_ROOT, "bench", "checkpoints",
                                         f"{model_type}_step_{step}.pt")
                save_checkpoint(model, optimizer, scheduler, step, ckpt_path)

    # ── Final Summary ─────────────────────────────────────────
    if is_main:
        print(f"\n{TAG} " + "=" * 60)
        print(f"{TAG} Training Complete! ({step} steps, {epoch + 1} epochs)")

        try:
            results = run_all_evals(model, tokenizer, val_ds, device,
                                    eval_cfg.get("benchmarks", []),
                                    batch_size=train_cfg["batch_size"], max_batches=100,
                                    max_examples=eval_cfg.get("max_eval_examples", 1024))
            # #3 length-extrapolation bpb (real-data long-range): default = train len + 2x
            _sl = train_cfg.get("seq_len", 1024)
            le_lengths = eval_cfg.get("eval_lengths") or [_sl, 2 * _sl]
            try:
                results["length_extrap"] = evaluate_length_extrapolation(
                    model, le_lengths, train_cfg.get("val_split", 0.05),
                    train_cfg.get("seed", 42), train_cfg["batch_size"], device, tokenizer=tokenizer)
            except torch.cuda.OutOfMemoryError:
                print(f"{TAG} length-extrap OOM — skipping"); torch.cuda.empty_cache()
            log_eval_metrics(step, results, tokens=tokens_seen)
            print(f"{TAG} Final val loss: {results['val_loss']:.4f}")
            print(f"{TAG} Final perplexity: {results['val_ppl']:.2f}")
            if results.get("val_bpb") is not None:
                print(f"{TAG} Final bits-per-byte: {results['val_bpb']:.4f}")
            if "hellaswag" in results:
                hs = results["hellaswag"]
                print(f"{TAG} HellaSwag: acc_norm={hs.get('acc_norm',0):.4f} "
                      f"margin={hs.get('margin',0):+.3f} auc={hs.get('auc',0):.3f}")
            if "length_extrap" in results:
                le = results["length_extrap"]
                row = "  ".join(f"L{L}={m['bpb']:.4f}" for L, m in le["per_len"].items() if m.get("bpb"))
                print(f"{TAG} Length-extrap bpb: {row}  | degradation×={le['ratio']:.3f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{TAG} Final eval OOM")

        print(f"{TAG} " + "=" * 60)

        ckpt_path = os.path.join(REPO_ROOT, "bench", "checkpoints", f"{model_type}_final.pt")
        save_checkpoint(model, optimizer, scheduler, step, ckpt_path)

        collector.remove_hooks()

        if not args.no_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
