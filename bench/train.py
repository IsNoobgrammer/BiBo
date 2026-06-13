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
from eval import evaluate, generate_samples, get_tokenizer, run_all_evals
from metrics import (
    init_wandb, log_train_metrics, log_eval_metrics, log_samples,
    save_checkpoint, load_checkpoint, ThroughputMeter, format_time,
    MetricsCollector, estimate_mfu,
)


def set_deterministic(seed):
    """Make training fully deterministic for fair comparison."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--no_triton", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--resume", type=str, default=None)
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
    if args.wandb_name is not None:
        cfg["logging"]["wandb_name"] = args.wandb_name

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
    """Apply torch.compile — uses reduce-overhead mode to avoid OOM from buffer pre-allocation."""
    return torch.compile(model, mode="reduce-overhead", fullgraph=False)


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
        set_deterministic(train_cfg.get("seed", 42))

    model_type = cfg["model"]["type"]

    if is_main:
        print(f"{TAG} " + "=" * 60)
        print(f"{TAG} BiBo Benchmark — {model_type.upper()}")
        print(f"{TAG} " + "=" * 60)
        print(f"{TAG}   Device: {device} (world_size={world_size})")
        print(f"{TAG}   Batch: {train_cfg['batch_size']} x {train_cfg['grad_accum']} accum")
        print(f"{TAG}   Steps: {train_cfg['total_steps']}")
        print(f"{TAG}   Optimizer: {train_cfg.get('optimizer', 'muon_adamw8bit')}")
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

    _peek_loader = create_dataloader(train_ds, batch_size=1, shuffle=False)
    _peek_batch = next(iter(_peek_loader))
    _max_id = _peek_batch["input_ids"].max().item()
    del _peek_loader, _peek_batch
    _safe_vocab = max(actual_vocab_size, _max_id + 1)

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
        if is_main:
            print(f"{TAG} Applying Triton kernels...")
        apply_triton_kernels(model, config, no_triton=False)
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
                                    eval_cfg.get("benchmarks", []))
            print(f"{TAG} Val loss: {results['val_loss']:.4f}, PPL: {results['val_ppl']:.2f}")
            if "hellaswag" in results:
                print(f"{TAG} HellaSwag: {results['hellaswag']['accuracy']:.4f}")
            if "arc_challenge" in results:
                print(f"{TAG} ARC-Challenge: {results['arc_challenge']['accuracy']:.4f}")
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

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss / accum_steps

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

            n_tokens = input_ids.numel() * accum_steps
            throughput.update(n_tokens)
            unscaled_loss = loss.item() * accum_steps

            # ── Logging ───────────────────────────────────────
            if is_main and (step % log_every == 0 or step == 1):
                lr_now = scheduler.get_last_lr()[0]
                tps = throughput.tokens_per_sec()
                gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                # Collect internal metrics
                internal = collector.compute() if collector.enabled else {}
                collector.reset()

                log_train_metrics(step, unscaled_loss, lr_now, gn, tps, step_time, extra=internal)

                pct = step / total_steps * 100
                eta = (total_steps - step) * step_time
                print(
                    f"  step {step:>6d}/{total_steps} ({pct:5.1f}%) | "
                    f"loss={unscaled_loss:.4f} | lr={lr_now:.2e} | "
                    f"tps={tps:.0f} | {step_time:.3f}s | "
                    f"grad={gn:.4f} | ETA={format_time(eta)}"
                )

            # ── Validation + Benchmarks ───────────────────────
            if is_main and step % eval_every == 0:
                try:
                    with collector.track():
                        results = run_all_evals(
                            model, tokenizer, val_ds, device,
                            eval_cfg.get("benchmarks", []),
                            max_batches=100,
                        )
                    log_eval_metrics(step, results)
                    print(f"    [VAL] loss={results['val_loss']:.4f} ppl={results['val_ppl']:.2f}")
                    if "hellaswag" in results:
                        print(f"    [BENCH] HellaSwag={results['hellaswag']['accuracy']:.4f}")
                    if "arc_challenge" in results:
                        print(f"    [BENCH] ARC={results['arc_challenge']['accuracy']:.4f}")
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
                                    eval_cfg.get("benchmarks", []), max_batches=100)
            log_eval_metrics(step, results)
            print(f"{TAG} Final val loss: {results['val_loss']:.4f}")
            print(f"{TAG} Final perplexity: {results['val_ppl']:.2f}")
            if "hellaswag" in results:
                print(f"{TAG} HellaSwag: {results['hellaswag']['accuracy']:.4f}")
            if "arc_challenge" in results:
                print(f"{TAG} ARC-Challenge: {results['arc_challenge']['accuracy']:.4f}")
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
