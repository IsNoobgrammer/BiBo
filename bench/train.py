"""
BiBo Benchmark — Main Training Script

Usage:
    # Single GPU
    python bench/train.py

    # Multi-GPU (2×T4)
    torchrun --nproc_per_node=2 bench/train.py

    # With custom args
    torchrun --nproc_per_node=2 bench/train.py --batch_size 16 --total_steps 50000
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import math
import time
import argparse
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard

# Ensure repo root and bench dir are in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, BENCH_DIR)

from config import BIBO_50M_BASELINE, build_model, count_params
from data import load_benchmark_data, create_dataloader
from optim import create_optimizer, create_scheduler, HAS_MUON, HAS_BNB
from eval import evaluate, generate_samples, get_tokenizer
from utils import (
    init_wandb, log_train_metrics, log_val_metrics, log_samples,
    save_checkpoint, load_checkpoint, ThroughputMeter, format_time,
)


def parse_args():
    p = argparse.ArgumentParser(description="BiMo Benchmark Training")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--total_steps", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--val_split", type=float, default=0.05)

    # Eval & Logging
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=10)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="bench/checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # WandB
    p.add_argument("--wandb_project", type=str, default="bibo-bench")
    p.add_argument("--wandb_name", type=str, default="baseline-50m")
    p.add_argument("--wandb_notes", type=str, default="BiBo ~50M baseline, 2×T4, QTK-81K")

    # Mode
    p.add_argument("--eval_only", action="store_true", help="Run eval only, no training")
    p.add_argument("--no_compile", action="store_true", help="Skip torch.compile")
    p.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    return p.parse_args()


def setup_distributed():
    """Setup distributed training if available."""
    if not dist.is_available() or not dist.is_nccl_available():
        return False, 0, 1, "cpu"

    if "RANK" not in os.environ:
        # Single GPU
        if torch.cuda.is_available():
            return False, 0, 1, "cuda:0"
        return False, 0, 1, "cpu"

    # Multi-GPU via torchrun
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
    """Apply torch.compile with settings safe for MoE."""
    return torch.compile(model, mode="reduce-overhead", fullgraph=False)


def train(args):
    """Main training loop."""
    # ── Setup ──────────────────────────────────────────────────
    is_distributed, rank, world_size, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if is_main:
        print("=" * 60)
        print("BiBo Benchmark Training")
        print("=" * 60)
        print(f"  Device: {device} (world_size={world_size})")
        print(f"  Muon: {HAS_MUON}, bitsandbytes: {HAS_BNB}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Total steps: {args.total_steps}")
        print(f"  Seq len: {args.seq_len}")
        print(f"  LR: {args.lr}, Muon LR: {args.muon_lr}")
        print("=" * 60)

    # ── Data (load first — need max token ID for vocab sizing) ───
    if is_main:
        print("[train] Loading dataset...")
    train_ds, val_ds = load_benchmark_data(seq_len=args.seq_len, val_split=args.val_split)

    # ── Tokenizer ───────────────────────────────────────────────
    if is_main:
        print("[train] Loading tokenizer...")
    tokenizer = get_tokenizer()
    actual_vocab_size = len(tokenizer)  # includes added tokens, not just base BPE

    # Quick sanity check — 1 batch, just to verify data loads
    _peek_loader = create_dataloader(train_ds, batch_size=1, shuffle=False)
    _peek_batch = next(iter(_peek_loader))
    _max_id = _peek_batch["input_ids"].max().item()
    del _peek_loader, _peek_batch
    _safe_vocab = max(actual_vocab_size, _max_id + 1)
    if is_main:
        print(f"[train] Tokenizer: vocab_size={tokenizer.vocab_size}, len={actual_vocab_size}")
        print(f"[train] Dataset max token ID: {_max_id}, safe vocab: {_safe_vocab}")

    # ── Model ──────────────────────────────────────────────────
    if is_main:
        print("[train] Building model...")
    model, config = build_model()

    # Resize embeddings to cover all token IDs
    if _safe_vocab != config.vocab_size:
        if is_main:
            print(f"[train] Resizing embeddings: {config.vocab_size} -> {_safe_vocab}")
        model.resize_token_embeddings(_safe_vocab)
        config.vocab_size = _safe_vocab

    stats = count_params(config)
    if is_main:
        print(f"[train] Model params: {stats['total_m']:.2f}M total, {stats['active_m']:.2f}M active, ratio={stats['ratio']:.2f}x")
        print(f"[train] Experts: {config.num_routed_experts} routed + {config.num_shared_experts} shared, top-{config.num_experts_per_tok}")
        print(f"[train] Layers: {config.num_hidden_layers} ({stats['num_moe_layers']} MoE + {stats['num_dense_layers']} dense)")

    # Gradient checkpointing — saves ~40% activation memory
    # MoE routing is data-dependent (different experts each recomputation),
    # so we must skip the strict metadata check. Two approaches:
    #   1. use_reentrant=True  — old checkpoint, no shape check (Mixtral's fix)
    #   2. use_reentrant=False, determinism_check='none' — new, skip check
    # Use approach 1 (proven with Mixtral 8x7B MoE)
    if is_main:
        print("[train] Enabling gradient checkpointing (use_reentrant=True for MoE)...")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    config.use_cache = False  # incompatible with grad checkpointing

    model = model.to(device)

    # FSDP2 for multi-GPU
    if is_distributed:
        if is_main:
            print("[train] Wrapping with FSDP2...")
        model = wrap_fsdp(model, device)

    # torch.compile
    if not args.no_compile:
        if is_main:
            print("[train] Compiling model with torch.compile...")
        try:
            model = compile_model(model)
            if is_main:
                print("[train] torch.compile OK")
        except Exception as e:
            if is_main:
                print(f"[train] torch.compile failed: {e}, continuing in eager mode")

    # ── Optimizer & Scheduler ──────────────────────────────────
    optimizer = create_optimizer(model, lr=args.lr, muon_lr=args.muon_lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.total_steps)

    # ── DataLoader ───────────────────────────────────────────────
    train_loader = create_dataloader(train_ds, batch_size=args.batch_size)

    # ── WandB ──────────────────────────────────────────────────
    if not args.no_wandb and is_main:
        wandb_config = {
            "model": "BiBo-baseline",
            # Training
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "total_steps": args.total_steps,
            "warmup_steps": args.warmup_steps,
            "lr": args.lr,
            "muon_lr": args.muon_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seq_len": args.seq_len,
            "seed": args.seed,
            # Hardware
            "world_size": world_size,
            "has_muon": HAS_MUON,
            "compiled": not args.no_compile,
            # Architecture
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "moe_layers": stats["num_moe_layers"],
            "dense_layers": stats["num_dense_layers"],
            "num_routed_experts": config.num_routed_experts,
            "num_shared_experts": config.num_shared_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "polyglu_expert_multiplier": config.polyglu_expert_multiplier,
            "special_expert_pairs": config.special_expert_pairs,
            "moe_intermediate_size": config.moe_intermediate_size,
            "shared_expert_type": config.shared_expert_type,
            "router_type": config.router_type,
            "router_lambda": config.router_lambda,
            "use_ssmax": config.use_ssmax,
            # Param counts
            "total_params": stats["total"],
            "total_params_m": stats["total_m"],
            "active_params": stats["active"],
            "active_params_m": stats["active_m"],
            "param_ratio": stats["ratio"],
            "embed_params": stats["embed"],
            "attn_params": stats["attn"],
            "dense_params": stats["dense"],
            "moe_routed_params": stats["moe_routed"],
            "moe_shared_params": stats["moe_shared"],
        }
        init_wandb(wandb_config, project=args.wandb_project, name=args.wandb_name, notes=args.wandb_notes)

    # ── Resume ─────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume)

    # ── Eval Only ──────────────────────────────────────────────
    if args.eval_only:
        if is_main:
            val_loss, val_ppl = evaluate(model, val_ds, batch_size=args.batch_size, device=device)
            print(f"[eval] Val loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
            samples = generate_samples(model, device=device)
            for s in samples:
                print(f"\n  Prompt: {s['prompt']}")
                print(f"  Generated: {s['generated'][:200]}...")
        return

    # ── Training Loop ──────────────────────────────────────────
    if is_main:
        print(f"\n[train] Starting training from step {start_step}...")
        print(f"[train] {len(train_ds)} train samples, {len(val_ds)} val samples")
        print(f"[train] {len(train_loader)} batches per epoch")
        print()

    throughput = ThroughputMeter(warmup=3)
    model.train()
    epoch = 0
    step = start_step
    data_iter = iter(train_loader)
    accum_steps = args.grad_accum
    micro_step = 0  # tracks micro-steps within an accumulation window
    optimizer.zero_grad(set_to_none=True)

    while step < args.total_steps:
        # Refill dataloader when exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if is_main:
                print(f"[train] Epoch {epoch} complete, restarting dataloader")
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward — use_cache=False for training (no KV cache needed)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss / accum_steps  # scale for accumulation

        # Backward (accumulate gradients)
        loss.backward()
        micro_step += 1

        # Only step optimizer every accum_steps micro-batches
        if micro_step % accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            step_time = time.time() - t0

            # Throughput
            n_tokens = input_ids.numel() * accum_steps
            throughput.update(n_tokens)

            # Unscale loss for logging
            unscaled_loss = loss.item() * accum_steps

            # ── Logging (main rank only) ───────────────────────
            if is_main and (step % args.log_every == 0 or step == 1):
                lr_now = scheduler.get_last_lr()[0]
                tps = throughput.tokens_per_sec()
                gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                log_train_metrics(step, unscaled_loss, lr_now, gn, tps, step_time)

                pct = step / args.total_steps * 100
                eta_steps = args.total_steps - step
                eta_sec = eta_steps * step_time
                print(
                    f"  step {step:>6d}/{args.total_steps} ({pct:5.1f}%) | "
                    f"loss={unscaled_loss:.4f} | lr={lr_now:.2e} | "
                    f"tps={tps:.0f} | time={step_time:.3f}s | "
                    f"grad={gn:.4f} | "
                    f"ETA={format_time(eta_sec)}"
                )

            # ── Validation ─────────────────────────────────────
            if is_main and step % args.eval_every == 0:
                val_loss, val_ppl = evaluate(model, val_ds, batch_size=args.batch_size, device=device)
                log_val_metrics(step, val_loss, val_ppl)
                print(f"  [VAL] step={step} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")

                if val_loss < 2.8:
                    print(f"  *** TARGET REACHED: val_loss={val_loss:.4f} < 2.8 at step {step} ***")

            # ── Sample Generation ──────────────────────────────
            if is_main and step % args.sample_every == 0:
                samples = generate_samples(model, device=device)
                log_samples(step, samples)
                print(f"  [SAMPLES] step={step}:")
                for s in samples[:2]:
                    gen_preview = s["generated"][:150].replace("\n", " ")
                    print(f"    '{s['prompt']}' -> '{gen_preview}...'")

            # ── Checkpointing ──────────────────────────────────
            if is_main and step % (args.eval_every * 5) == 0:
                ckpt_path = os.path.join(REPO_ROOT, "bench", "checkpoints", f"step_{step}.pt")
                save_checkpoint(model, optimizer, scheduler, step, ckpt_path)

    # ── Final Summary ──────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"  Total steps: {step}")
        print(f"  Epochs: {epoch + 1}")

        # Final eval
        val_loss, val_ppl = evaluate(model, val_ds, batch_size=args.batch_size, device=device)
        log_val_metrics(step, val_loss, val_ppl)
        print(f"  Final val loss: {val_loss:.4f}")
        print(f"  Final val perplexity: {val_ppl:.2f}")
        print(f"  Target (<2.8): {'HIT' if val_loss < 2.8 else 'MISSED'}")
        print("=" * 60)

        # Final samples
        samples = generate_samples(model, device=device)
        log_samples(step, samples)
        print("\nFinal samples:")
        for s in samples:
            print(f"\n  Prompt: {s['prompt']}")
            print(f"  Generated: {s['generated'][:300]}")

        # Save final checkpoint
        ckpt_path = os.path.join(REPO_ROOT, "bench", "checkpoints", "final.pt")
        save_checkpoint(model, optimizer, scheduler, step, ckpt_path)

        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
