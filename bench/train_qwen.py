"""
Qwen3MoE Benchmark — Comparable to BiBo bench/train.py

Same data pipeline, tokenizer, optimizer, eval, and training loop.
Only the model and config differ. This enables direct comparison.

Usage:
    python bench/train_qwen.py
    python bench/train_qwen.py --batch_size 4 --total_steps 500
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

from data import load_benchmark_data, create_dataloader
from optim import create_optimizer, create_scheduler, HAS_MUON, HAS_BNB
from eval import evaluate, generate_samples, get_tokenizer
from utils import (
    init_wandb, log_train_metrics, log_val_metrics, log_samples,
    save_checkpoint, load_checkpoint, ThroughputMeter, format_time,
)

# ─────────────────────────────────────────────────────────────
# Qwen3MoE imports
# ─────────────────────────────────────────────────────────────
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM


# ─────────────────────────────────────────────────────────────
# Qwen3MoE Config — matched to BiBo baseline
# ─────────────────────────────────────────────────────────────

QWEN_72M_BASELINE = Qwen3MoeConfig(
    vocab_size=81000,               # Will be resized to tokenizer len
    hidden_size=320,                # Same as BiBo
    intermediate_size=1024,         # Same as BiBo (dense MLP)
    num_hidden_layers=10,           # Same as BiBo
    num_attention_heads=5,          # Same as BiBo
    num_key_value_heads=1,          # Same as BiBo (GQA 5:1)
    max_position_embeddings=2048,   # Same as BiBo
    hidden_act="silu",              # Qwen uses SiLU everywhere
    # MoE — 8 homogeneous SwiGLU experts, top-2
    num_experts=8,                  # 8 routed experts (same count as BiBo)
    num_experts_per_tok=2,          # Top-2 routing (same as BiBo)
    moe_intermediate_size=768,      # Per-expert FFN (same as BiBo)
    decoder_sparse_step=1,          # MoE every layer
    mlp_only_layers=[0, 9],         # First+last dense (same as BiBo)
    norm_topk_prob=False,
    router_aux_loss_coef=0.001,
    # Other
    tie_word_embeddings=True,
    attention_dropout=0.0,
    attention_bias=False,
    use_sliding_window=False,
    use_cache=True,
)


def count_params_qwen(config):
    """Count total and active params for a Qwen3MoeConfig."""
    model = Qwen3MoeForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.model.embed_tokens.parameters())

    attn_total = 0
    dense_total = 0
    moe_routed_total = 0
    moe_router_total = 0
    num_moe = 0
    num_dense = 0

    for layer in model.model.layers:
        # Attention + norms
        attn_total += sum(p.numel() for p in layer.self_attn.parameters())
        attn_total += sum(p.numel() for p in layer.input_layernorm.parameters())
        attn_total += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            # MoE layer
            num_moe += 1
            moe = layer.mlp
            moe_router_total += sum(p.numel() for p in moe.gate.parameters())
            moe_routed_total += sum(p.numel() for p in moe.experts.parameters())
        else:
            # Dense MLP layer
            num_dense += 1
            dense_total += sum(p.numel() for p in layer.mlp.parameters())

    # Active params per token
    routed_per_layer = moe_routed_total / max(num_moe, 1)
    active_routed = routed_per_layer * (config.num_experts_per_tok / config.num_experts) * num_moe
    active_total = embed + attn_total + dense_total + active_routed + moe_router_total

    del model
    return {
        "total": total,
        "total_m": total / 1e6,
        "active": int(active_total),
        "active_m": active_total / 1e6,
        "ratio": total / max(active_total, 1),
        "embed": embed,
        "attn": attn_total,
        "dense": dense_total,
        "moe_routed": moe_routed_total,
        "moe_router": moe_router_total,
        "num_moe_layers": num_moe,
        "num_dense_layers": num_dense,
        "routed_per_layer": int(routed_per_layer),
    }


def build_model(config=None):
    if config is None:
        config = QWEN_72M_BASELINE
    model = Qwen3MoeForCausalLM(config)
    return model, config


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3MoE Benchmark Training")

    # Training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
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
    p.add_argument("--log_every", type=int, default=10)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default="bench/checkpoints_qwen")
    p.add_argument("--resume", type=str, default=None)

    # WandB
    p.add_argument("--wandb_project", type=str, default="bibo-bench")
    p.add_argument("--wandb_name", type=str, default="qwen")
    p.add_argument("--wandb_notes", type=str, default="Qwen3MoE ~72M baseline, 2×T4, QTK-81K")

    # Mode
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing")

    return p.parse_args()


def setup_distributed():
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


def compile_model(model):
    return torch.compile(model, mode="default", fullgraph=False)


def train(args):
    TAG = "[qwen]"
    is_distributed, rank, world_size, device = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if is_main:
        print(f"{TAG} " + "=" * 60)
        print(f"{TAG} Qwen3MoE Benchmark Training")
        print(f"{TAG} " + "=" * 60)
        print(f"{TAG}   Device: {device} (world_size={world_size})")
        print(f"{TAG}   Muon: {HAS_MUON}, bitsandbytes: {HAS_BNB}")
        print(f"{TAG}   Batch size: {args.batch_size}, grad_accum: {args.grad_accum}")
        print(f"{TAG}   Total steps: {args.total_steps}")
        print(f"{TAG}   Seq len: {args.seq_len}")
        print(f"{TAG}   LR: {args.lr}")
        print(f"{TAG} " + "=" * 60)

    # ── Data ───────────────────────────────────────────────────
    if is_main:
        print(f"{TAG} [train] Loading dataset...")
    train_ds, val_ds = load_benchmark_data(seq_len=args.seq_len, val_split=args.val_split)

    # ── Tokenizer ──────────────────────────────────────────────
    if is_main:
        print(f"{TAG} [train] Loading tokenizer...")
    tokenizer = get_tokenizer()
    actual_vocab_size = len(tokenizer)

    _peek_loader = create_dataloader(train_ds, batch_size=1, shuffle=False)
    _peek_batch = next(iter(_peek_loader))
    _max_id = _peek_batch["input_ids"].max().item()
    del _peek_loader, _peek_batch
    _safe_vocab = max(actual_vocab_size, _max_id + 1)
    if is_main:
        print(f"{TAG} [train] Tokenizer: vocab_size={tokenizer.vocab_size}, len={actual_vocab_size}")
        print(f"{TAG} [train] Dataset max token ID: {_max_id}, safe vocab: {_safe_vocab}")

    # ── Model ──────────────────────────────────────────────────
    if is_main:
        print(f"{TAG} [train] Building Qwen3MoE model...")
    model, config = build_model()

    if _safe_vocab != config.vocab_size:
        if is_main:
            print(f"{TAG} [train] Resizing embeddings: {config.vocab_size} -> {_safe_vocab}")
        model.resize_token_embeddings(_safe_vocab)
        config.vocab_size = _safe_vocab

    stats = count_params_qwen(config)
    if is_main:
        print(f"{TAG} [train] Model params: {stats['total_m']:.2f}M total, {stats['active_m']:.2f}M active, ratio={stats['ratio']:.2f}x")
        print(f"{TAG} [train] Experts: {config.num_experts} routed, top-{config.num_experts_per_tok}")
        print(f"{TAG} [train] Layers: {config.num_hidden_layers} ({stats['num_moe_layers']} MoE + {stats['num_dense_layers']} dense)")

    # Gradient checkpointing — DISABLED by default for MFU
    if getattr(args, 'grad_checkpoint', False):
        if is_main:
            print(f"{TAG} [train] Enabling gradient checkpointing (use_reentrant=True for MoE)...")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
    else:
        if is_main:
            print(f"{TAG} [train] Gradient checkpointing DISABLED for max throughput")
    config.use_cache = False
    model = model.to(device)

    # ── FSDP2 ──────────────────────────────────────────────────
    if is_distributed:
        if is_main:
            print(f"{TAG} [train] Wrapping with FSDP2...")
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

    # ── torch.compile ──────────────────────────────────────────
    if not args.no_compile:
        if is_main:
            print(f"{TAG} [train] Compiling model...")
        try:
            model = compile_model(model)
            if is_main:
                print(f"{TAG} [train] torch.compile OK")
        except Exception as e:
            if is_main:
                print(f"{TAG} [train] torch.compile failed: {e}")

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = create_optimizer(model, lr=args.lr, muon_lr=args.muon_lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.total_steps)

    # ── DataLoader ─────────────────────────────────────────────
    train_loader = create_dataloader(train_ds, batch_size=args.batch_size)

    # ── WandB ──────────────────────────────────────────────────
    if not args.no_wandb and is_main:
        wandb_config = {
            "model": "Qwen3MoE-baseline",
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
            "world_size": world_size,
            "has_muon": HAS_MUON,
            "has_bnb": HAS_BNB,
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
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
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
        }
        init_wandb(wandb_config, project=args.wandb_project, name=args.wandb_name, notes=args.wandb_notes)

    # ── Resume ─────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume)

    # ── Eval Only ──────────────────────────────────────────────
    if args.eval_only:
        if is_main:
            val_loss, val_ppl = evaluate(model, val_ds, batch_size=args.batch_size, device=device, max_batches=100)
            print(f"{TAG} [eval] Val loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        return

    # ── Training Loop ──────────────────────────────────────────
    if is_main:
        print(f"{TAG} \n[train] Starting training from step {start_step}...")
        print(f"{TAG} [train] {len(train_ds)} train samples, {len(val_ds)} val samples")
        print(f"{TAG} [train] {len(train_loader)} batches per epoch")
        print()

    scaler = torch.amp.GradScaler()
    throughput = ThroughputMeter(warmup=3)
    model.train()
    epoch = 0
    step = start_step
    data_iter = iter(train_loader)
    accum_steps = args.grad_accum
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    while step < args.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if is_main:
                print(f"{TAG} [train] Epoch {epoch} complete, restarting dataloader")
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            step_time = time.time() - t0

            n_tokens = input_ids.numel() * accum_steps
            throughput.update(n_tokens)

            unscaled_loss = loss.item() * accum_steps

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

            if is_main and step % args.eval_every == 0:
                try:
                    val_loss, val_ppl = evaluate(model, val_ds, batch_size=max(args.batch_size, 8), device=device, max_batches=100)
                    log_val_metrics(step, val_loss, val_ppl)
                    print(f"{TAG}   [VAL] step={step} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"{TAG}   [VAL] step={step} | OOM — skipping")
                    torch.cuda.empty_cache()

            if is_main and step % args.sample_every == 0:
                try:
                    samples = generate_samples(model, device=device, prompts=["The meaning of life is"], max_new_tokens=50)
                    log_samples(step, samples)
                except torch.cuda.OutOfMemoryError:
                    print(f"{TAG}   [SAMPLES] step={step} | OOM — skipping")
                    torch.cuda.empty_cache()

    # ── Final Summary ──────────────────────────────────────────
    if is_main:
        print(f"{TAG} " + "\n" + "=" * 60)
        print(f"{TAG} Training Complete!")
        print(f"{TAG}   Total steps: {step}")
        print(f"{TAG}   Epochs: {epoch + 1}")

        try:
            val_loss, val_ppl = evaluate(model, val_ds, batch_size=max(args.batch_size, 8), device=device, max_batches=100)
            log_val_metrics(step, val_loss, val_ppl)
            print(f"{TAG}   Final val loss: {val_loss:.4f}")
            print(f"{TAG}   Final val perplexity: {val_ppl:.2f}")
            print(f"{TAG}   Target (<2.8): {'HIT' if val_loss < 2.8 else 'MISSED'}")
        except torch.cuda.OutOfMemoryError:
            print(f"{TAG}   [WARN] Final eval OOM — skipping")
            torch.cuda.empty_cache()

        print(f"{TAG} " + "=" * 60)

        ckpt_path = os.path.join(REPO_ROOT, "bench", "checkpoints_qwen", "final.pt")
        save_checkpoint(model, optimizer, scheduler, step, ckpt_path)

        if not args.no_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
