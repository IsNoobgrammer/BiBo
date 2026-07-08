"""Paired BiBo-vs-Qwen comparison on the SAME val sequences (#1 — the sharpest diff).

Loads two checkpoints, scores per-sequence bits-per-byte on identical val data (shuffle off →
same order), and reports PAIRED stats: mean bpb each, mean Δ(B−A), % A-wins, and a paired
bootstrap 95% CI on Δ. Paired cancels per-sequence difficulty variance, so a real gap shows up
at ~10× fewer sequences than comparing two independent run-averages.

bpb (bits-per-byte) is tokenizer-independent and directly comparable across the two models.

Usage (run on T4 with the saved checkpoints):
  python bench/compare.py \
    --a bench/configs/bibo.yaml     --a_ckpt bench/checkpoints/bibo_step_500.pt \
    --b bench/configs/qwen3moe.yaml --b_ckpt bench/checkpoints/qwen3moe_step_500.pt \
    --batch 8 --max_batches 40
"""
import os, sys, argparse, math, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "disabled")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "bench"))
import yaml, torch
import torch.nn.functional as F


def load_model(cfg_path, ckpt_path, device):
    from models import build_model_from_config, resize_embeddings
    cfg = yaml.safe_load(open(cfg_path))
    model, mcfg = build_model_from_config(cfg)
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model", sd)                              # save_checkpoint wraps in {"model": ...}
        # Training resized embeddings to the tokenizer vocab (81920); match it before loading,
        # else embed_tokens/lm_head shape-mismatch (config vocab 81000 != checkpoint 81920).
        ckpt_vocab = sd.get("model.embed_tokens.weight", sd.get("lm_head.weight")).shape[0]
        resize_embeddings(model, mcfg, ckpt_vocab)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [load] {os.path.basename(ckpt_path)}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        print(f"  [load] WARNING: no checkpoint at {ckpt_path} — using random init")
    return model.to(device).eval(), cfg["model"]["type"]


@torch.no_grad()
def per_seq_bpb(model, loader, tokenizer, device, max_batches):
    """Per-sequence bits-per-byte over the (fixed-order) val loader."""
    out = []
    for nb, batch in enumerate(loader):
        if nb >= max_batches:
            break
        ids = batch["input_ids"].to(device); labels = batch["labels"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits              # (B,S,V) — labels omitted ⇒ logits returned
        B, S, V = logits.shape
        nll = F.cross_entropy(logits.float().view(-1, V), labels.view(-1),
                              ignore_index=-100, reduction="none").view(B, S)
        mask = labels != -100
        nats = (nll * mask).sum(1)                             # per-seq NLL (nats)
        ids_c, mask_c = ids.cpu(), mask.cpu()
        for bi in range(B):
            pred = ids_c[bi][mask_c[bi]].tolist()
            nbytes = len(tokenizer.decode(pred, skip_special_tokens=False).encode("utf-8")) if pred else 1
            out.append((nats[bi].item() / math.log(2)) / max(nbytes, 1))
        del logits, nll
    return out


def bootstrap_ci(deltas, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="bench/configs/bibo.yaml")
    ap.add_argument("--a_ckpt", default="bench/checkpoints/bibo_step_500.pt")
    ap.add_argument("--b", default="bench/configs/qwen3moe.yaml")
    ap.add_argument("--b_ckpt", default="bench/checkpoints/qwen3moe_step_500.pt")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=40)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from data import load_benchmark_data, create_dataloader
    from eval import get_tokenizer
    tok = get_tokenizer()
    _, val_ds = load_benchmark_data(seq_len=args.seq, val_split=args.val_split, seed=args.seed)
    # shuffle=False ⇒ identical sequence order for both models (the pairing)
    mk = lambda: create_dataloader(val_ds, batch_size=args.batch, shuffle=False)

    print(f"Model A: {args.a}")
    ma, na = load_model(args.a, args.a_ckpt, dev)
    a = per_seq_bpb(ma, mk(), tok, dev, args.max_batches)
    del ma; torch.cuda.empty_cache()
    print(f"Model B: {args.b}")
    mb, nb_ = load_model(args.b, args.b_ckpt, dev)
    b = per_seq_bpb(mb, mk(), tok, dev, args.max_batches)
    del mb; torch.cuda.empty_cache()

    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    deltas = [bb - aa for aa, bb in zip(a, b)]                 # >0 ⇒ A (BiBo) better (lower bpb)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    mean_d = sum(deltas) / n
    a_wins = sum(d > 0 for d in deltas) / n
    lo, hi = bootstrap_ci(deltas)
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant (CI crosses 0)"

    print(f"\n{'='*64}\nPAIRED bpb over {n} val sequences (lower = better)\n{'='*64}")
    print(f"  A ({na:>9}): {mean_a:.4f} bpb")
    print(f"  B ({nb_:>9}): {mean_b:.4f} bpb")
    print(f"  Δ (B−A)     : {mean_d:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]  → {sig}")
    print(f"  A wins on   : {a_wins*100:.1f}% of sequences")
    better = na if mean_d > 0 else nb_
    print(f"  → {better} has lower bpb" + (f" (Δ {abs(mean_d):.4f} bits/byte)" if lo > 0 or hi < 0 else " — but within noise"))


if __name__ == "__main__":
    main()
