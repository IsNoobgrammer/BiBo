"""
BiBo Benchmark — Evaluation & Router Diagnostics

Validation loss (multi-batch, running avg) + comprehensive router analysis
to verify BiBo's MoE internals are working correctly.

Logs:
- Per-batch running val loss + final average
- Router expert preference (tokens per expert, load imbalance)
- Routing confidence (top-k weight stats, entropy)
- Expert diversity (how many experts actually get used)
- Bias buffer state
- Shared expert contribution magnitude
- Per-activation-type load (SiLU vs ReLU² vs Tanh vs Identity vs Zero)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import create_dataloader


# ─────────────────────────────────────────────────────────────
# Tokenizer (lazy load)
# ─────────────────────────────────────────────────────────────

_TOKENIZER = None

def get_tokenizer(name="fhai50032/QTK-81K"):
    global _TOKENIZER
    if _TOKENIZER is None:
        print(f"[eval] Loading tokenizer: {name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(name, use_fast=True)
        print(f"[eval] Tokenizer vocab size: {_TOKENIZER.vocab_size}")
    return _TOKENIZER


# ─────────────────────────────────────────────────────────────
# Expert Layout Names (for readable logging)
# ─────────────────────────────────────────────────────────────

def get_expert_names(config):
    """Return human-readable names for each expert index."""
    names = []
    for g in range(config.polyglu_expert_multiplier):
        names.append(f"SiLU_GLU_{g}")
        names.append(f"ReLU2_GLU_{g}")
        names.append(f"Tanh_GLU_{g}")
    for p in range(config.special_expert_pairs):
        names.append(f"Identity_{p}")
    for p in range(config.special_expert_pairs):
        names.append(f"Zero_{p}")
    return names


def get_activation_groups(config):
    """Map expert indices to activation type groups."""
    groups = {}
    idx = 0
    for g in range(config.polyglu_expert_multiplier):
        groups[idx] = "SiLU_GLU"; idx += 1
        groups[idx] = "ReLU2_GLU"; idx += 1
        groups[idx] = "Tanh_GLU"; idx += 1
    for p in range(config.special_expert_pairs):
        groups[idx] = "Identity"; idx += 1
    for p in range(config.special_expert_pairs):
        groups[idx] = "Zero"; idx += 1
    return groups


# ─────────────────────────────────────────────────────────────
# Router Diagnostics Hook
# ─────────────────────────────────────────────────────────────

class RouterDiagnostics:
    """
    Collects routing statistics across all MoE layers during eval.
    
    Hooks into BiBoMoERouter.forward to capture:
    - top_k_indices: which experts were selected
    - top_k_weights: how much weight each selected expert got
    - raw router logits/scores (via a temporary hook)
    """
    def __init__(self, model, config):
        self.config = config
        self.num_routed = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.expert_names = get_expert_names(config)
        self.act_groups = get_activation_groups(config)
        self.hooks = []
        self.reset()
        self._install_hooks(model)

    def reset(self):
        """Clear all accumulated stats."""
        self.layer_data = defaultdict(lambda: {
            "tokens_per_expert": torch.zeros(self.num_routed, dtype=torch.long),
            "weight_sum_per_expert": torch.zeros(self.num_routed, dtype=torch.float64),
            "top1_weights": [],
            "topk_weights_flat": [],
            "total_tokens": 0,
        })
        self.router_logits_stats = defaultdict(list)  # layer_idx -> list of (mean, std, min, max)
        self.bias_snapshots = {}  # layer_idx -> bias tensor

    def _install_hooks(self, model):
        """Install forward hooks on all MoE layers."""
        from src.modeling.ffn.moe import BiBoMoELayer
        # Handle compiled models
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        layers = base_model.model.layers

        for layer_idx, layer in enumerate(layers):
            if isinstance(layer.mlp, BiBoMoELayer):
                moe = layer.mlp
                # Snapshot bias
                self.bias_snapshots[layer_idx] = moe.gate.bias.detach().cpu().clone()
                # Hook on the gate (router)
                hook = moe.gate.register_forward_hook(
                    self._make_router_hook(layer_idx)
                )
                self.hooks.append(hook)

    def _make_router_hook(self, layer_idx):
        """Create a forward hook closure for a specific layer."""
        def hook_fn(module, input, output):
            top_k_indices, top_k_weights = output  # (B, S, K), (B, S, K)
            B, S, K = top_k_indices.shape
            num_tokens = B * S

            data = self.layer_data[layer_idx]
            data["total_tokens"] += num_tokens

            # Flatten to (B*S, K)
            flat_idx = top_k_indices.reshape(-1, K)
            flat_w = top_k_weights.float().reshape(-1, K)

            # Tokens per expert
            for k in range(K):
                counts = torch.bincount(flat_idx[:, k], minlength=self.num_routed)
                data["tokens_per_expert"] += counts.cpu()

            # Weight stats
            data["weight_sum_per_expert"] += torch.zeros(self.num_routed, dtype=torch.float64).scatter_add_(
                0, flat_idx.reshape(-1).cpu(), flat_w.reshape(-1).double().cpu()
            )
            data["top1_weights"].append(flat_w[:, 0].cpu())
            data["topk_weights_flat"].append(flat_w.cpu())

            # Router logit stats (from the internal state — approximate via scores)
            # We can get the raw logits by re-running the projection, but that's expensive.
            # Instead, capture weight distribution stats as a proxy.
            weight_entropy = -(flat_w * (flat_w + 1e-10).log()).sum(-1).mean().item()
            self.router_logits_stats[layer_idx].append({
                "weight_entropy": weight_entropy,
                "top1_mean": flat_w[:, 0].mean().item(),
                "top1_std": flat_w[:, 0].std().item(),
                "topk_mean": flat_w.mean().item(),
                "topk_std": flat_w.std().item(),
                "topk_max": flat_w.max().item(),
                "topk_min": flat_w.min().item(),
            })
        return hook_fn

    def remove_hooks(self):
        """Remove all installed hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_summary(self):
        """Compute aggregate statistics from collected data."""
        summary = {}

        for layer_idx, data in sorted(self.layer_data.items()):
            tpe = data["tokens_per_expert"].float()
            total = data["total_tokens"]
            wsum = data["weight_sum_per_expert"]

            # Load balance metrics
            ideal_load = total * self.top_k / self.num_routed  # perfect balance
            load_imbalance = (tpe.max() - tpe.min()) / (ideal_load + 1e-6)
            load_cv = tpe.std() / (tpe.mean() + 1e-6)  # coefficient of variation

            # Expert utilization: how many experts got > 1% of tokens
            threshold = total * self.top_k * 0.01
            active_experts = (tpe > threshold).sum().item()

            # Average weight per expert (when selected)
            avg_weight = wsum / (tpe.double() + 1e-6)

            # Top-1 confidence
            all_top1 = torch.cat(data["top1_weights"])
            all_topk = torch.cat(data["topk_weights_flat"])

            # Per-activation-group load
            group_load = defaultdict(float)
            for eidx in range(self.num_routed):
                group_load[self.act_groups[eidx]] += tpe[eidx].item()
            total_load = sum(group_load.values())
            group_pct = {k: v / (total_load + 1e-6) * 100 for k, v in group_load.items()}

            # Router logit stats (averaged across batches)
            logit_stats = self.router_logits_stats[layer_idx]
            avg_entropy = sum(s["weight_entropy"] for s in logit_stats) / len(logit_stats)
            avg_top1_conf = sum(s["top1_mean"] for s in logit_stats) / len(logit_stats)

            # Bias state
            bias = self.bias_snapshots.get(layer_idx, torch.zeros(self.num_routed))

            layer_summary = {
                "tokens_per_expert": tpe.tolist(),
                "load_imbalance_ratio": load_imbalance.item(),
                "load_cv": load_cv.item(),
                "active_experts": active_experts,
                "total_experts": self.num_routed,
                "avg_weight_per_expert": avg_weight.tolist(),
                "top1_confidence_mean": all_top1.mean().item(),
                "top1_confidence_std": all_top1.std().item(),
                "topk_weight_mean": all_topk.mean().item(),
                "topk_weight_std": all_topk.std().item(),
                "weight_entropy_mean": avg_entropy,
                "avg_top1_conf": avg_top1_conf,
                "group_load_pct": dict(group_pct),
                "bias_mean": bias.mean().item(),
                "bias_std": bias.std().item(),
                "bias_min": bias.min().item(),
                "bias_max": bias.max().item(),
                "bias_values": bias.tolist(),
            }
            summary[layer_idx] = layer_summary

        return summary


# ─────────────────────────────────────────────────────────────
# Validation Loss (multi-batch with running loss)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_ds, batch_size=32, device="cuda", max_batches=None,
             min_batches=10, log_running=True, run_router_diagnostics=True):
    """
    Compute validation loss, perplexity, and router diagnostics.
    
    Args:
        model: BiBoForCausalLM
        val_ds: validation dataset
        batch_size: batch size for eval
        device: device
        max_batches: cap on number of batches (None = all)
        min_batches: minimum batches to run (default 10)
        log_running: print per-batch running loss
        run_router_diagnostics: collect router stats
    
    Returns:
        avg_loss, perplexity, router_summary (or None if diagnostics disabled)
    """
    model.eval()
    loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    # Get config from model
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    config = base_model.config

    # Install router diagnostics
    diag = None
    if run_router_diagnostics:
        diag = RouterDiagnostics(model, config)

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    running_losses = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

        # Count valid tokens (not -100)
        valid_tokens = (labels != -100).sum().item()
        batch_loss = loss.item()
        total_loss += batch_loss * valid_tokens
        total_tokens += valid_tokens
        n_batches += 1
        running_losses.append(batch_loss)

        if log_running:
            running_avg = total_loss / total_tokens
            print(f"    [val batch {n_batches:>3d}] "
                  f"batch_loss={batch_loss:.4f} | "
                  f"running_avg={running_avg:.4f}")

        # Respect max_batches but ensure min_batches
        if max_batches is not None and n_batches >= max(max_batches, min_batches):
            break
        # If no max_batches, still enforce min_batches (continue until dataset exhausted)

    # Cleanup hooks
    router_summary = None
    if diag is not None:
        router_summary = diag.compute_summary()
        diag.remove_hooks()

    model.train()

    if total_tokens == 0:
        return 0.0, 0.0, router_summary

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity, router_summary


# ─────────────────────────────────────────────────────────────
# Pretty-print Router Diagnostics
# ─────────────────────────────────────────────────────────────

def print_router_diagnostics(router_summary, config):
    """Print comprehensive router diagnostics to console."""
    if router_summary is None:
        return

    expert_names = get_expert_names(config)
    TAG = "[bibo]"

    print(f"\n{TAG} {'='*70}")
    print(f"{TAG} ROUTER DIAGNOSTICS")
    print(f"{TAG} {'='*70}")

    for layer_idx, stats in sorted(router_summary.items()):
        print(f"\n{TAG} ── Layer {layer_idx} ──────────────────────────────────────")

        # Load balance
        print(f"{TAG}   Load Balance:")
        print(f"{TAG}     Imbalance ratio: {stats['load_imbalance_ratio']:.3f} "
              f"(0=perfect, >1=severe)")
        print(f"{TAG}     CV (coeff of variation): {stats['load_cv']:.3f}")
        print(f"{TAG}     Active experts (>1% load): "
              f"{stats['active_experts']}/{stats['total_experts']}")

        # Per-expert token counts
        tpe = stats["tokens_per_expert"]
        total_assignments = sum(tpe)
        print(f"{TAG}   Expert Load (tokens assigned):")
        for eidx, (name, count) in enumerate(zip(expert_names, tpe)):
            pct = count / (total_assignments + 1e-6) * 100
            bar = "█" * int(pct * 2)  # visual bar
            print(f"{TAG}     [{eidx}] {name:<14s}: {count:>8.0f} "
                  f"({pct:5.1f}%) {bar}")

        # Per-activation-group load
        print(f"{TAG}   Activation Group Load:")
        for group, pct in sorted(stats["group_load_pct"].items()):
            print(f"{TAG}     {group:<12s}: {pct:5.1f}%")

        # Confidence / weight stats
        print(f"{TAG}   Routing Confidence:")
        print(f"{TAG}     Top-1 weight: {stats['top1_confidence_mean']:.4f} "
              f"± {stats['top1_confidence_std']:.4f}")
        print(f"{TAG}     Top-K mean weight: {stats['topk_weight_mean']:.4f} "
              f"± {stats['topk_weight_std']:.4f}")
        print(f"{TAG}     Weight entropy: {stats['weight_entropy_mean']:.4f} "
              f"(higher = more uniform)")

        # Bias state
        print(f"{TAG}   Router Bias:")
        print(f"{TAG}     mean={stats['bias_mean']:.4f}, "
              f"std={stats['bias_std']:.4f}, "
              f"range=[{stats['bias_min']:.4f}, {stats['bias_max']:.4f}]")
        bias_vals = stats["bias_values"]
        for eidx, (name, bv) in enumerate(zip(expert_names, bias_vals)):
            direction = "↑" if bv > 0.01 else ("↓" if bv < -0.01 else "·")
            print(f"{TAG}     [{eidx}] {name:<14s}: {bv:+.4f} {direction}")

        # Health check flags
        print(f"{TAG}   Health Checks:")
        # Check: is Zero expert getting too much load?
        zero_pct = stats["group_load_pct"].get("Zero", 0)
        if zero_pct > 15:
            print(f"{TAG}     ⚠️  Zero expert has {zero_pct:.1f}% load "
                  f"(wasted compute, router may be confused)")
        elif zero_pct < 1:
            print(f"{TAG}     ✓ Zero expert properly avoided ({zero_pct:.1f}%)")
        else:
            print(f"{TAG}     · Zero expert at {zero_pct:.1f}% "
                  f"(acceptable dump bucket)")

        # Check: is top-1 dominating? (bad for top-k > 1)
        if stats["top1_confidence_mean"] > 0.8:
            print(f"{TAG}     ⚠️  Top-1 confidence {stats['top1_confidence_mean']:.3f} "
                  f"> 0.8 — other experts underutilized!")
        else:
            print(f"{TAG}     ✓ Top-1 confidence {stats['top1_confidence_mean']:.3f} "
                  f"— good expert utilization")

        # Check: are all experts active?
        if stats["active_experts"] < stats["total_experts"]:
            print(f"{TAG}     ⚠️  Only {stats['active_experts']}/"
                  f"{stats['total_experts']} experts active — "
                  f"possible expert collapse")
        else:
            print(f"{TAG}     ✓ All {stats['total_experts']} experts active")

    print(f"\n{TAG} {'='*70}")


# ─────────────────────────────────────────────────────────────
# WandB Router Logging
# ─────────────────────────────────────────────────────────────

def log_router_diagnostics_wandb(step, router_summary, config):
    """Log router diagnostics to WandB for visualization."""
    try:
        import wandb
        if wandb.run is None:
            return
    except ImportError:
        return

    expert_names = get_expert_names(config)

    for layer_idx, stats in sorted(router_summary.items()):
        prefix = f"router/layer_{layer_idx}"

        # Scalar metrics
        wandb.log({
            f"{prefix}/load_imbalance": stats["load_imbalance_ratio"],
            f"{prefix}/load_cv": stats["load_cv"],
            f"{prefix}/active_experts": stats["active_experts"],
            f"{prefix}/top1_confidence_mean": stats["top1_confidence_mean"],
            f"{prefix}/top1_confidence_std": stats["top1_confidence_std"],
            f"{prefix}/topk_weight_mean": stats["topk_weight_mean"],
            f"{prefix}/weight_entropy": stats["weight_entropy_mean"],
            f"{prefix}/bias_std": stats["bias_std"],
            f"{prefix}/bias_range": stats["bias_max"] - stats["bias_min"],
            "val/step": step,
        })

        # Per-activation-group load
        for group, pct in stats["group_load_pct"].items():
            wandb.log({
                f"{prefix}/group_{group}_pct": pct,
                "val/step": step,
            })

        # Per-expert load as bar chart
        tpe = stats["tokens_per_expert"]
        total = sum(tpe)
        expert_data = [[name, count / (total + 1e-6) * 100]
                       for name, count in zip(expert_names, tpe)]
        table = wandb.Table(data=expert_data, columns=["expert", "load_pct"])
        wandb.log({
            f"{prefix}/expert_load": wandb.plot.bar(
                table, "expert", "load_pct",
                title=f"Layer {layer_idx} Expert Load %"
            ),
            "val/step": step,
        })

        # Bias values as bar chart
        bias_data = [[name, bv] for name, bv in zip(expert_names, stats["bias_values"])]
        bias_table = wandb.Table(data=bias_data, columns=["expert", "bias"])
        wandb.log({
            f"{prefix}/bias_values": wandb.plot.bar(
                bias_table, "expert", "bias",
                title=f"Layer {layer_idx} Router Bias"
            ),
            "val/step": step,
        })


# ─────────────────────────────────────────────────────────────
# Sample Generation
# ─────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    "The meaning of life is",
    "Once upon a time in a distant land",
    "In the year 2026, artificial intelligence",
    "The key to solving complex problems",
    "Scientists recently discovered that",
]


@torch.no_grad()
def generate_samples(
    model,
    prompts=None,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    device="cuda",
):
    """Generate text samples from prompts using the model."""
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    tokenizer = get_tokenizer()
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Manual generation loop
        generated_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_idx)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": text})

    model.train()
    return results


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    test = tokenizer.encode("Hello world, this is a test.")
    print(f"Encode test: {test}")
    print(f"Decode test: {tokenizer.decode(test)}")
