"""
BiBo Benchmark — Evaluation & Sample Generation

Validation loss, HellaSwag, ARC-Challenge, and text generation.
Tokenizers used ONLY for inference/decode, NOT during training.
"""

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import sys
import os
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
        print(f"  Loading tokenizer: {name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(
            name, use_fast=True, cache_dir=os.environ.get("PERSISTENT_DIR") or None)
    return _TOKENIZER


# ─────────────────────────────────────────────────────────────
# Validation Loss
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_ds, batch_size=32, device="cuda", max_batches=None,
             tokenizer=None):
    """Compute validation loss, perplexity, and bits-per-byte.

    bits-per-byte (bpb) is tokenizer-independent: total NLL in bits divided by
    the UTF-8 byte length of the decoded text. Needs the tokenizer to map the
    predicted token ids back to bytes; if None, bpb is returned as None.
    NLL is accumulated from the model's pure-CE loss (works even when Liger
    returns logits=None).
    """
    model.eval()
    loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0      # sum of NLL in nats
    total_tokens = 0
    total_bytes = 0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            # output_router_logits omitted -> pure LM cross-entropy for BiBo and Qwen alike
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

        valid_mask = labels != -100
        valid_tokens = valid_mask.sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

        if tokenizer is not None:
            # bytes of the predicted tokens, decoded jointly per sequence
            ids_cpu = input_ids.cpu()
            mask_cpu = valid_mask.cpu()
            for row, m in zip(ids_cpu, mask_cpu):
                pred_ids = row[m].tolist()
                if pred_ids:
                    text = tokenizer.decode(pred_ids, skip_special_tokens=False)
                    total_bytes += len(text.encode("utf-8"))

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    model.train()

    if total_tokens == 0:
        return 0.0, 0.0, None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    bpb = (total_loss / math.log(2)) / total_bytes if total_bytes > 0 else None
    return avg_loss, perplexity, bpb


# ─────────────────────────────────────────────────────────────
# Length-extrapolation bpb (#3) — the real long-range test on natural data
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_length_extrapolation(model, lengths, val_split, seed,
                                  batch_size, device, max_batches=40, tokenizer=None):
    """Val bpb at each seq length (train length + extrapolation). The data is packed-2K so 2048 is
    real text. BiBo (SSMax+NTK+partial-NoPE) should hold bpb as length grows where plain RoPE
    degrades. Same seed → identical val sequences, just truncated to each L.
    Returns {'per_len': {L: {loss,ppl,bpb}}, 'ratio': bpb(maxL)/bpb(trainL)}."""
    from data import load_benchmark_data
    per_len = {}
    base_len = min(lengths)
    for L in sorted(lengths):
        _, val_ds = load_benchmark_data(seq_len=L, val_split=val_split, seed=seed)
        bs = max(1, batch_size * base_len // L)          # ~constant tokens/batch as L grows
        loss, ppl, bpb = evaluate(model, val_ds, batch_size=bs, device=device,
                                  max_batches=max_batches, tokenizer=tokenizer)
        per_len[L] = {"loss": loss, "ppl": ppl, "bpb": bpb}
    lo = per_len[base_len].get("bpb"); hi = per_len[max(lengths)].get("bpb")
    return {"per_len": per_len, "ratio": (hi / lo) if (lo and hi) else 0.0}


# ─────────────────────────────────────────────────────────────
# Multiple-choice scoring (shared by HellaSwag + ARC)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_multiple_choice(model, tokenizer, device, dataset, ctx_key, batch_size=8, pad_id=0):
    """Score a multiple-choice dataset by completion log-likelihood — BATCHED on GPU.

    All (context+completion) sequences are flattened and run `batch_size` at a time in a
    single padded forward (right-pad; causal attention means completion positions never
    attend to the trailing pad, so no attention mask is needed and pad_id is irrelevant to
    the scored logprobs). This keeps the GPU busy instead of one batch=1 forward per choice.

    Reports both lm-eval-harness metrics:
      - acc      = argmax over choices of SUM(token log-probs)
      - acc_norm = argmax over choices of SUM(token log-probs) / len(choice_text_chars)
                   (length-normalized — the canonical HellaSwag/ARC headline).
    Each example: {ctx_key: str, "completions": [str], "gold_idx": int}.
    """
    model.eval()
    # 1. flatten every (example, choice) into one sequence to score
    seqs = []
    n_choices = []
    for ei, ex in enumerate(dataset):
        ctx_ids = tokenizer.encode(ex[ctx_key])
        n_choices.append(len(ex["completions"]))
        for ci, comp in enumerate(ex["completions"]):
            comp_ids = tokenizer.encode(comp) or [pad_id]
            seqs.append({"ei": ei, "ci": ci, "ctx_len": len(ctx_ids),
                         "comp_ids": comp_ids, "full": ctx_ids + comp_ids,
                         "char_len": max(len(comp), 1)})
    lp = [[float("-inf")] * c for c in n_choices]
    cl = [[1.0] * c for c in n_choices]

    # 2. batched forward, batch_size sequences at a time
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        maxlen = max(len(s["full"]) for s in chunk)
        inp = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        for j, s in enumerate(chunk):
            inp[j, :len(s["full"])] = torch.tensor(s["full"], dtype=torch.long)
        inp = inp.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=inp).logits                 # (b, maxlen, V)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for j, s in enumerate(chunk):
            c0, clen = s["ctx_len"], len(s["comp_ids"])
            pos = torch.arange(c0 - 1, c0 + clen - 1, device=device)   # predict comp[t] from pos c0+t-1
            tok = torch.tensor(s["comp_ids"], device=device)
            tlp = logprobs[j, pos, :].gather(1, tok.unsqueeze(1)).squeeze(1)
            lp[s["ei"]][s["ci"]] = tlp.sum().item()
            cl[s["ei"]][s["ci"]] = s["char_len"]

    # 3. acc + acc_norm + CONTINUOUS metrics per example.
    # margin/auc use length-normalized logprobs (lp/charlen) — they lift off 0 / 0.5 long before
    # accuracy leaves chance, so a weak model produces an ordered, moving signal where acc is noise.
    correct = correct_norm = 0
    margins, aucs = [], []
    for ei, ex in enumerate(dataset):
        gold = ex["gold_idx"]
        nc = n_choices[ei]
        correct += int(max(range(nc), key=lambda k: lp[ei][k]) == gold)
        correct_norm += int(max(range(nc), key=lambda k: lp[ei][k] / cl[ei][k]) == gold)
        norm = [lp[ei][k] / cl[ei][k] for k in range(nc)]
        others = [norm[k] for k in range(nc) if k != gold]
        if others:
            margins.append(norm[gold] - sum(others) / len(others))   # >0 ⇒ prefers gold
            aucs.append(sum(norm[gold] > o for o in others) / len(others))  # P(gold > random distractor); 0.5=chance
    total = len(dataset)
    _mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "acc_norm": correct_norm / max(total, 1),
        "margin": _mean(margins),     # length-norm Δ-logprob (gold − mean distractor)
        "auc": _mean(aucs),           # ranking AUC, 0.5 = chance
        "correct": correct,
        "correct_norm": correct_norm,
        "total": total,
    }


# ─────────────────────────────────────────────────────────────
# HellaSwag Benchmark
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, device, max_examples=None, batch_size=8):
    """
    Evaluate on HellaSwag.
    Score completions by sum of log-probs of completion tokens.
    Returns {accuracy, correct, total}.
    """
    from data import load_hellaswag
    dataset = load_hellaswag(max_examples)
    res = _score_multiple_choice(
        model, tokenizer, device, dataset,
        ctx_key="ctx", batch_size=batch_size)
    return res


# ─────────────────────────────────────────────────────────────
# Run All Evaluations
# (ARC-Challenge removed 2026-06-26 — pure noise at 119M/8M-tokens; HellaSwag kept as a
#  weak sanity signal, val_loss/bpb is the real discriminator.)
# ─────────────────────────────────────────────────────────────

def run_all_evals(model, tokenizer, val_ds, device, benchmarks, batch_size=8,
                  max_batches=100, max_examples=None):
    """
    Run val_loss + all benchmark evaluations.
    Returns combined results dict.
    """
    results = {}

    val_loss, val_ppl, val_bpb = evaluate(model, val_ds, batch_size=batch_size,
                                           device=device, max_batches=max_batches,
                                           tokenizer=tokenizer)
    results["val_loss"] = val_loss
    results["val_ppl"] = val_ppl
    results["val_bpb"] = val_bpb

    for bench in benchmarks:
        if bench == "hellaswag":
            results["hellaswag"] = evaluate_hellaswag(
                model, tokenizer, device, max_examples=max_examples, batch_size=batch_size)

    return results


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
def generate_samples(model, prompts=None, max_new_tokens=100, temperature=0.8,
                     top_p=0.9, device="cuda"):
    """Generate text samples from prompts using the model."""
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    tokenizer = get_tokenizer()
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated_ids = input_ids.clone()

        ent_sum, top1_sum, n_steps, first_token = 0.0, 0.0, 0, None
        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=generated_ids)
                last_logits = outputs.logits[:, -1, :]
            # Collapse diagnostic on the RAW (temp=1, un-truncated) distribution: top-1 prob ~0.95+
            # and entropy ~0 → hard distribution collapse; moderate values → plain underfit.
            p = F.softmax(last_logits.float(), dim=-1)
            top1_sum += p.max(-1).values.item()
            ent_sum += -(p * p.clamp_min(1e-12).log()).sum(-1).item()
            n_steps += 1

            logits = last_logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_idx)
            if first_token is None:
                first_token = next_token.item()
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append({
            "prompt": prompt, "generated": text,
            "top1_prob": top1_sum / max(n_steps, 1),     # mean over generated steps
            "entropy": ent_sum / max(n_steps, 1),        # nats; near 0 → collapsed
            "first_token": first_token,                  # same across prompts → collapse
        })

    model.train()
    return results
