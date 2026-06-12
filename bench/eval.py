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
        _TOKENIZER = AutoTokenizer.from_pretrained(name, use_fast=True)
    return _TOKENIZER


# ─────────────────────────────────────────────────────────────
# Validation Loss
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_ds, batch_size=32, device="cuda", max_batches=None):
    """Compute validation loss and perplexity."""
    model.eval()
    loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

    model.train()

    if total_tokens == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


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

    model.eval()
    correct = 0
    total = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        for example in batch:
            ctx_ids = tokenizer.encode(example["ctx"], return_tensors="pt").to(device)
            ctx_len = ctx_ids.shape[1]

            log_probs = []
            for completion in example["completions"]:
                comp_ids = tokenizer.encode(completion, return_tensors="pt").to(device)
                full_ids = torch.cat([ctx_ids, comp_ids], dim=-1)

                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=full_ids)
                    logits = outputs.logits

                comp_logits = logits[0, ctx_len - 1: ctx_len + comp_ids.shape[1] - 1, :]
                comp_logprobs = F.log_softmax(comp_logits, dim=-1)
                token_logprobs = comp_logprobs.gather(1, comp_ids[0].unsqueeze(1)).squeeze(1)
                log_probs.append(token_logprobs.sum().item())

            predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
            if predicted == example["gold_idx"]:
                correct += 1
            total += 1

    model.train()
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─────────────────────────────────────────────────────────────
# ARC-Challenge Benchmark
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_arc_challenge(model, tokenizer, device, max_examples=None, batch_size=8):
    """
    Evaluate on ARC-Challenge.
    Multiple choice by log-prob sum.
    Returns {accuracy, correct, total}.
    """
    from data import load_arc_challenge
    dataset = load_arc_challenge(max_examples)

    model.eval()
    correct = 0
    total = 0

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        for example in batch:
            question_ids = tokenizer.encode(example["question"], return_tensors="pt").to(device)
            q_len = question_ids.shape[1]

            log_probs = []
            for choice in example["completions"]:
                choice_ids = tokenizer.encode(choice, return_tensors="pt").to(device)
                full_ids = torch.cat([question_ids, choice_ids], dim=-1)

                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(input_ids=full_ids)
                    logits = outputs.logits

                choice_logits = logits[0, q_len - 1: q_len + choice_ids.shape[1] - 1, :]
                choice_logprobs = F.log_softmax(choice_logits, dim=-1)
                token_logprobs = choice_logprobs.gather(1, choice_ids[0].unsqueeze(1)).squeeze(1)
                log_probs.append(token_logprobs.sum().item())

            predicted = max(range(len(log_probs)), key=lambda i: log_probs[i])
            if predicted == example["gold_idx"]:
                correct += 1
            total += 1

    model.train()
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


# ─────────────────────────────────────────────────────────────
# Run All Evaluations
# ─────────────────────────────────────────────────────────────

def run_all_evals(model, tokenizer, val_ds, device, benchmarks, batch_size=8,
                  max_batches=100, max_examples=None):
    """
    Run val_loss + all benchmark evaluations.
    Returns combined results dict.
    """
    results = {}

    val_loss, val_ppl = evaluate(model, val_ds, batch_size=batch_size,
                                 device=device, max_batches=max_batches)
    results["val_loss"] = val_loss
    results["val_ppl"] = val_ppl

    for bench in benchmarks:
        if bench == "hellaswag":
            results["hellaswag"] = evaluate_hellaswag(
                model, tokenizer, device, max_examples=max_examples, batch_size=batch_size)
        elif bench == "arc_challenge":
            results["arc_challenge"] = evaluate_arc_challenge(
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

        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            probs = F.softmax(sorted_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_idx)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": text})

    model.train()
    return results
