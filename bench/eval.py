"""
BiBo Benchmark — Evaluation & Sample Generation

Validation loss computation + text generation using QTK-81K tokenizer.
Tokenizer is used ONLY here (for inference/decode), NOT during training.
"""

import math
import torch
import torch.nn.functional as F
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

        # Count valid tokens (not -100)
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
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    return avg_loss, perplexity


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
