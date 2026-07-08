"""Qualitative samples generated with a manual KV-CACHE decode loop (verified cache-correct on both arms;
BiBo .generate() works but Qwen's .generate() hits a mask-dtype bug in this transformers build, so we
decode manually with explicit cache_position and no integer mask). Default: 2 English + 2 Hindi prompts."""
from .. import _paths  # noqa: F401
import torch
from transformers import DynamicCache

DEFAULT_PROMPTS = [
    ("en", "The meaning of life is"),
    ("en", "Once upon a time, in a small village,"),
    ("hi", "भारत की राजधानी दिल्ली है और"),
    ("hi", "एक बार की बात है, एक गाँव में"),
]


@torch.no_grad()
def kv_generate(model, prompt_ids, max_new_tokens=48, eos_id=None):
    """Greedy decode with a KV cache. cache_position is explicit; no attention_mask is passed (single-token
    decode is trivially causal), which sidesteps the Qwen long-mask SDPA bug."""
    gen = prompt_ids.clone()
    cache = DynamicCache()
    past = 0
    cur = prompt_ids
    for _ in range(max_new_tokens):
        q_len = cur.shape[1]
        cache_position = torch.arange(past, past + q_len, device=cur.device)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True, cache_position=cache_position)
        cache = out.past_key_values
        past += q_len
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, cur], dim=1)
        if eos_id is not None and int(cur.item()) == eos_id:
            break
    return gen


@torch.no_grad()
def generate_samples(model, tokenizer, prompts=DEFAULT_PROMPTS, max_new_tokens=48,
                     device="cuda", dtype=torch.bfloat16):
    """Returns [{lang, prompt, completion}] — 2 en + 2 hi by default, via the KV-cache loop."""
    was_training = model.training
    model.eval()
    eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    out = []
    for lang, text in prompts:
        ids = torch.tensor([tokenizer.encode(text)], device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
            gen = kv_generate(model, ids, max_new_tokens=max_new_tokens, eos_id=eos)
        completion = tokenizer.decode(gen[0, ids.shape[1]:].tolist())
        out.append({"lang": lang, "prompt": text, "completion": completion})
    if was_training:
        model.train()
    return out
