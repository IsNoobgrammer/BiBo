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


def rep_ngram(ids, n=4):
    """Fraction of n-grams that are repeats. 0 = every window unique, ->1 = looping."""
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 0.0 if not g else 1.0 - len(set(g)) / len(g)


@torch.no_grad()
def kv_generate(model, prompt_ids, max_new_tokens=48, eos_id=None):
    """Greedy decode with a KV cache. cache_position is explicit; no attention_mask is passed (single-token
    decode is trivially causal), which sidesteps the Qwen long-mask SDPA bug.

    Also returns the per-step ENTROPY and top-1 probability of the next-token distribution, because
    bpb cannot see what they see. bpb is teacher-forced mean NLL, and cross-entropy decomposes as
    model entropy + KL: a model can lower it by matching the marginal while staying flat, never
    committing to anything. Greedy decode then argmaxes a near-uniform distribution and one bad
    pick cascades. Measured on the box-5 checkpoints: the BEST bpb of three (per-dim c+d, 0.6609)
    generated punctuation salad at entropy 4.9/6.5, while the WORST (b1s2 softmax, 0.6654) wrote
    the only coherent text at entropy 2.26.
    """
    gen = prompt_ids.clone()
    cache = DynamicCache()
    past = 0
    cur = prompt_ids
    ent, top1 = [], []
    for _ in range(max_new_tokens):
        q_len = cur.shape[1]
        cache_position = torch.arange(past, past + q_len, device=cur.device)
        out = model(input_ids=cur, past_key_values=cache, use_cache=True, cache_position=cache_position)
        cache = out.past_key_values
        past += q_len
        lg = out.logits[:, -1].float()
        p = lg.softmax(-1)
        ent.append(float(-(p * (p + 1e-12).log()).sum()))
        top1.append(float(p.max()))
        cur = lg.argmax(-1, keepdim=True)
        gen = torch.cat([gen, cur], dim=1)
        if eos_id is not None and int(cur.item()) == eos_id:
            break
    return gen, ent, top1


@torch.no_grad()
def generate_samples(model, tokenizer, prompts=DEFAULT_PROMPTS, max_new_tokens=48,
                     device="cuda", dtype=torch.bfloat16):
    """Returns [{lang, prompt, completion, entropy, top1, rep4}] — 2 en + 2 hi by default.

    entropy/top1/rep4 are the health metrics. They catch TWO opposite pathologies that bpb ranks
    identically and that repetition alone cannot tell apart:
        low entropy + high rep4  -> confident loop     (measured: 0.80 / 0.93)
        high entropy + high rep4 -> no commitment      (measured: 4.92 / 0.83)
        healthy                                        (measured: 2.26 / 0.03)
    So entropy is not to be minimised -- minimising it IS the looping failure. It wants a band.
    """
    was_training = model.training
    model.eval()
    eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    out = []
    for lang, text in prompts:
        ids = torch.tensor([tokenizer.encode(text)], device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
            gen, ent, top1 = kv_generate(model, ids, max_new_tokens=max_new_tokens, eos_id=eos)
        new_ids = gen[0, ids.shape[1]:].tolist()
        completion = tokenizer.decode(new_ids)
        out.append({"lang": lang, "prompt": text, "completion": completion,
                    "entropy": sum(ent) / max(len(ent), 1),
                    "top1": sum(top1) / max(len(top1), 1),
                    "rep4": rep_ngram(new_ids, 4)})
    if was_training:
        model.train()
    return out
