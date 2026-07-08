"""Log-likelihood multiple-choice eval — pick the highest per-token-logprob option (NOT generation,
which is hopeless at 137M). Parallel English/Hindi benchmarks, accuracy reported per source and rolled
up per LANGUAGE (Hindi separate). Verified sources: Belebele (4-way), XNLI (3-way), Global-MMLU (4-way).

Run offline self-check (no downloads):  python -m ablate.common.eval.mcq
"""
from .. import _paths  # noqa: F401
import functools
import math
import torch
import torch.nn.functional as F


def _wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion (better than normal approx at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _z_vs_chance(k, n, chance):
    """z-score of the observed accuracy above the chance baseline (one-sample proportion test)."""
    if n == 0 or chance in (0.0, 1.0):
        return 0.0
    se = math.sqrt(chance * (1 - chance) / n)
    return round((k / n - chance) / se, 2) if se > 0 else 0.0


@torch.no_grad()
def _cont_logprob(model, tokenizer, context, continuation, device, dtype):
    """Per-token mean log-prob of `continuation` given `context` (length-normalized -> acc_norm)."""
    ctx = tokenizer.encode(context)
    full = tokenizer.encode(context + continuation)
    n_cont = len(full) - len(ctx)
    if n_cont <= 0:
        return -1e30
    ids = torch.tensor(full, dtype=torch.long, device=device)
    inp = ids[:-1].unsqueeze(0)
    tgt = ids[1:]
    with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
        out = model.model(input_ids=inp, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        logits = model.lm_head(h)
    logp = F.log_softmax(logits.float()[0], dim=-1)      # (T, V)
    # continuation targets occupy the last n_cont positions of tgt
    sel = logp[-n_cont:].gather(-1, tgt[-n_cont:].unsqueeze(-1)).squeeze(-1)
    return (sel.sum() / n_cont).item()


@torch.no_grad()
def run_mcq(model, tokenizer, sources, device="cuda", dtype=torch.bfloat16):
    """sources: list of dicts {name, lang, items:[{context, options:[str], gold:int}]}.
    Returns per_source acc + per_language acc + overall."""
    per_source, lang_acc = {}, {}
    tot_correct = tot_n = 0
    for src in sources:
        correct = 0
        items = src["items"]
        n_opt = len(items[0]["options"]) if items else 0
        chance = 1.0 / n_opt if n_opt else 0.0
        for it in items:
            scores = [_cont_logprob(model, tokenizer, it["context"], " " + opt, device, dtype)
                      for opt in it["options"]]
            correct += int(int(torch.tensor(scores).argmax()) == it["gold"])
        n = len(items)
        per_source[src["name"]] = {
            "acc": round(correct / max(n, 1), 4), "lang": src["lang"], "n": n, "n_options": n_opt,
            "chance": round(chance, 4), "ci95": _wilson(correct, n), "z_vs_chance": _z_vs_chance(correct, n, chance),
        }
        a = lang_acc.setdefault(src["lang"], [0, 0, 0.0])
        a[0] += correct; a[1] += n; a[2] += chance * n            # expected-chance-weighted by n
        tot_correct += correct; tot_n += n
    per_language = {}
    for lang, (k, n, exp_chance) in lang_acc.items():
        c = exp_chance / max(n, 1)
        per_language[lang] = {"acc": round(k / max(n, 1), 4), "n": n, "chance": round(c, 4),
                              "ci95": _wilson(k, n), "z_vs_chance": _z_vs_chance(k, n, c)}
    return {
        "per_source": per_source,
        "per_language": per_language,                            # includes "hi" (acc + n + ci95 + z_vs_chance)
        "overall": round(tot_correct / max(tot_n, 1), 4),
    }


# ───────────────────────── dataset adapters (build items lists) ─────────────────────────
def _take(ds, n):
    out = []
    for ex in ds:
        out.append(ex)
        if len(out) >= n:
            break
    return out


@functools.lru_cache(maxsize=None)                       # cached: periodic eval won't re-download
def belebele_items(lang_cfg, n=500):                     # 4-way reading comprehension, parallel
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", lang_cfg, split="test", streaming=True)
    items = []
    for e in _take(ds, n):
        ctx = f"{e['flores_passage']}\nQuestion: {e['question']}\nAnswer:"
        opts = [e["mc_answer1"], e["mc_answer2"], e["mc_answer3"], e["mc_answer4"]]
        items.append({"context": ctx, "options": opts, "gold": int(e["correct_answer_num"]) - 1})
    return items


@functools.lru_cache(maxsize=None)
def xnli_items(lang, n=500):                             # 3-way entailment, parallel (33% floor)
    from datasets import load_dataset
    ds = load_dataset("facebook/xnli", lang, split="test", streaming=True)
    verb = ["True", "Neither", "False"]                  # 0=entailment,1=neutral,2=contradiction
    items = []
    for e in _take(ds, n):
        ctx = f"{e['premise']} Question: {e['hypothesis']} True, False, or Neither? Answer:"
        items.append({"context": ctx, "options": verb, "gold": int(e["label"])})
    return items


@functools.lru_cache(maxsize=None)
def global_mmlu_items(lang, n=500):                      # 4-way knowledge (expect near-chance @137M)
    from datasets import load_dataset
    ds = load_dataset("CohereLabs/Global-MMLU", lang, split="test", streaming=True)
    items = []
    for e in _take(ds, n):
        ctx = (f"{e['question']}\nA. {e['option_a']}\nB. {e['option_b']}\n"
               f"C. {e['option_c']}\nD. {e['option_d']}\nAnswer:")
        items.append({"context": ctx, "options": ["A", "B", "C", "D"], "gold": "ABCD".index(e["answer"])})
    return items


def default_sources(n=500, with_global_mmlu=False):
    """Parallel En+Hi MCQ sources. Each built defensively — an unavailable/gated source is skipped
    (warned) rather than crashing the eval. Global-MMLU opt-in (near-chance at this scale)."""
    specs = [("belebele_en", "en", belebele_items, "eng_Latn"),
             ("belebele_hi", "hi", belebele_items, "hin_Deva"),
             ("xnli_en", "en", xnli_items, "en"),
             ("xnli_hi", "hi", xnli_items, "hi")]
    if with_global_mmlu:
        specs += [("gmmlu_en", "en", global_mmlu_items, "en"),
                  ("gmmlu_hi", "hi", global_mmlu_items, "hi")]
    srcs = []
    for name, lang, builder, arg in specs:
        try:
            srcs.append({"name": name, "lang": lang, "items": builder(arg, n)})
        except Exception as e:
            print(f"[mcq] skip {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    return srcs


# ───────────────────────── offline self-check ─────────────────────────
def _selfcheck():
    import torch.nn as nn

    class _CharTok:
        def encode(self, text):
            return list(text.encode("utf-8"))

    class _Base(nn.Module):
        def forward(self, input_ids, use_cache=False):
            B, S = input_ids.shape
            class O: pass
            o = O(); o.last_hidden_state = torch.zeros(B, S, 8); return o

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Base(); self.lm_head = nn.Linear(8, 256, bias=False)
            nn.init.zeros_(self.lm_head.weight)

    srcs = [
        {"name": "t_en", "lang": "en", "items": [
            {"context": "q?", "options": ["a", "bb", "ccc"], "gold": 0} for _ in range(5)]},
        {"name": "t_hi", "lang": "hi", "items": [
            {"context": "प्रश्न?", "options": ["क", "खख", "गगग"], "gold": 1} for _ in range(5)]},
    ]
    r = run_mcq(_Model().eval(), _CharTok(), srcs, device="cpu", dtype=torch.float32)
    assert "hi" in r["per_language"], "Hindi segment missing"
    assert all(0.0 <= d["acc"] <= 1.0 and "ci95" in d and "z_vs_chance" in d for d in r["per_language"].values())
    print(f"[mcq self-check] hi={r['per_language']['hi']} overall={r['overall']:.3f}  OK", flush=True)


if __name__ == "__main__":
    _selfcheck()
