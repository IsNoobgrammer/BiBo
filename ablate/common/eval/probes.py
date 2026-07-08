"""Small-scale-sensitive capability probes, English AND Hindi (parity). Synthetic, deterministic (no
downloads), scored by the same LL-MCQ machinery as mcq.py so a 137M model gives real signal.

Probes:
  induction : in-context key->value recall (tests induction heads / attention+rope).
  copy      : reproduce a token seen earlier in context (selective copy).
Each built as MCQ items (pick the right value among distractors) in en + hi (Devanagari) -> per-language
accuracy, random = 1/n_options. Run offline self-check:  python -m ablate.common.eval.probes
"""
from .. import _paths  # noqa: F401
from .mcq import run_mcq

# deterministic vocabularies (no RNG that would break resume); en tokens + hi (Devanagari) tokens
_EN = ["apple", "river", "table", "cloud", "stone", "tiger", "green", "music", "seven", "north"]
_HI = ["सेब", "नदी", "मेज", "बादल", "पत्थर", "बाघ", "हरा", "संगीत", "सात", "उत्तर"]


def _induction_items(vocab, n=100, pairs=4, n_opt=4):
    items = []
    for i in range(n):
        keys = [f"k{j}" for j in range(pairs)]
        vals = [vocab[(i + j) % len(vocab)] for j in range(pairs)]
        q = i % pairs                                   # which key to query
        ctx = " ".join(f"{keys[j]}: {vals[j]}." for j in range(pairs)) + f" {keys[q]}:"
        gold_val = vals[q]
        distract = [vocab[(i + pairs + d) % len(vocab)] for d in range(n_opt - 1)]
        options = [gold_val] + distract
        items.append({"context": ctx, "options": options, "gold": 0})
    return items


def _copy_items(vocab, n=100, span=5, n_opt=4):
    items = []
    for i in range(n):
        seq = [vocab[(i + j) % len(vocab)] for j in range(span)]
        pos = i % span
        ctx = "sequence: " + " ".join(seq) + f" | token {pos} is:"
        gold = seq[pos]
        distract = [vocab[(i + span + d) % len(vocab)] for d in range(n_opt - 1)]
        items.append({"context": ctx, "options": [gold] + distract, "gold": 0})
    return items


def default_sources(n=100):
    return [
        {"name": "induction_en", "lang": "en", "items": _induction_items(_EN, n)},
        {"name": "induction_hi", "lang": "hi", "items": _induction_items(_HI, n)},
        {"name": "copy_en", "lang": "en", "items": _copy_items(_EN, n)},
        {"name": "copy_hi", "lang": "hi", "items": _copy_items(_HI, n)},
    ]


def run_probes(model, tokenizer, n=100, device="cuda", dtype=None):
    import torch
    dtype = dtype or torch.bfloat16
    return run_mcq(model, tokenizer, default_sources(n), device=device, dtype=dtype)


def _selfcheck():
    import torch
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

    r = run_probes(_Model().eval(), _CharTok(), n=10, device="cpu", dtype=torch.float32)
    assert "hi" in r["per_language"] and "en" in r["per_language"], "need en+hi probe segments"
    assert all(0.0 <= d["acc"] <= 1.0 for d in r["per_language"].values())
    print(f"[probes self-check] en_acc={r['per_language']['en']['acc']} hi_acc={r['per_language']['hi']['acc']}  OK", flush=True)


if __name__ == "__main__":
    _selfcheck()
