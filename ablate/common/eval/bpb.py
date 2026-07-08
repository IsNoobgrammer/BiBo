"""Bits-per-byte eval — tokenizer-independent, per source, rolled up per LANGUAGE and per DOMAIN.

bpb = (sum teacher-forced NLL in nats / ln2) / total UTF-8 bytes. Tokenizer-independent because the
denominator is bytes, so BiBo and Qwen are comparable regardless of tokenization. Hindi is reported
as its own language segment.

Run standalone self-check (no data/tokenizer download):  python -m ablate.common.eval.bpb
"""
from .. import _paths  # noqa: F401
import math
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


@torch.no_grad()
def _text_nll_bytes(model, tokenizer, text, seq_len, device, dtype):
    """Return (sum_nll_nats, n_bytes, n_pred_tokens) for one text under teacher forcing."""
    ids = tokenizer.encode(text)
    n_bytes = len(text.encode("utf-8"))
    if len(ids) < 2:
        return 0.0, n_bytes, 0
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    nll = 0.0
    n_pred = 0
    for i in range(0, len(ids) - 1, seq_len):
        window = ids[i:i + seq_len + 1]
        if window.numel() < 2:
            break
        inp = window[:-1].unsqueeze(0)
        tgt = window[1:].unsqueeze(0)
        with torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
            out = model.model(input_ids=inp, use_cache=False)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            logits = model.lm_head(h)
        nll += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               tgt.reshape(-1), reduction="sum").item()
        n_pred += tgt.numel()
    return nll, n_bytes, n_pred


def run_bpb(model, tokenizer, manifest, seq_len=1024, device="cuda", dtype=torch.bfloat16, n_override=None):
    """Returns dict: per_source / per_language / per_domain / overall bpb (+ token counts).
    n_override caps samples/source (cheap periodic eval); None = each source's full n."""
    per_source = {}
    lang_acc, dom_acc = {}, {}
    tot_nll = tot_bytes = 0.0
    for src in manifest:
        s_nll = s_bytes = s_tok = 0.0
        n = min(src.n, n_override) if n_override else src.n
        for text in src.loader(n):
            nll, nb, nt = _text_nll_bytes(model, tokenizer, text, seq_len, device, dtype)
            s_nll += nll; s_bytes += nb; s_tok += nt
        bpb = (s_nll / LN2) / max(s_bytes, 1)
        per_source[src.name] = {"bpb": bpb, "lang": src.lang, "domain": src.domain, "tokens": int(s_tok)}
        for acc, key in ((lang_acc, src.lang), (dom_acc, src.domain)):
            a = acc.setdefault(key, [0.0, 0.0])
            a[0] += s_nll; a[1] += s_bytes
        tot_nll += s_nll; tot_bytes += s_bytes
    roll = lambda acc: {k: (v[0] / LN2) / max(v[1], 1) for k, v in acc.items()}
    return {
        "per_source": per_source,
        "per_language": roll(lang_acc),           # includes "hi" explicitly
        "per_domain": roll(dom_acc),
        "overall": (tot_nll / LN2) / max(tot_bytes, 1),
    }


# ───────────────────────── self-check (no downloads) ─────────────────────────
def _selfcheck():
    import torch.nn as nn
    from .manifest import SELFCHECK_MANIFEST

    V = 256

    class _CharTok:                       # byte-level: 1 token per byte -> n_tokens == n_bytes
        def encode(self, text):
            return list(text.encode("utf-8"))

    class _Base(nn.Module):               # returns zeros -> uniform logits -> NLL = ln(V) per token
        def forward(self, input_ids, use_cache=False):
            B, S = input_ids.shape
            class O: pass
            o = O(); o.last_hidden_state = torch.zeros(B, S, 8); return o

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Base()
            self.lm_head = nn.Linear(8, V, bias=False)
            nn.init.zeros_(self.lm_head.weight)   # all logits 0 -> uniform softmax

    m = _Model().eval()
    r = run_bpb(m, _CharTok(), SELFCHECK_MANIFEST, seq_len=64, device="cpu", dtype=torch.float32)
    expected = math.log2(V)               # uniform over V, 1 token/byte -> bpb == log2(V)
    got = r["overall"]
    assert "hi" in r["per_language"], "Hindi segment missing from rollup"
    assert abs(got - expected) < 0.05, f"bpb formula off: got {got:.4f} expected {expected:.4f}"
    print(f"[bpb self-check] overall bpb={got:.4f} (expected log2({V})={expected:.4f})  "
          f"langs={list(r['per_language'])}  OK", flush=True)


if __name__ == "__main__":
    _selfcheck()
