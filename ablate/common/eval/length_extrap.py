"""Length-extrapolation bpb: evaluate bits-per-byte at increasing context windows, per language
(English AND Hindi). Directly probes partial RoPE (BiBo) vs full RoPE (Qwen) — the one place the two
arms must diverge mechanistically. Reports bpb[lang][L] and the degradation ratio vs the train length.

Concatenates each language's short held-out sentences (FLORES) into a long stream, then measures bpb
with different window sizes L. Run offline self-check:  python -m ablate.common.eval.length_extrap
"""
from .. import _paths  # noqa: F401
import torch
from .bpb import _text_nll_bytes, LN2


def run_length_extrap(model, tokenizer, manifest, lengths=(1024, 2048, 4096), train_len=1024,
                      device="cuda", dtype=torch.bfloat16, max_chars=200_000):
    """manifest: bpb Source list (uses domain=='text' sources). Returns {lang: {L: bpb, ...,
    'degradation': bpb[maxL]/bpb[train_len]}} for each language present (en, hi, ...)."""
    # build one long text per language from the text-domain sources
    lang_text = {}
    for src in manifest:
        if src.domain != "text":
            continue
        joined = "\n".join(src.loader(src.n))[:max_chars]
        lang_text[src.lang] = (lang_text.get(src.lang, "") + "\n" + joined)[:max_chars]
    out = {}
    for lang, text in lang_text.items():
        per_L = {}
        for L in lengths:
            nll, nb, _ = _text_nll_bytes(model, tokenizer, text, L, device, dtype)
            per_L[L] = (nll / LN2) / max(nb, 1)
        base = per_L.get(train_len) or next(iter(per_L.values()))
        per_L["degradation"] = per_L[max(lengths)] / max(base, 1e-9)   # >1 = worse at long context
        out[lang] = per_L
    return out


def _selfcheck():
    import math
    import torch.nn as nn
    from .manifest import SELFCHECK_MANIFEST

    V = 256

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
            self.model = _Base(); self.lm_head = nn.Linear(8, V, bias=False)
            nn.init.zeros_(self.lm_head.weight)

    r = run_length_extrap(_Model().eval(), _CharTok(), SELFCHECK_MANIFEST,
                          lengths=(16, 32, 64), train_len=16, device="cpu", dtype=torch.float32)
    assert "hi" in r and "en" in r, "need en+hi length-extrap segments"
    # uniform model -> bpb == log2(V) at every length -> degradation ~1.0
    for lang, d in r.items():
        assert abs(d[64] - math.log2(V)) < 0.1 and abs(d["degradation"] - 1.0) < 0.02, (lang, d)
    print(f"[length_extrap self-check] en={ {k: round(v,2) for k,v in r['en'].items()} }  "
          f"hi_degradation={r['hi']['degradation']:.3f}  OK", flush=True)


if __name__ == "__main__":
    _selfcheck()
