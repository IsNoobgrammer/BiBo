"""In-context-learning (ICL) curve — a SEPARATE eval metric with its own `eval/icl_*` namespace.
NEVER folded into probes / mcq / bpb: it is computed and logged on its own so it can be read
independently of the loss metrics.

It measures how much in-context (key -> value) bindings help recall as the number of demonstrated
pairs (shots) grows. For each shot count k and language it reports:
  acc(k) : LL-MCQ accuracy of recalling the queried key's value (chance = 1/n_options)
  nll(k) : mean negative log-prob (nats/token) of the gold value given the context

Headlines (the "slope"):
  jump_acc   = acc@1 - acc@0                 raw induction — does ONE in-context binding help
  slope_acc  = d(acc) / d(log2 k),  k>=1     does accuracy hold/rise as context grows
  slope_nll  = d(nll) / d(log2 k),  k>=1     does the gold get cheaper (more confident) with more shots

English + Hindi (Devanagari) in parity. Fully deterministic (no downloads, no RNG that would break
resume). Offline self-check:  python -m ablate.common.eval.icl
"""
from .. import _paths  # noqa: F401
import math
import torch
from .mcq import _cont_logprob
from .probes import _EN, _HI

SHOTS = (0, 1, 2, 4, 8)


def _icl_items(vocab, k, n=100, n_opt=4):
    """k in-context (key: value) pairs, then query one bound key; options = gold + distractors.
    k=0 = no binding in context (anchor: gold not derivable -> ~chance)."""
    V = len(vocab)
    items = []
    for i in range(n):
        if k == 0:
            gold = vocab[i % V]
            ctx = "k0:"
        else:
            keys = [f"k{j}" for j in range(k)]
            vals = [vocab[(i + j) % V] for j in range(k)]
            qi = i % k
            ctx = " ".join(f"{keys[j]}: {vals[j]}." for j in range(k)) + f" {keys[qi]}:"
            gold = vals[qi]
        # distinct distractors != gold
        distract, d = [], 0
        while len(distract) < n_opt - 1:
            cand = vocab[(i + k + d) % V]
            if cand != gold and cand not in distract:
                distract.append(cand)
            d += 1
        items.append({"context": ctx, "options": [gold] + distract, "gold": 0})
    return items


@torch.no_grad()
def _score(model, tokenizer, items, device, dtype):
    correct, nll_sum = 0, 0.0
    for it in items:
        scores = [_cont_logprob(model, tokenizer, it["context"], " " + opt, device, dtype)
                  for opt in it["options"]]
        correct += int(max(range(len(scores)), key=lambda j: scores[j]) == it["gold"])
        nll_sum += -scores[it["gold"]]                       # -mean-logprob of gold = nll/token
    n = len(items)
    return correct / max(n, 1), nll_sum / max(n, 1)


def _slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0


@torch.no_grad()
def run_icl(model, tokenizer, shots=SHOTS, n=100, n_opt=4, device="cuda", dtype=None):
    """Returns nested result incl per-language acc/nll curves, slopes, and a `flat` dict of eval/icl_*."""
    dtype = dtype or torch.bfloat16
    langs = {"en": _EN, "hi": _HI}
    result = {"shots": list(shots), "chance": round(1.0 / n_opt, 4), "per_language": {}, "flat": {}}
    for lang, vocab in langs.items():
        acc, nll = {}, {}
        for k in shots:
            a, nl = _score(model, tokenizer, _icl_items(vocab, k, n, n_opt), device, dtype)
            acc[k], nll[k] = round(a, 4), round(nl, 4)
        ks = [k for k in shots if k >= 1]
        xs = [math.log2(k) for k in ks]
        slope_acc = round(_slope(xs, [acc[k] for k in ks]), 4)
        slope_nll = round(_slope(xs, [nll[k] for k in ks]), 4)
        jump = round(acc.get(1, 0.0) - acc.get(0, 0.0), 4) if 0 in shots else None
        result["per_language"][lang] = {"acc": acc, "nll": nll, "slope_acc": slope_acc,
                                        "slope_nll": slope_nll, "jump_acc": jump}
        for k in shots:
            result["flat"][f"eval/icl_acc_k{k}_{lang}"] = acc[k]
            result["flat"][f"eval/icl_nll_k{k}_{lang}"] = nll[k]
        result["flat"][f"eval/icl_slope_acc_{lang}"] = slope_acc
        result["flat"][f"eval/icl_slope_nll_{lang}"] = slope_nll
        if jump is not None:
            result["flat"][f"eval/icl_jump_acc_{lang}"] = jump
    return result


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

    r = run_icl(_Model().eval(), _CharTok(), n=8, device="cpu", dtype=torch.float32)
    assert set(r["per_language"]) == {"en", "hi"}, "need en+hi ICL segments"
    for lang, d in r["per_language"].items():
        assert set(d["acc"]) == set(SHOTS) and all(0.0 <= v <= 1.0 for v in d["acc"].values())
        assert all(v >= 0.0 for v in d["nll"].values())
    assert any(k.startswith("eval/icl_slope_acc_") for k in r["flat"]), "missing slope in flat dict"
    print(f"[icl self-check] en slope_acc={r['per_language']['en']['slope_acc']} "
          f"hi slope_acc={r['per_language']['hi']['slope_acc']}  flat_keys={len(r['flat'])}  OK", flush=True)


if __name__ == "__main__":
    _selfcheck()
