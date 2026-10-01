"""Per-language val + expert specialisation by language on a checkpoint (the corpus is hi35 / en65).

    python -m ablate.tools.lang_probe --result a_result.json,b_result.json [--seqs 32]
    python -m ablate.tools.lang_probe --repo fhai50032/bibo-base-1b-6k-s23 --subs final

Each held-out TARGET token is classed by the script of its decoded text: `hi` (any Devanagari U+0900-097F),
`en` (ASCII letters, no Devanagari), `other` (digits / punctuation / whitespace / specials).
  val      per class: token share, next-token CE (the token being PREDICTED is the classed one)
  routing  per routed layer, on the token whose hidden state is routed (position t, classed by token t itself):
           JS(hi || en) of the expert-usage distributions in bits (0 = same experts, 1 = disjoint), the number of
           experts whose hi share of (hi + en) selections is > 2x / < 0.5x the layer's overall hi share (language-
           leaning experts), and the mean top-1 routing weight per class (router confidence per language)
"""
from ablate.common import _paths  # noqa: F401
import argparse
import math

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_hub, load_from_result
from ablate.common import validation as _val

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


def classify(tok, ids):
    """token id list -> list of 'hi' | 'en' | 'other' by the script of each token's decoded text."""
    out = []
    for t in ids:
        s = tok.decode([t])
        if any("ऀ" <= ch <= "ॿ" for ch in s):
            out.append("hi")
        elif any(ch.isascii() and ch.isalpha() for ch in s):
            out.append("en")
        else:
            out.append("other")
    return out


def js_bits(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12) / b.clamp_min(1e-12)).log2()).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


@torch.no_grad()
def probe(model, tok, hold):
    layers = [(i, l) for i, l in enumerate(model.model.layers)
              if hasattr(getattr(l, "mlp", None), "experts") and l.mlp.gate.top_k < l.mlp.gate.num_routed_experts]
    E = {i: l.mlp.gate.num_routed_experts for i, l in layers}
    cap = {i: [] for i, _ in layers}

    def mk(i):
        def hook(mod, args):
            k = args[1].shape[-1]
            cap[i].append((args[1].reshape(-1, k).long(), args[2].reshape(-1, k).float()))
        return hook

    hs = [l.mlp.experts.register_forward_pre_hook(mk(i)) for i, l in layers]
    W = model.lm_head.weight
    ce_all, cls_tgt, cls_pos = [], [], []
    for b in range(hold.shape[0]):
        ids = hold[b]
        with AMP:
            h = model.model(input_ids=ids[None, :-1], use_cache=False).last_hidden_state[0]
        ce_all.append(F.cross_entropy(h.float() @ W.float().t(), ids[1:], reduction="none"))
        c = classify(tok, ids.tolist())
        cls_pos += c[:-1]            # routed token = input position t
        cls_tgt += c[1:]             # predicted token = t + 1
    for hk in hs:
        hk.remove()
    ce = torch.cat(ce_all)
    val = {}
    for c in ("hi", "en", "other"):
        m = torch.tensor([x == c for x in cls_tgt], device=DEV)
        val[c] = (m.float().mean().item(), ce[m].mean().item() if m.any() else float("nan"))
    val["all"] = (1.0, ce.mean().item())
    pos = {c: torch.tensor([x == c for x in cls_pos], device=DEV) for c in ("hi", "en")}
    route = {}
    for i, _ in layers:
        idx = torch.cat([a for a, _ in cap[i]]); w = torch.cat([b for _, b in cap[i]])
        cnt = {c: torch.bincount(idx[pos[c]].reshape(-1), minlength=E[i]).float() for c in pos}
        p = {c: cnt[c] / cnt[c].sum() for c in cnt}
        hi_share = cnt["hi"] / (cnt["hi"] + cnt["en"]).clamp_min(1)
        g = float(cnt["hi"].sum() / (cnt["hi"].sum() + cnt["en"].sum()))
        route[i] = dict(js=js_bits(p["hi"], p["en"]), hi_lean=int((hi_share > min(2 * g, 0.999)).sum()),
                        en_lean=int((hi_share < 0.5 * g).sum()),
                        top1_hi=w[pos["hi"]].max(-1).values.mean().item(), top1_en=w[pos["en"]].max(-1).values.mean().item())
    return val, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    ap.add_argument("--subs", default="final")
    ap.add_argument("--result", default="")
    ap.add_argument("--seqs", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("fhai50032/QTK-81K")
    srcs = [("result", p) for p in a.result.split(",") if p] or [("hub", s) for s in a.subs.split(",")]
    hold = None
    for kind, sub in srcs:
        model, cfg = load_from_result(sub) if kind == "result" else load_from_hub(a.repo, sub)
        model.eval()
        if hold is None:
            hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, DEV)
        val, route = probe(model, tok, hold)
        name = cfg.run_tag if kind == "result" else f"{a.repo.split('/')[-1]}/{sub}"
        print(f"\n==== {name}  ({a.seqs} x 1024 holdout tokens)")
        print("val CE by language of the predicted token: " +
              " | ".join(f"{c} {v[1]:.4f} ({100 * v[0]:.1f}% of tokens)" for c, v in val.items()))
        print(f"{'L':>2} {'JS(hi||en) bits':>16} {'hi-leaning':>11} {'en-leaning':>11} {'top1 w hi':>10} {'top1 w en':>10}")
        for i, r in route.items():
            print(f"{i:>2} {r['js']:16.4f} {r['hi_lean']:11d} {r['en_lean']:11d} {r['top1_hi']:10.4f} {r['top1_en']:10.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
