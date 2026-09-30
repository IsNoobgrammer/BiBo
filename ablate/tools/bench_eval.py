"""Zero- and few-shot benchmarks + text samples for a --hf_repo checkpoint (English and Hindi).

    python -m ablate.tools.bench_eval --repo fhai50032/bibo-base-1b-6k-s23 --sub step4000 [--limit N]
    python -m ablate.tools.bench_eval --repo ... --sub step3000 --samples_only

Scoring follows lm-evaluation-harness zero-shot conventions so the numbers sit next to published tables:
multiple choice = sum log p(continuation | context); `acc` picks the max raw log-likelihood, `acc_norm`
the max per-BYTE log-likelihood (HellaSwag / ARC / PIQA are usually quoted as acc_norm). WinoGrande uses
partial scoring (the option goes in the CONTEXT, the shared suffix is scored). LAMBADA: exact greedy match
of every token of the last word. Every sequence starts with <|im_end|> (81914), the corpus's document
separator, so the model sees a document start the way it did in training.

Tasks: en hellaswag, arc_easy, piqa, winogrande, lambada, belebele_eng | hi xstorycloze_hi, belebele_hin.
Chance: 25 / 25 / 50 / 50 / 0 / 25 | 50 / 25.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import json
import re

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_hub

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
SEP = 81914
PAD = 0


def _hs_pre(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def tasks(limit, only=None):
    from datasets import load_dataset
    L = (lambda ds: ds.select(range(min(limit, len(ds))))) if limit else (lambda ds: ds)
    out = {}

    def add(name, fn):
        if only and name not in only:
            return
        try:
            out[name] = fn()
        except Exception as e:                       # a dataset that will not load is reported, not fatal
            print(f"[bench] {name} SKIPPED: {type(e).__name__}: {str(e)[:120]}", flush=True)

    def hellaswag():
        r = []
        for d in L(load_dataset("Rowan/hellaswag", split="validation")):
            ctx = _hs_pre(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
            r.append(("mc", ctx, [" " + _hs_pre(e) for e in d["endings"]], int(d["label"])))
        return r

    def arc_easy():
        r = []
        for d in L(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")):
            labs = d["choices"]["label"]
            r.append(("mc", f"Question: {d['question']}\nAnswer:", [" " + t for t in d["choices"]["text"]],
                      labs.index(d["answerKey"])))
        return r

    def piqa():
        r = []
        for d in L(load_dataset("baber/piqa", split="validation")):
            r.append(("mc", f"Question: {d['goal']}\nAnswer:", [" " + d["sol1"], " " + d["sol2"]], int(d["label"])))
        return r

    def winogrande():
        r = []
        for d in L(load_dataset("allenai/winogrande", "winogrande_xl", split="validation")):
            i = d["sentence"].index("_")
            suffix = " " + d["sentence"][i + 1:].strip()
            ctxs = [d["sentence"][:i] + d["option1"], d["sentence"][:i] + d["option2"]]
            r.append(("partial", ctxs, suffix, int(d["answer"]) - 1))
        return r

    def lambada():
        r = []
        for d in L(load_dataset("EleutherAI/lambada_openai", "en", split="test")):
            ctx, w = d["text"].rsplit(" ", 1)
            r.append(("greedy", ctx, " " + w, 0))
        return r

    def xstory_hi():
        r = []
        for d in L(load_dataset("juletxara/xstory_cloze", "hi", split="eval")):
            ctx = " ".join(d[f"input_sentence_{k}"] for k in range(1, 5))
            r.append(("mc", ctx, [" " + d["sentence_quiz1"], " " + d["sentence_quiz2"]], int(d["answer_right_ending"]) - 1))
        return r

    def belebele(lang):
        def f():
            r = []
            for d in L(load_dataset("facebook/belebele", lang, split="test")):
                ctx = (f"P: {d['flores_passage']}\nQ: {d['question'].strip()}\nA: {d['mc_answer1']}\n"
                       f"B: {d['mc_answer2']}\nC: {d['mc_answer3']}\nD: {d['mc_answer4']}\nAnswer:")
                r.append(("mc", ctx, [" A", " B", " C", " D"], int(d["correct_answer_num"]) - 1))
            return r
        return f

    add("hellaswag", hellaswag)
    add("arc_easy", arc_easy)
    add("piqa", piqa)
    add("winogrande", winogrande)
    add("lambada", lambada)
    add("belebele_eng", belebele("eng_Latn"))
    add("xstorycloze_hi", xstory_hi)
    add("belebele_hin", belebele("hin_Deva"))
    return out


class Scorer:
    def __init__(self, model, tok, max_len=1024, bs=32):
        self.m, self.tok, self.max_len, self.bs = model, tok, max_len, bs
        self.W = model.lm_head.weight

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    @torch.no_grad()
    def loglik(self, pairs):
        """[(context str, continuation str)] -> [(sum logp, all-greedy bool, n bytes)]."""
        items = []
        for c, x in pairs:
            ci, xi = [SEP] + self.enc(c), self.enc(x)
            ids = (ci + xi)[-self.max_len:]
            items.append((ids, len(xi), len(x.encode("utf-8"))))
        order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
        res = [None] * len(items)
        for b in range(0, len(order), self.bs):
            idx = order[b:b + self.bs]
            T = max(len(items[i][0]) for i in idx)
            T = (T + 127) // 128 * 128              # right-pad (causal: pads never reach real positions)
            batch = torch.full((len(idx), T), PAD, dtype=torch.long)
            for r, i in enumerate(idx):
                batch[r, :len(items[i][0])] = torch.tensor(items[i][0])
            batch = batch.to(DEV)
            with AMP:
                h = self.m.model(input_ids=batch, use_cache=False).last_hidden_state
            for r, i in enumerate(idx):
                ids, n, nb = items[i]
                s = len(ids) - n
                lg = F.log_softmax(h[r, s - 1:len(ids) - 1].float() @ self.W.float().t(), -1)
                tgt = batch[r, s:len(ids)]
                lp = lg.gather(-1, tgt[:, None]).sum().item()
                res[i] = (lp, bool((lg.argmax(-1) == tgt).all()), nb)
        return res

    def run(self, items, shots=0, seed=1234):
        """shots > 0: prepend `shots` solved examples drawn from the SAME split (never the scored item),
        fixed per item by seed, joined by blank lines -- the lm-eval few-shot layout."""
        import random
        demo = lambda it: (it[1] + it[2][it[3]] if it[0] == "mc" else
                           it[1][it[3]] + it[2] if it[0] == "partial" else it[1] + it[2])
        pairs, spans = [], []
        for j, (kind, ctx, cont, gold) in enumerate(items):
            if shots:
                rng = random.Random(seed + j)
                pool = [i for i in rng.sample(range(len(items)), shots + 1) if i != j][:shots]
                pre = "\n\n".join(demo(items[i]) for i in pool) + "\n\n"
                ctx = [pre + c for c in ctx] if kind == "partial" else pre + ctx
            k0 = len(pairs)
            if kind == "mc":
                pairs += [(ctx, c) for c in cont]
            elif kind == "partial":
                pairs += [(c, cont) for c in ctx]
            else:
                pairs.append((ctx, cont))
            spans.append((k0, len(pairs)))
        ll = self.loglik(pairs)
        acc = accn = 0
        for (kind, _c, _x, gold), (a, b) in zip(items, spans):
            r = ll[a:b]
            if kind == "greedy":
                acc += r[0][1]; accn += r[0][1]
                continue
            acc += max(range(len(r)), key=lambda j: r[j][0]) == gold
            accn += max(range(len(r)), key=lambda j: r[j][0] / max(r[j][2], 1)) == gold
        n = len(items)
        return 100 * acc / n, 100 * accn / n, n


PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a little girl who",
    "The main reason the sky looks blue is",
    "Here is a simple recipe for making tea:",
    "In 1947, India",
    "भारत की राजधानी",
    "एक बार की बात है, एक गाँव में",
    "स्वस्थ रहने के लिए हमें",
]


@torch.no_grad()
def samples(model, tok, n_new=80, temp=0.8, topk=50, seed=0):
    W = model.lm_head.weight
    g = torch.Generator(device=DEV).manual_seed(seed)
    for p in PROMPTS:
        for mode in ("greedy", f"T={temp}"):
            ids = torch.tensor([[SEP] + tok.encode(p, add_special_tokens=False)], device=DEV)
            for _ in range(n_new):
                with AMP:
                    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
                lg = h.float() @ W.float().t()
                if mode == "greedy":
                    nx = lg.argmax()
                else:
                    v, ix = (lg / temp).topk(topk)
                    nx = ix[torch.multinomial(torch.softmax(v, -1), 1, generator=g)][0]
                if int(nx) in (SEP, PAD):
                    break
                ids = torch.cat([ids, nx.view(1, 1)], 1)
            text = tok.decode(ids[0, 1:].tolist())
            print(f"--- [{mode}] {p!r}\n{text}\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sub", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--samples_only", action="store_true")
    ap.add_argument("--no_samples", action="store_true")
    ap.add_argument("--shots", default="0,5")
    ap.add_argument("--tasks", default="")          # comma list; empty = all
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    model, cfg = load_from_hub(a.repo, a.sub)
    model.eval()
    tok = AutoTokenizer.from_pretrained("fhai50032/QTK-81K")
    if not a.no_samples:
        print(f"\n==== samples: {a.repo}/{a.sub}")
        samples(model, tok)
    if a.samples_only:
        return
    sc = Scorer(model, tok, bs=a.bs)
    out = {}
    T = tasks(a.limit, set(filter(None, a.tasks.split(","))))
    for k in (int(v) for v in a.shots.split(",")):
        print(f"\n==== {k}-shot: {a.repo}/{a.sub}\n{'task':16s} {'n':>6s} {'acc':>7s} {'acc_norm':>9s}")
        for name, items in T.items():
            acc, accn, n = sc.run(items, shots=k)
            out[f"{name}_{k}shot"] = {"acc": acc, "acc_norm": accn, "n": n}
            print(f"{name:16s} {n:6d} {acc:7.2f} {accn:9.2f}", flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
