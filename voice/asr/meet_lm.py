"""Meeting clips: CTC beam search with a KenLM word n-gram and / or hotwords (pyctcdecode), on logits that
ab_meeting.py --logits saved. CPU only, so it runs next to training.

    python voice/asr/meet_lm.py LOGITS.npz --tok TOKENIZER.model [--lm lm.bin --unigrams uni.txt]
           [--alpha 0 0.3 0.5 0.8] [--beta 0 1 2] [--bp 0 0.5 1.0] [--beam 32] [--hotwords w1 w2 ...] [--hw_weight 8]

alpha = LM weight (0 = no LM: plain beam search), beta = word insertion bonus, bp = blank penalty on the logits.
Prints one MEETLM line per (model, look-ahead, alpha, beta, bp) with pooled WER (S D I) and per-clip WER, then the
best setting per look-ahead and a leave-one-clip-out estimate (setting chosen on the other clip, scored on this one).
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import edits, normalize_v2  # noqa: E402

G = {}


def _init(a, labels):
    from pyctcdecode import build_ctcdecoder
    uni = [w.strip() for w in open(a.unigrams, encoding="utf-8")] if a.unigrams else None
    G["dec"] = {al: build_ctcdecoder(labels, kenlm_model_path=a.lm if al > 0 else None, unigrams=uni if al > 0 else None,
                                     alpha=al, beta=0.0, unk_score_offset=a.unk) for al in a.alpha}
    G["a"] = a


def _decode(job):
    key, z, al, be, bp = job
    a, dec = G["a"], G["dec"][al]
    dec.reset_params(alpha=al, beta=be) if al > 0 else None
    z = z.astype(np.float32).copy()
    z[:, -1] -= bp                                         # blank is the last class
    lp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    kw = {"hotwords": a.hotwords, "hotword_weight": a.hw_weight} if a.hotwords else {}
    return key, al, be, bp, dec.decode(lp, beam_width=a.beam, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--unigrams", default=None)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.8])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--bp", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--las", nargs="+", default=None, help="look-aheads to decode (keys in the npz), default all")
    ap.add_argument("--hotwords", nargs="*", default=None)
    ap.add_argument("--hw_weight", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--unk", type=float, default=-10.0, help="LM score offset of a word outside the unigrams")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if not a.lm:
        a.alpha = [0.0]
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tok)
    labels = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    labels = ["⁇" if p == "<unk>" else p for p in labels] + [""]   # unk: never a word in the refs; "" = blank (last)
    d = np.load(a.logits)
    refs = [r.split() for r in d["refs"]]
    n = sum(map(len, refs))
    keys = [k for k in d.files if "|" in k]
    groups = sorted({k.rsplit("|", 1)[0] for k in keys})
    if a.las:
        groups = [g for g in groups if g.split("|")[1] in a.las]
    jobs = [(g + f"|{i}", d[g + f"|{i}"], al, be, bp) for g in groups for i in range(len(refs))
            for al in a.alpha for be in (a.beta if al > 0 else [0.0]) for bp in a.bp]
    res = {}
    with Pool(a.workers, initializer=_init, initargs=(a, labels)) as p:
        for key, al, be, bp, text in p.imap_unordered(_decode, jobs, chunksize=1):
            g, i = key.rsplit("|", 1)
            res.setdefault((g, al, be, bp), {})[int(i)] = normalize_v2(text)
    table = {}
    for (g, al, be, bp), hyps in sorted(res.items()):
        sdi = [edits(refs[i], hyps[i]) for i in range(len(refs))]
        s, dd, ii = (sum(x[k] for x in sdi) for k in range(3))
        per = [100 * sum(sdi[i]) / len(refs[i]) for i in range(len(refs))]
        table[(g, al, be, bp)] = (100 * (s + dd + ii) / n, per)
        la = g.split("|")[1]
        print(f"MEETLM{a.tag} {g.split('|')[0]:28s} la {la:>3s} alpha {al:.2f} beta {be:.1f} bp {bp:.2f} | pooled "
              f"{100 * (s + dd + ii) / n:5.2f} (S {100 * s / n:5.2f} D {100 * dd / n:5.2f} I {100 * ii / n:5.2f}) | "
              + " ".join(f"{x:5.1f}" for x in per), flush=True)
    for g in groups:
        rows = {k[1:]: v for k, v in table.items() if k[0] == g}
        best = min(rows, key=lambda k: rows[k][0])
        loo = []
        for i in range(len(refs)):
            pick = min(rows, key=lambda k: sum(rows[k][1][j] for j in range(len(refs)) if j != i))
            loo.append(rows[pick][1][i])
        print(f"BEST{a.tag} {g} alpha/beta/bp {best} -> {rows[best][0]:.2f} | leave-one-clip-out per-clip {[round(x, 1) for x in loo]}",
              flush=True)


if __name__ == "__main__":
    main()
