"""Final-checkpoint probe for the MTP arms: a big-holdout val and "does the head lean on Emb(t+1)?".

    python -m ablate.tools.mtp_probe <result.json> [<result.json> ...] [--n_seqs 256]

For every checkpoint, on the SAME frozen holdout (n_seqs x 1024 tokens, pad id 0 masked):
  val      main-head CE (+ bpb), token-weighted -- the 2-seq training val, with 128x the tokens
  mtp      the MTP head's CE vs t+2
  mtp_e0   same with Emb(t+1) replaced by zeros            (with-embedding heads only)
  mtp_eP   same with Emb(t+1) replaced by a shuffled token (with-embedding heads only)
  grads    at h = the trunk's final hidden state: |dCE1/dh|, |d(w*CE2)/dh|, cos(dCE1, dCE2), and
           |d(w*CE2)/dEmb(t+1)|  -- how much of the MTP gradient reaches the trunk vs the embedding

The MTP forward needs the trunk's RoPE tensors and block archive, which BiBoModel only hands over in
training mode; only the top module's flag is flipped (submodules stay in eval, so the MoE balancer
never updates). Checkpoints are final-step states, so the gradient numbers describe the END of
training, not its average.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import math

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


def ce_sum(x, W, tgt, pad):
    lg = (x.float() @ W.float().t())
    m = tgt != pad
    return F.cross_entropy(lg[m], tgt[m], reduction="sum"), int(m.sum())


def probe(res, hold, bpt, bs, mtp_w=0.3):
    model, c = load_from_result(res)
    W = model.lm_head.weight
    pad = 0
    out = {"tag": getattr(c, "run_tag", res)}
    has_mtp = getattr(model, "mtp", None) is not None
    use_emb = has_mtp and model.mtp.use_emb
    acc = {k: [0.0, 0] for k in ("val", "mtp", "mtp_e0", "mtp_eP")}
    gstats = []
    g = torch.Generator(device=DEV).manual_seed(0)
    for i in range(0, hold.shape[0], bs):
        ids = hold[i:i + bs]
        inp, t1 = ids[:, :-1], ids[:, 1:]
        t2 = torch.cat([ids[:, 2:], ids.new_full((ids.shape[0], 1), pad)], 1)
        with torch.no_grad(), AMP:
            model.model.training = True                     # hand over RoPE + archive (top flag only)
            h = model.model(input_ids=inp, use_cache=False).last_hidden_state
            model.model.training = False
            pe, br = model.model._mtp_cache
            H = h.shape[-1]
            s, n = ce_sum(h.reshape(-1, H), W, t1.reshape(-1), pad)
            acc["val"][0] += s.item(); acc["val"][1] += n
            if has_mtp:
                e = model.model.embed_tokens(t1)
                variants = {"mtp": e}
                if use_emb:
                    variants["mtp_e0"] = torch.zeros_like(e)
                    perm = torch.randint(0, e.shape[0] * e.shape[1], (e.shape[0] * e.shape[1],), device=DEV, generator=g)
                    variants["mtp_eP"] = e.reshape(-1, H)[perm].reshape(e.shape)
                for k, ev in variants.items():
                    x = model.mtp(h, ev, pe, br)
                    s, n = ce_sum(x.reshape(-1, H), W, t2.reshape(-1), pad)
                    acc[k][0] += s.item(); acc[k][1] += n
        if has_mtp and i < 4 * bs:                          # gradient probe on the first 4 slices
            with AMP:
                hh = h.detach().float().requires_grad_(True)
                ee = model.model.embed_tokens(t1).detach().float().requires_grad_(True)
                s1, n1 = ce_sum(hh.reshape(-1, H), W, t1.reshape(-1), pad)
                g1, = torch.autograd.grad(s1 / n1, hh)
                x = model.mtp(hh, ee, pe, br)
                s2, n2 = ce_sum(x.reshape(-1, H), W, t2.reshape(-1), pad)
                g2h, g2e = torch.autograd.grad(mtp_w * s2 / n2, (hh, ee), allow_unused=True)
            cos = F.cosine_similarity(g1.reshape(-1), g2h.reshape(-1), dim=0).item()
            gstats.append((g1.norm().item(), g2h.norm().item(), cos,
                           0.0 if g2e is None else g2e.norm().item()))
        model.model._mtp_cache = None
    for k, (s, n) in acc.items():
        if n:
            out[k] = s / n
    out["val_bpb"] = out["val"] / (math.log(2) * bpt)
    if gstats:
        m = [sum(x[j] for x in gstats) / len(gstats) for j in range(4)]
        out.update(g_main_h=m[0], g_mtp_h=m[1], g_cos=m[2], g_mtp_emb=m[3], g_ratio=m[1] / m[0])
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--n_seqs", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    import json
    c0 = json.load(open(a.results[0]))["config"]
    hold = _val.build_holdout(c0["dataset"], c0["seq_len"], a.n_seqs, DEV)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_val.TOKENIZER)
    tg = hold[:, 1:]
    mk = tg != 0
    bpt = sum(len(tok.decode(r[m].tolist()).encode("utf-8")) for r, m in zip(tg, mk)) / int(mk.sum())
    print(f"[probe] holdout {tuple(hold.shape)}  {int(mk.sum())} scored tokens  {bpt:.3f} bytes/token", flush=True)
    rows = [probe(r, hold, bpt, a.bs) for r in a.results]
    keys = ["val", "val_bpb", "mtp", "mtp_e0", "mtp_eP", "g_main_h", "g_mtp_h", "g_ratio", "g_cos", "g_mtp_emb"]
    print("\n" + f"{'run':44s}" + "".join(f"{k:>11s}" for k in keys))
    for r in rows:
        print(f"{r['tag'][-44:]:44s}" + "".join(f"{r[k]:11.4f}" if k in r else f"{'-':>11s}" for k in keys))
    print("MTP_PROBE_DONE")


if __name__ == "__main__":
    main()
