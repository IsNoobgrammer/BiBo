"""Final-checkpoint probe for the MTP arms: a big-holdout val and "does the head lean on Emb(t+1)?".

    python -m ablate.tools.mtp_probe <result.json> [<result.json> ...] [--n_seqs 256] [--push_wandb]

--push_wandb writes the numbers into each run's W&B SUMMARY under probe/* (t1 = main-head top-1 on the
next token, the learning-signal read; t2 = the MTP head's top-1 on the token after next and its greedy
draft acceptance, the drafter read). The run is looked up in mtp-ablations first (MTP runs were moved
there after training), then in the project its result.json recorded.

For every checkpoint, on the SAME frozen holdout (n_seqs x 1024 tokens, pad id 0 masked):
  val      main-head CE (+ bpb), token-weighted -- the 2-seq training val, with 128x the tokens
  mtp      the MTP head's CE vs t+2
  mtp_e0   same with Emb(t+1) replaced by zeros            (with-embedding heads only)
  mtp_eP   same with Emb(t+1) replaced by a shuffled token (with-embedding heads only)
  spec     greedy self-speculative decoding with the MTP head as a 1-token drafter: the main head's
           greedy token g1 at i, the head drafts d2 for i+2 (a with-emb head is fed Emb(g1), its own
           prediction, as at inference), ACCEPTED if d2 == the main head's greedy token at i+1.
           Counted only where the text equals the greedy choice (g1 == t_{i+1}), so teacher-forced
           text == self-generated text and the rate is exact. Also d2's top-1 vs the true t+2.
  grads    at h = the trunk's final hidden state: |dCE1/dh|, |d(w*CE2)/dh|, cos(dCE1, dCE2), and
           |d(w*CE2)/dEmb(t+1)|  -- how much of the MTP gradient reaches the trunk vs the embedding

The MTP forward needs the trunk's RoPE tensors and block archive, which BiBoModel only hands over in
training mode; only the top module's flag is flipped (submodules stay in eval, so the MoE balancer
never updates). Checkpoints are final-step states, so the gradient numbers describe the END of
training, not its average.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import json
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


def push(res, row, n_seqs, n_tok):
    import wandb
    r = json.load(open(res))
    rid = r.get("wandb_id")
    if not rid:
        print(f"[probe] no wandb_id in {res}, not pushed"); return
    api = wandb.Api(timeout=60)
    run = None
    for proj in ("mtp-ablations", r.get("wandb_project")):
        try:
            run = api.run(f"{api.default_entity}/{proj}/{rid}"); break
        except Exception:
            continue
    if run is None:
        print(f"[probe] run {rid} not found, not pushed"); return
    m = {"val": "val_bigholdout", "val_bpb": "val_bpb_bigholdout", "mtp": "mtp_ce", "mtp_e0": "mtp_ce_emb_zeroed",
         "main_top1": "t1_top1", "d2_top1": "t2_top1", "spec_acc": "t2_accept", "spec_elig": "t2_eligible",
         "g_cos": "grad_cos_main_mtp", "g_ratio": "grad_ratio_mtp_main"}
    upd = {f"probe/{v}": row[k] for k, v in m.items() if k in row}
    upd.update({"probe/n_seqs": n_seqs, "probe/n_tokens": n_tok})
    run.summary.update(upd)
    print(f"[probe] pushed {len(upd)} keys to {run.project}/{rid} ({run.name})", flush=True)


def probe(res, hold, bpt, bs, mtp_w=0.3):
    model, c = load_from_result(res)
    W = model.lm_head.weight
    pad = 0
    out = {"tag": getattr(c, "run_tag", res)}
    has_mtp = getattr(model, "mtp", None) is not None
    use_emb = has_mtp and model.mtp.use_emb
    acc = {k: [0.0, 0] for k in ("val", "mtp", "mtp_e0", "mtp_eP")}
    gstats = []
    spec = [0, 0, 0, 0, 0]          # accepted, eligible, all positions, d2 == t+2, main top1 == t+1
    g = torch.Generator(device=DEV).manual_seed(0)
    for i in range(0, hold.shape[0], bs):
        ids = hold[i:i + bs]
        inp, t1 = ids[:, :-1], ids[:, 1:]
        t2 = torch.cat([ids[:, 2:], ids.new_full((ids.shape[0], 1), pad)], 1)
        with torch.no_grad(), AMP:
            model.model.training = True                     # hand over RoPE + archive (top flag only)
            h = model.model(input_ids=inp, use_cache=False).last_hidden_state
            model.model.training = False
            pe, br = model.model._mtp_cache if has_mtp else (None, None)   # no head -> no handover
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
                # ---- speculative-decoding acceptance (greedy, 1 draft token)
                Wb = W.to(h.dtype)
                g1 = (h @ Wb.t()).argmax(-1)                                 # main greedy, (b, S)
                xd = model.mtp(h, model.model.embed_tokens(g1) if use_emb else e, pe, br)
                d2 = (xd @ Wb.t()).argmax(-1)                                # draft for i+2
                ok = (g1[:, :-1] == t1[:, :-1]) & (t1[:, :-1] != pad) & (t2[:, :-1] != pad)
                spec[0] += int(((d2[:, :-1] == g1[:, 1:]) & ok).sum())
                spec[1] += int(ok.sum())
                valid2 = (t2[:, :-1] != pad)
                spec[2] += int(valid2.sum())
                spec[3] += int(((d2[:, :-1] == t2[:, :-1]) & valid2).sum())
                spec[4] += int(((g1 == t1) & (t1 != pad)).sum())
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
    if spec[1]:
        out.update(spec_acc=spec[0] / spec[1], spec_elig=spec[1] / spec[2], d2_top1=spec[3] / spec[2],
                   main_top1=spec[4] / spec[2])
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
    ap.add_argument("--push_wandb", action="store_true")
    a = ap.parse_args()
    c0 = json.load(open(a.results[0]))["config"]
    hold = _val.build_holdout(c0["dataset"], c0["seq_len"], a.n_seqs, DEV)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_val.TOKENIZER)
    tg = hold[:, 1:]
    mk = tg != 0
    bpt = sum(len(tok.decode(r[m].tolist()).encode("utf-8")) for r, m in zip(tg, mk)) / int(mk.sum())
    print(f"[probe] holdout {tuple(hold.shape)}  {int(mk.sum())} scored tokens  {bpt:.3f} bytes/token", flush=True)
    rows = []
    for r in a.results:
        rows.append(probe(r, hold, bpt, a.bs))
        if a.push_wandb:
            push(r, rows[-1], a.n_seqs, int(mk.sum()))
    keys = ["val", "val_bpb", "mtp", "mtp_e0", "mtp_eP", "g_main_h", "g_mtp_h", "g_ratio", "g_cos", "g_mtp_emb",
            "main_top1", "d2_top1", "spec_elig", "spec_acc"]
    print("\n" + f"{'run':44s}" + "".join(f"{k:>11s}" for k in keys))
    for r in rows:
        print(f"{r['tag'][-44:]:44s}" + "".join(f"{r[k]:11.4f}" if k in r else f"{'-':>11s}" for k in keys))
    print("MTP_PROBE_DONE")


if __name__ == "__main__":
    main()
