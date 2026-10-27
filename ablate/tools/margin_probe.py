"""Softmax Collapse exposure on a real checkpoint (arXiv 2610.31642): how many next-token predictions sit past the
precision cliffs, how often the pre-fix tkf ce_factored got the label's gradient sign wrong, and how far the MoE
router's sigmoid logits go (torch.sigmoid's backward y(1 - y) is exactly 0 once y rounds to 1, logit > ~16.6).

    python -m ablate.tools.margin_probe --repo fhai50032/bibo-base-1b-6k-s23 --subs step1000,final --seqs 64

Per token: logits L = bf16(h W^T) (what every CE kernel sees), margin m = L[label] - logsumexp(L without label).
  true     label grad -e^(lso - lse)
  torch    exp(L[label] - lse) - 1 in fp32 (PyTorch, chunked CE pre-fix): 0 once m > ~16.6
  old_fac  bf16(e^L[label]) * e^-lse - 1 (ce_factored pre-fix): bf16 rounding noise once m > ~6
"""
from ablate.common import _paths  # noqa: F401
import argparse

import torch

from ablate.common.report_ckpt import load_from_hub
from ablate.common import validation as _val

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
EDGES = [0, 2, 4, 5.5, 7, 10, 16.6, 25, 1e9]


def label_stats(h, W, y):
    out = []
    for i in range(0, h.shape[0], 4096):
        L = (h[i:i + 4096].to(torch.bfloat16) @ W.t()).float()
        lab = y[i:i + 4096]
        r = torch.arange(L.shape[0], device=DEV)
        Lt = L[r, lab]
        lse = torch.logsumexp(L, 1)
        o = L.clone()
        o[r, lab] = -float("inf")
        lso = torch.logsumexp(o, 1)
        true = -torch.exp(lso.double() - lse.double())
        torch_g = torch.exp(Lt - lse) - 1
        old = torch.exp(Lt).to(torch.bfloat16).float() * torch.exp(-lse) - 1
        out.append(torch.stack([Lt - lso, true.float(), torch_g, old], 1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="fhai50032/bibo-base-1b-6k-s23")
    ap.add_argument("--subs", default="final")
    ap.add_argument("--seqs", type=int, default=64)
    a = ap.parse_args()
    hold = None
    for sub in a.subs.split(","):
        model, cfg = load_from_hub(a.repo, sub)
        model.eval()
        if hold is None:
            hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, DEV)
        rl = []
        hooks = []
        for mod in model.modules():
            if type(mod).__name__ == "BiBoMoERouter":
                orig = mod.router_logits

                def wrap(x, _o=orig):
                    z = _o(x)
                    rl.append(z.detach().float().flatten()[::7])      # subsample: memory
                    return z
                mod.router_logits = wrap
                hooks.append(mod)
        stats = []
        W = model.lm_head.weight.to(torch.bfloat16)
        with torch.no_grad(), AMP:
            for b in range(0, hold.shape[0], 4):
                ids = hold[b:b + 4]
                h = model.model(input_ids=ids[:, :-1], use_cache=False)
                h = h[0] if isinstance(h, (tuple, list)) else getattr(h, "last_hidden_state", h)
                y = ids[:, 1:].reshape(-1)
                keep = y != 0                                                # --pad_id 0
                stats.append(label_stats(h.reshape(-1, h.shape[-1])[keep], W, y[keep]))
        s = torch.cat(stats)
        m, true, tg, old = s.unbind(1)
        n = m.numel()
        print(f"\n==== {a.repo.split('/')[-1]}/{sub}: {n} tokens")
        print(f"{'margin':>12} {'tokens':>8} {'share':>7} {'torch g=0':>10} {'old sign wrong':>15} {'old |err|/|true|':>17}")
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sel = (m >= lo) & (m < hi)
            k = int(sel.sum())
            if not k:
                continue
            z = (tg[sel] == 0).float().mean().item()
            wrong = (torch.sign(old[sel]) != torch.sign(true[sel])).float().mean().item()
            err = ((old[sel] - true[sel]).abs() / true[sel].abs().clamp_min(1e-30)).median().item()
            print(f"{lo:>5}-{hi:<6} {k:>8} {k / n:>7.2%} {z:>10.2%} {wrong:>15.2%} {err:>17.2e}")
        print(f"past 5.5: {(m > 5.5).float().mean():.2%}  past 7: {(m > 7).float().mean():.2%}  "
              f"past 16.6: {(m > 16.6).float().mean():.4%}  | old kernel sign wrong overall: "
              f"{(torch.sign(old) != torch.sign(true)).float().mean():.2%}")
        if rl:
            r = torch.cat(rl)
            q = torch.quantile(r[torch.randperm(r.numel(), device=r.device)[:1_000_000]],
                               torch.tensor([0.5, 0.99, 0.999], device=r.device))
            print(f"router logits ({len(hooks)} routers): max {r.max():.2f}  p50 {q[0]:.2f}  p99 {q[1]:.2f}  "
                  f"p99.9 {q[2]:.2f}  >10: {(r > 10).float().mean():.4%}  >16.6 (sigmoid grad = 0): "
                  f"{(r > 16.6).float().mean():.4%}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
