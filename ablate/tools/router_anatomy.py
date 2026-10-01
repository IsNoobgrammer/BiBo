"""Router logit scale + per-rank expert weight across --hf_repo checkpoints (is z-loss needed? do the low-rank
active experts actually contribute?).

    python -m ablate.tools.router_anatomy --repo fhai50032/bibo-base-1b-6k-s23 --subs step1000,step3000,final [--seqs 16]

Per routed layer, on a fixed holdout, recomputed from the experts' forward_pre_hook (x = the residual entering
the MoE norm; patches._mk_mlp replays the hook with it): logits = rmsnorm(x) @ Wr^T, s = sigmoid(logits),
selection on s + bias (aux-free balancing), weights w = s_sel / sum(s_sel) (norm_topk_prob "sum").
  logit    mean / std / p99 / max over ALL experts, and top-1 logit mean
  sat%     selected experts with s > 0.99: d s / d logit = s(1-s) < 0.01, i.e. almost no router gradient
  w@r      mean normalised weight at rank r = 1..k (sorted per token), and the share of tokens whose
           k-th weight is < 0.02 / < 0.05 -- how much the last active experts contribute
"""
from ablate.common import _paths  # noqa: F401
import argparse

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_hub, load_from_result
from ablate.common import validation as _val

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def probe(model, hold):
    layers = [(i, l) for i, l in enumerate(model.model.layers)
              if hasattr(getattr(l, "mlp", None), "experts") and l.mlp.gate.top_k < l.mlp.gate.num_routed_experts]
    acc = {i: [] for i, _ in layers}

    def mk(i, layer):
        g, ln = layer.mlp.gate, layer.post_attention_layernorm
        @torch.autocast("cuda", enabled=False)     # the forward runs under bf16 autocast; the probe must not
        def hook(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            z = F.rms_norm(x, (x.shape[-1],), ln.weight.float(), eps=ln.variance_epsilon) @ g.gate_proj.weight.float().t()
            s = torch.sigmoid(z)
            sel = (s + g.bias.float()) if g.bias is not None else s
            idx = sel.topk(g.top_k, -1).indices
            ss = s.gather(-1, idx)
            w = (ss / ss.sum(-1, keepdim=True)).sort(-1, descending=True).values
            acc[i].append(dict(z=z, ztop=z.max(-1).values, sat=(ss > 0.99).float().mean(), w=w))
        return hook

    hs = [l.mlp.experts.register_forward_pre_hook(mk(i, l)) for i, l in layers]
    for b in range(hold.shape[0]):
        with AMP:
            model.model(input_ids=hold[b:b + 1, :-1], use_cache=False)
    for h in hs:
        h.remove()
    rows = {}
    for i, a in acc.items():
        z = torch.cat([d["z"] for d in a]).flatten()
        w = torch.cat([d["w"] for d in a])
        rows[i] = dict(mean=z.mean().item(), std=z.std().item(),
                       p99=torch.quantile(z[torch.randperm(z.numel(), device=z.device)[:2_000_000]], 0.99).item(),
                       max=z.max().item(), top1=torch.cat([d["ztop"] for d in a]).mean().item(),
                       sat=100 * torch.stack([d["sat"] for d in a]).mean().item(),
                       w=w.mean(0).tolist(), lo2=100 * (w[:, -1] < 0.02).float().mean().item(),
                       lo5=100 * (w[:, -1] < 0.05).float().mean().item())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    ap.add_argument("--subs", default="step1000,step3000,final")
    ap.add_argument("--result", default="")          # OR local ..._result.json checkpoints, comma list
    ap.add_argument("--seqs", type=int, default=16)
    a = ap.parse_args()
    hold = None
    srcs = [("result", p) for p in a.result.split(",") if p] or [("hub", s) for s in a.subs.split(",")]
    for kind, sub in srcs:
        model, cfg = load_from_result(sub) if kind == "result" else load_from_hub(a.repo, sub)
        model.eval()
        if hold is None:
            hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, DEV)   # the run's own held-out (last) shard
        R = probe(model, hold)
        k = len(next(iter(R.values()))["w"])
        name = f"{cfg.run_tag} E={cfg.experts}" if kind == "result" else f"{a.repo}/{sub}"
        print(f"\n==== {name}  ({a.seqs} x 1024 holdout tokens; top-{k})")
        print(f"{'L':>3} {'logit mean':>10} {'std':>6} {'p99':>6} {'max':>6} {'top1':>6} {'sat%':>6}  "
              + " ".join(f"w@{r + 1:<3}" for r in range(k)) + f" {'k<.02%':>7} {'k<.05%':>7}")
        for i, r in R.items():
            print(f"{i:>3} {r['mean']:10.2f} {r['std']:6.2f} {r['p99']:6.2f} {r['max']:6.1f} {r['top1']:6.2f} {r['sat']:6.2f}  "
                  + " ".join(f"{x:5.3f}" for x in r["w"]) + f" {r['lo2']:7.2f} {r['lo5']:7.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
