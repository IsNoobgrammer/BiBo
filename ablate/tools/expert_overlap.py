"""Do a token's top-k experts produce REDUNDANT outputs? (the premise of arxiv 2505.22323's orthogonality loss L_o)

    python -m ablate.tools.expert_overlap --result a_result.json,b_result.json [--seqs 8]
    python -m ablate.tools.expert_overlap --repo fhai50032/bibo-base-1b-6k-s23 --subs step2000,final

Per routed layer, on a fixed holdout: the experts' forward_pre_hook gives (x = residual entering the MoE norm on the
fused path, top-k idx, normalised weights w). hn = rmsnorm(x) * norm_weight; each selected expert's OUTPUT is recomputed
exactly as BiBoFusedExperts does (radial act: silu(g / r) * r ** sigmoid(theta_e), r = rms(g)), UNWEIGHTED. Ranks are
sorted by w per token. Same metrics on k RANDOM non-selected experts for the same token = the baseline for "unrelated".

  cos       mean pairwise cosine between the k outputs (signed) and |cos|; cos^2 = the paper's L_o per pair, normalised
  c12/c16   cosine between the rank-1 and rank-2 / rank-6 outputs
  eff_rank  participation ratio of the k x H output matrix's singular values (k = all distinct, 1 = all parallel)
  cancel    ||sum_r w_r o_r|| / sum_r w_r ||o_r||  (1 = the weighted outputs add constructively, small = they cancel)
  norm@r    mean ||o_r|| per rank;  share@r = mean w_r ||o_r|| / sum -- what rank r actually contributes to the output
"""
from ablate.common import _paths  # noqa: F401
import argparse
import os

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_hub, load_from_result
from ablate.common import validation as _val
from src.modeling.ffn.moe import _NORMSILU_EPS

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def expert_outputs(ex, hn, idx):
    """hn (N,H) fp32, idx (N,k) -> outputs (N,k,H) fp32 of experts idx[n, r] on token n (unweighted)."""
    N, k = idx.shape
    flat_e = idx.reshape(-1)
    flat_t = torch.arange(N, device=hn.device).repeat_interleave(k)
    out = torch.zeros(N * k, hn.shape[1], device=hn.device, dtype=torch.float32)
    for e in flat_e.unique().tolist():
        sel = (flat_e == e).nonzero().flatten()
        x = hn[flat_t[sel]]
        gu = x @ ex.gate_up_proj[e].float().t()
        g, u = gu.chunk(2, -1)
        r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
        a = F.silu(g / r) * r.pow(torch.sigmoid(ex.radial_theta[e].float()))
        out[sel] = (a * u) @ ex.down_proj[e].float().t()
    return out.view(N, k, -1)


def overlap(O, w=None):
    """O (N,k,H), w (N,k) sorted desc or None -> dict of per-token-mean metrics."""
    N, k, _ = O.shape
    nrm = O.norm(dim=-1).clamp_min(1e-12)
    U = O / nrm[..., None]
    C = U @ U.transpose(1, 2)                                   # (N,k,k) cosines
    off = ~torch.eye(k, dtype=torch.bool, device=O.device)
    c = C[:, off]
    s = torch.linalg.svdvals(O)                                 # (N,k)
    pr = s.square().sum(-1).square() / s.pow(4).sum(-1).clamp_min(1e-30)
    d = dict(cos=c.mean().item(), abscos=c.abs().mean().item(), cos2=c.square().mean().item(),
             c12=C[:, 0, 1].mean().item(), c16=C[:, 0, k - 1].mean().item(), eff_rank=pr.mean().item(),
             norm=nrm.mean(0).tolist())
    if w is not None:
        wo = (w[..., None] * O).sum(1).norm(dim=-1)
        d["cancel"] = (wo / (w * nrm).sum(-1).clamp_min(1e-12)).mean().item()
        contrib = w * nrm
        d["share"] = (contrib / contrib.sum(-1, keepdim=True)).mean(0).tolist()
    return d


@torch.no_grad()
def probe(model, hold, seed=0):
    g0 = torch.Generator(device=DEV).manual_seed(seed)
    layers = [(i, l) for i, l in enumerate(model.model.layers)
              if hasattr(getattr(l, "mlp", None), "experts") and l.mlp.gate.top_k < l.mlp.gate.num_routed_experts]
    cap = {i: [] for i, _ in layers}

    def mk(i):
        @torch.autocast("cuda", enabled=False)
        def hook(mod, args):
            cap[i].append((args[0].reshape(-1, args[0].shape[-1]).float(), args[1].reshape(-1, args[1].shape[-1]),
                           args[2].reshape(args[1].reshape(-1, args[1].shape[-1]).shape).float()))
        return hook

    hs = [l.mlp.experts.register_forward_pre_hook(mk(i)) for i, l in layers]
    for b in range(hold.shape[0]):
        with AMP:
            model.model(input_ids=hold[b:b + 1, :-1], use_cache=False)
    for h in hs:
        h.remove()
    rows = {}
    for i, l in layers:
        x = torch.cat([c[0] for c in cap[i]]); idx = torch.cat([c[1] for c in cap[i]]).long(); w = torch.cat([c[2] for c in cap[i]])
        ln, ex, E, k = l.post_attention_layernorm, l.mlp.experts, l.mlp.gate.num_routed_experts, l.mlp.gate.top_k
        hn = F.rms_norm(x, (x.shape[-1],), ln.weight.float(), eps=ln.variance_epsilon)
        w, order = w.sort(-1, descending=True)
        idx = idx.gather(1, order)
        sel = overlap(expert_outputs(ex, hn, idx), w)
        # k random experts NOT in the token's selection
        sc = torch.rand(idx.shape[0], E, device=DEV, generator=g0).scatter_(1, idx, -1.0)
        ridx = sc.topk(k, -1).indices
        rnd = overlap(expert_outputs(ex, hn, ridx))
        rows[i] = (sel, rnd)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    ap.add_argument("--subs", default="final")
    ap.add_argument("--result", default="")
    ap.add_argument("--seqs", type=int, default=8)
    a = ap.parse_args()
    srcs = [("result", p) for p in a.result.split(",") if p] or [("hub", s) for s in a.subs.split(",")]
    hold = None
    for kind, sub in srcs:
        model, cfg = load_from_result(sub) if kind == "result" else load_from_hub(a.repo, sub)
        assert "megakernel" in str(cfg.patches), "hook input is the PRE-norm residual only on the fused (megakernel) path"
        model.eval()
        if hold is None:
            hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, DEV)
        R = probe(model, hold)
        name = cfg.run_tag if kind == "result" else f"{a.repo.split('/')[-1]}/{sub}"
        print(f"\n==== {name}  ({a.seqs} x 1024 holdout tokens)  SELECTED top-k experts | [RANDOM k experts]")
        print(f"{'L':>2} {'cos':>12} {'|cos|':>12} {'cos^2':>12} {'c12':>6} {'c16':>6} {'eff_rank':>12} {'cancel':>6}  "
              f"{'norm@1..k':<34} share@1..k")
        for i, (s, r) in R.items():
            print(f"{i:>2} {s['cos']:+.3f} [{r['cos']:+.3f}] {s['abscos']:.3f} [{r['abscos']:.3f}] {s['cos2']:.3f} [{r['cos2']:.3f}] "
                  f"{s['c12']:+.3f} {s['c16']:+.3f} {s['eff_rank']:5.2f} [{r['eff_rank']:4.2f}] {s['cancel']:6.3f}  "
                  + " ".join(f"{v:.2f}" for v in s["norm"]) + "  " + " ".join(f"{v:.3f}" for v in s["share"]), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
