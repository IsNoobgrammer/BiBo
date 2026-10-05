"""Does our model rely on outlier-driven rescaling (arxiv 2601.22966)? Residual sinks + attention sinks on a checkpoint.

    python -m ablate.tools.outlier_probe --result a_result.json [--seqs 4]
    python -m ablate.tools.outlier_probe --repo fhai50032/bibo-base-1b-6k-s23 --subs final

The paper's signatures: (1) RESIDUAL SINK -- a few residual channels reach |h| in the 1e3s (they report 2,800-6,000 peak)
and the RMSNorm weight at that channel collapses (0.004), so the outlier only shrinks every other channel after the norm;
(2) ATTENTION SINK -- a few keys (usually token 0) take most of the softmax mass with a small value norm.
Per layer, on the residual stream entering the layer (input of input_layernorm):
  max|h|    largest activation;  r = |h_d| / ||h||_2 for the top channel (token-wise max, mean over tokens)
  top dims  channels most often holding a token's max, and the input_layernorm weight there vs the weight median
Per attention layer (inputs captured from attn_xsa, recomputed in fp32): mean softmax mass on key 0 and the mean max
probability per row; value-norm ratio of key 0 vs the rest.
"""
from ablate.common import _paths  # noqa: F401
import argparse

import torch

from ablate.common.report_ckpt import load_from_hub, load_from_result
from ablate.common import validation as _val
import kernels.sm120.attn_xsa as AX
from ablate.tools.attn_precision import post_qk

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def probe(model, ids):
    import src.modeling.attn.base as _ab
    _ab.FUSED_ATTN = True
    model.train()
    res, att, orig = {}, [], AX.attn_xsa

    def pre(i):
        def hook(mod, args, kwargs):
            h = (args[0] if args else kwargs["hidden_states"]).float().reshape(-1, args[0].shape[-1] if args else kwargs["hidden_states"].shape[-1])
            res[i] = h
        return hook

    def spy(q, k, v, **kw):
        att.append((q.detach(), k.detach(), v.detach(), kw))
        return orig(q, k, v, **kw)

    hs = [l.register_forward_pre_hook(pre(i), with_kwargs=True) for i, l in enumerate(model.model.layers)]
    AX.attn_xsa = spy
    try:
        with AMP:
            model.model(input_ids=ids, use_cache=False)
    finally:
        AX.attn_xsa = orig
        for h in hs:
            h.remove()
    print(f"{'L':>2} {'max|h|':>8} {'rms':>6} {'r_top':>6} {'top dims (share)':<34} {'w@top / w median':<22} {'w min':>6}")
    for i, h in res.items():
        w = model.model.layers[i].input_layernorm.weight.float()
        a = h.abs()
        top = a.argmax(-1)
        cnt = torch.bincount(top, minlength=h.shape[-1]).float() / top.numel()
        dims = cnt.topk(3)
        r = (a.max(-1).values / h.norm(dim=-1).clamp_min(1e-12)).mean().item()
        wd = " ".join(f"{w[d].item():.3f}" for d in dims.indices.tolist())
        print(f"{i:>2} {a.max().item():8.1f} {h.pow(2).mean().sqrt().item():6.2f} {r:6.3f} "
              + " ".join(f"{d}({s:.2f})" for d, s in zip(dims.indices.tolist(), dims.values.tolist())).ljust(34)
              + f" {wd} / {w.median().item():.3f}".ljust(22) + f" {w.min().item():6.3f}", flush=True)
    print(f"\n{'L':>2} {'kind':6} {'mass@key0':>9} {'mean max p':>10} {'|v0| / mean|v|':>14}")
    for li, (q, k, v, kw) in enumerate(att):
        qp, kp = post_qk(q, k, kw)
        S, G = q.shape[2], q.shape[1] // k.shape[1]
        i = torch.arange(S, device=DEV)
        m = i[None, :] <= i[:, None]
        if kw["window"] is not None:
            m = m & (i[:, None] - i[None, :] < kw["window"])
        lg = ((qp @ kp.repeat_interleave(G, 1).transpose(-1, -2)) * kw["scale"]).masked_fill(~m, float("-inf"))
        p = torch.softmax(lg.float(), -1)
        mass0 = p[..., 64:, 0].mean().item() if kw["window"] is None else float("nan")   # rows that can see key 0
        vn = v.float().norm(dim=-1)
        print(f"{li:>2} {'global' if kw['window'] is None else 'w' + str(kw['window']):6} {mass0:9.3f} "
              f"{p.amax(-1).mean().item():10.3f} {(vn[..., 0] / vn.mean(-1)).mean().item():14.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    ap.add_argument("--subs", default="final")
    ap.add_argument("--result", default="")
    ap.add_argument("--seqs", type=int, default=4)
    a = ap.parse_args()
    srcs = [("result", p) for p in a.result.split(",") if p] or [("hub", s) for s in a.subs.split(",")]
    for kind, sub in srcs:
        model, cfg = load_from_result(sub) if kind == "result" else load_from_hub(a.repo, sub)
        hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, DEV)
        print(f"\n==== {cfg.run_tag if kind == 'result' else a.repo + '/' + sub}")
        probe(model, hold[:1, :-1])
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
