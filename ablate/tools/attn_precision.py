"""Attention backward precision on REAL activations: are our logits / mean keys in the regime where the bf16 dS
leak (GProj arxiv 2609.34272, KohakuFA) corrupts dQ?

    python -m ablate.tools.attn_precision --repo fhai50032/bibo-base-1b-6k-s23 --subs step1000,final [--seqs 2]

Every attention layer's attn_xsa call is captured once (q, k, v projections + qk-norm weights + RoPE tables + XSA
alpha), then replayed: our kernel vs attn_xsa_reference in fp64, same random upstream gradient. Per layer:
  maxlog   largest logit (after qk-norm, RoPE, scale) over the causal / window mask
  c        mean-key ratio ||mean_j k_j|| / rms_j ||k_j - mean||, per kv head (max over heads), k after norm + RoPE.
           The leak in dQ is ~ proportional to it (synthetic sweep: c 1 -> 0.4%, 4 -> 1.2%, 16 -> 4.7% dQ error)
  dQ dK dV relative error of our kernel vs fp64 (both returned in bf16; fp32 eager reaches ~1e-4 here)
"""
from ablate.common import _paths  # noqa: F401
import argparse

import torch

from ablate.common.report_ckpt import load_from_hub
from ablate.common import validation as _val
import kernels.sm120.attn_xsa as AX

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()


def post_qk(q, k, kw):
    """q, k after the kernel's own prep (norm * w * scale, then RoPE), fp64."""
    q, k = q.double(), k.double()
    if kw["q_norm_w"] is not None:
        eps = kw["eps"]
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * kw["q_norm_w"].double()
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * kw["k_norm_w"].double()
    q, k = q * kw["q_scale"], k * kw["k_scale"]
    if kw["cos"] is not None:
        c, s = kw["cos"].double(), kw["sin"].double()
        c, s = (c[:, None], s[:, None]) if c.dim() == 3 else (c, s)
        q, k = q * c + AX._rotate_half(q) * s, k * c + AX._rotate_half(k) * s
    return q, k


def replay(args, kw):
    q, k, v = args
    leaves = lambda: [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    g = torch.Generator(device=DEV).manual_seed(0)
    dz = torch.randn(q.shape, device=DEV, generator=g, dtype=torch.float32)
    out = {}
    for name in ("ours", "fp64"):
        lq, lk, lv = leaves()
        if name == "ours":
            z = AX.attn_xsa(lq, lk, lv, **kw)
        else:
            z = AX.attn_xsa_reference(lq, lk, lv, dtype=torch.float64, **kw)
        z.backward(dz.to(z.dtype))
        out[name] = (lq.grad, lk.grad, lv.grad)
    with torch.no_grad():
        qp, kp = post_qk(q, k, kw)
        S, G = q.shape[2], q.shape[1] // k.shape[1]
        i = torch.arange(S, device=DEV)
        m = i[None, :] <= i[:, None]
        if kw["window"] is not None:
            m = m & (i[:, None] - i[None, :] < kw["window"])
        lg = (qp @ kp.repeat_interleave(G, 1).transpose(-1, -2)) * kw["scale"]
        maxlog = lg.masked_fill(~m, float("-inf")).amax().item()
        mean = kp.mean(2, keepdim=True)                                    # (B, Hkv, 1, D)
        spread = (kp - mean).pow(2).sum(-1).mean(2).sqrt()                 # (B, Hkv)
        c = (mean.squeeze(2).norm(dim=-1) / spread).amax().item()
    (dq, dk, dv), (rq, rk, rv) = out["ours"], out["fp64"]
    return dict(maxlog=maxlog, c=c, dq=rel(dq, rq), dk=rel(dk, rk), dv=rel(dv, rv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="fhai50032/bibo-base-1b-6k-s23")
    ap.add_argument("--subs", default="final")
    ap.add_argument("--seqs", type=int, default=2)
    a = ap.parse_args()
    hold = None
    for sub in a.subs.split(","):
        model, cfg = load_from_hub(a.repo, sub)
        model.train()                                   # the fused attention path is the training path
        if hold is None:
            hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, DEV)
        cap, orig = [], AX.attn_xsa

        def spy(q, k, v, **kw):
            cap.append(((q.detach().clone(), k.detach().clone(), v.detach().clone()),
                        {kk: (vv.detach().clone() if torch.is_tensor(vv) else vv) for kk, vv in kw.items()}))
            return orig(q, k, v, **kw)
        AX.attn_xsa = spy
        try:
            with torch.no_grad(), AMP:
                model.model(input_ids=hold[:1, :-1], use_cache=False)
        finally:
            AX.attn_xsa = orig
        print(f"\n==== {a.repo.split('/')[-1]}/{sub}: {len(cap)} attention layers, S={hold.shape[1] - 1}")
        print(f"{'L':>2} {'kind':6} {'maxlog':>7} {'c':>6} {'dQ err':>8} {'dK err':>8} {'dV err':>8}")
        for li, (args, kw) in enumerate(cap):
            r = replay(args, kw)
            kind = "global" if kw["window"] is None else f"w{kw['window']}"
            print(f"{li:>2} {kind:6} {r['maxlog']:7.1f} {r['c']:6.2f} {r['dq']:8.2e} {r['dk']:8.2e} {r['dv']:8.2e}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
