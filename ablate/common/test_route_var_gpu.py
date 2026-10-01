"""GPU check + calibration for the routing-variance loss (patches.ROUTE_VAR, train.py --route_var_coef).

    python -m ablate.common.test_route_var_gpu [--repo fhai50032/bibo-base-1b-6k-s23 --sub final]

On a REAL checkpoint and real held-out text (fused megakernel path, as in training):
  1. L_v is finite and collected once per routed layer
  2. its gradient reaches ONLY the router weights (gate_proj) -- every other parameter gets none from it
  3. sign: one small step along -grad(L_v) INCREASES the routing variance
  4. calibration: L_v's value and its router-grad norm vs the CE's router-grad norm on the same batch -> gamma
"""
from ablate.common import _paths  # noqa: F401
import argparse

import torch

from ablate.common.report_ckpt import load_from_hub
from ablate.common import patches as P
from ablate.common import validation as _val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="fhai50032/bibo-base-1b-6k-s23")
    ap.add_argument("--sub", default="final")
    ap.add_argument("--seqs", type=int, default=4)
    a = ap.parse_args()
    model, cfg = load_from_hub(a.repo, a.sub)
    model.train()                                       # training-mode forward, as in train.py
    hold = _val.build_holdout(cfg.dataset, 1024, a.seqs, "cuda")
    ids = hold[:, :-1]
    routers = {n: p for n, p in model.named_parameters() if n.endswith("mlp.gate.gate_proj.weight")}
    amp = torch.autocast("cuda", dtype=torch.bfloat16)

    def fwd():
        with amp:
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state
        return h

    # CE router gradient (reference scale)
    P.ROUTE_VAR["coef"], P.ROUTE_VAR["acc"] = 0.0, []
    model.zero_grad(set_to_none=True)
    h = fwd()
    ce = torch.nn.functional.cross_entropy((h.float() @ model.lm_head.weight.float().t()).reshape(-1, model.lm_head.weight.shape[0]),
                                           hold[:, 1:].reshape(-1))
    ce.backward()
    g_ce = {n: p.grad.float().norm().item() for n, p in routers.items()}

    # L_v alone
    P.ROUTE_VAR["coef"], P.ROUTE_VAR["acc"] = 1.0, []
    model.zero_grad(set_to_none=True)
    fwd()
    acc = P.ROUTE_VAR["acc"]
    assert len(acc) == len(routers) - 1 or len(acc) == len(routers), f"one L_v per routed layer, got {len(acc)} for {len(routers)} routers"
    lv = torch.stack(acc).sum()
    assert torch.isfinite(lv), "L_v not finite"
    lv.backward()
    leaked = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0 and n not in routers]
    assert not leaked, f"L_v gradient leaked into non-router params: {leaked[:5]}"
    g_lv = {n: (p.grad.float().norm().item() if p.grad is not None else 0.0) for n, p in routers.items()}
    var0 = -lv.item()

    # sign: one step along -grad L_v must increase the variance
    with torch.no_grad():
        for n, p in routers.items():
            if p.grad is not None:
                p -= 1e-2 * p.grad / (p.grad.norm() + 1e-12) * p.norm()
    P.ROUTE_VAR["acc"] = []
    with torch.no_grad():
        P.ROUTE_VAR["coef"] = 1.0
        torch.set_grad_enabled(True)
        fwd()
        var1 = -torch.stack(P.ROUTE_VAR["acc"]).sum().item()
    P.ROUTE_VAR["coef"], P.ROUTE_VAR["acc"] = 0.0, []
    print(f"{a.repo}/{a.sub}: sum over {len(acc)} routed layers of Var(E*p): {var0:.4f} -> {var1:.4f} after a 1% step along -grad")
    assert var1 > var0, "a step along -grad(L_v) did not increase the routing variance: sign is wrong"
    print(f"{'router':40s} {'|g_CE|':>10s} {'|g_Lv|':>10s} {'ratio Lv/CE':>12s}")
    for n in routers:
        if g_lv[n] > 0:
            print(f"{n:40s} {g_ce[n]:10.3e} {g_lv[n]:10.3e} {g_lv[n] / max(g_ce[n], 1e-12):12.2f}")
    r = sorted(g_lv[n] / max(g_ce[n], 1e-12) for n in routers if g_lv[n] > 0)
    med = r[len(r) // 2]
    print(f"median |g_Lv| / |g_CE| = {med:.2f} -> gamma making L_v's router grad ~10% / ~50% of CE's: "
          f"{0.1 / med:.3g} / {0.5 / med:.3g}")
    print("ROUTE_VAR_GPU PASS")


if __name__ == "__main__":
    main()
