"""Deep four-way interp: silu vs normsilu vs radial vs silu-a, per layer and per expert.

Runs the REAL patched forward (the same Triton MoE path the models trained under), because the
eager expert path only understands ("silu","relu2","normsilu")[e % 3] and would silently compute
the wrong activation for radial / silu-a. A selfcheck recomputes one layer's expert output from the
weights and aborts if it disagrees, so "it ran" can never be mistaken for "it ran correctly".

THE CHAIN THIS MEASURES, per expert e:
    g = x @ Wg^T            gate pre-activation      (raw)
    u = x @ Wu^T            up pre-activation        (raw, NEVER normalized by any arm)
    r = rms(g over I)       per-token gate scale
    z = SiLU input          silu: g | normsilu,radial: g/r | silu-a: alpha_e * g
    gain                    radial: r^p (p=sigmoid(theta_e)); otherwise 1
    a = gain * SiLU(z)      activation
    prod = a * u            <- WHAT DOWN_PROJ ACTUALLY CONSUMES; the quantity that decides
                               dynamic range / fp8 headroom. Neither branch alone tells you this.
    out = prod @ Wd^T       expert contribution before the router weight

COUNTERFACTUAL BASELINES (what p and alpha buy over doing nothing):
    radial  p -> 0 is exactly normsilu (gain 1) and p -> 1 is g*sigma(g/r) (gain r). We log the
            realised gain r^p against both ends, so "how far off normsilu did p travel" is a number.
    silu-a  alpha = 1 is exactly plain silu, which is the init. Its drift from 1.0 is the effect.

Quantiles come from a per-expert reservoir (exact on the sample, capped so memory stays flat).
"""
from . import _paths  # noqa: F401
import argparse
import json

import torch
import torch.nn.functional as F

ARMS = {  # name -> (act code name, act_scale_learnable, act_scale_init)
    "silu":     ("silu",     False, 1.0),
    "normsilu": ("normsilu", False, 1.0),
    "radial":   ("radial",   True,  0.0),
    "silu-a":   ("silu",     True,  1.0),
}


class Stat:
    """Streaming moments + a capped reservoir for exact-on-sample quantiles."""

    def __init__(self, cap=6000):
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.mn, self.mx = float("inf"), float("-inf")
        self.neg = 0
        self.cap, self.res = cap, []

    def add(self, x):
        x = x.detach().float().flatten()
        if x.numel() == 0:
            return
        self.n += x.numel()
        self.s1 += x.sum().item()
        self.s2 += x.pow(2).sum().item()
        self.s3 += x.pow(3).sum().item()
        self.s4 += x.pow(4).sum().item()
        self.mn = min(self.mn, x.min().item())
        self.mx = max(self.mx, x.max().item())
        self.neg += int((x < 0).sum())
        take = min(x.numel(), 256)
        idx = torch.randint(0, x.numel(), (take,), device=x.device)
        self.res.append(x[idx].cpu())
        if len(self.res) > 64:                       # keep memory flat: collapse + subsample
            r = torch.cat(self.res)
            self.res = [r[torch.randperm(r.numel())[:self.cap]]]

    def out(self):
        if not self.n:
            return dict(n=0, **{k: float("nan") for k in
                                ("mean", "std", "rms", "min", "max", "skew", "kurtosis",
                                 "absmax_over_rms", "frac_neg", "p1", "p25", "p50", "p75", "p99",
                                 "iqr", "p99_over_p50")})
        m = self.s1 / self.n
        var = max(self.s2 / self.n - m * m, 0.0)
        sd, rms = var ** 0.5, (self.s2 / self.n) ** 0.5
        skew = kurt = float("nan")
        if sd > 1e-12:
            skew = (self.s3 / self.n - 3 * m * var - m ** 3) / sd ** 3
            kurt = (self.s4 / self.n - 4 * m * self.s3 / self.n
                    + 6 * m * m * self.s2 / self.n - 3 * m ** 4) / var ** 2
        r = torch.cat(self.res)
        q = torch.quantile(r, torch.tensor([0.01, 0.25, 0.50, 0.75, 0.99])).tolist()
        return dict(n=self.n, mean=m, std=sd, rms=rms, min=self.mn, max=self.mx, skew=skew,
                    kurtosis=kurt,
                    absmax_over_rms=(max(abs(self.mn), abs(self.mx)) / rms if rms > 1e-12 else float("nan")),
                    frac_neg=self.neg / self.n,
                    p1=q[0], p25=q[1], p50=q[2], p75=q[3], p99=q[4], iqr=q[3] - q[1],
                    p99_over_p50=(q[4] / q[2] if abs(q[2]) > 1e-9 else float("nan")))


def _nm(xs):
    v = [x for x in xs if x == x]
    return sum(v) / len(v) if v else float("nan")


def build(arm, ckpt, dev, polyglu_mult, top_k):
    """Mirror train.py's setup exactly, then load. add_situ_params MUST precede load_state_dict."""
    from ablate.common import patches as patchmod
    from ablate.common.models import build_arm
    from ablate.common.train import ACT_CODES
    act, learn, init = ARMS[arm]
    patchmod.ROUTER_GATE, patchmod.ROUTER_NORM, patchmod.ROUTER_SCALE = "sigmoid", "sum", 1.0
    patchmod.ACT_CYCLE = [ACT_CODES[act]]
    model, cfg = build_arm("bibo_min", device=dev, dtype=torch.float32, attn_impl="sdpa",
                           polyglu_mult=polyglu_mult, top_k=top_k)
    if learn:
        patchmod.add_situ_params(model, init=init)
    patchmod.apply(["liger_norm", "liger_rope", "moe", "router_gate"])
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg, missing, unexpected


def act_of(arm, g, alpha):
    """(z, gain, act) for one expert's raw gate. alpha is that expert's scalar (or None)."""
    r = g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    if arm == "silu":
        z, gain = g, torch.ones_like(r)
    elif arm == "normsilu":
        z, gain = g / r, torch.ones_like(r)
    elif arm == "radial":
        p = torch.sigmoid(alpha)
        z, gain = g / r, r.pow(p)
    elif arm == "silu-a":
        z, gain = alpha * g, torch.ones_like(r)
    else:
        raise ValueError(arm)
    return z, gain, gain * F.silu(z), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--polyglu_mult", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--dataset", default="/home/marimo/work/data/bip2")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    from ablate.common.data import token_batches

    model, _cfg, missing, unexpected = build(a.arm, a.ckpt, dev, a.polyglu_mult, a.top_k)
    print(f"[{a.arm}] loaded missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    experts = [(n, m) for n, m in model.named_modules() if hasattr(m, "gate_up_proj")]
    E, twoI, D = experts[0][1].gate_up_proj.shape
    I = twoI // 2
    # per-expert act scalar (radial theta / silu-a alpha), else None
    def scal(mod):
        for nm_ in ("situ_alpha", "alpha", "act_alpha"):
            v = getattr(mod, nm_, None)
            if v is not None:
                return v.detach().float().reshape(-1)
        return None
    print(f"[{a.arm}] {len(experts)} MoE layers E={E} I={I} d={D} "
          f"scalars={'yes' if scal(experts[0][1]) is not None else 'no'}", flush=True)

    KEYS = ("gate", "up", "z", "act", "prod", "outv", "gain", "tokrms")
    ST = [{k: [Stat() for _ in range(E)] for k in KEYS} | {"load": [0] * E,
          "lin": [0] * E, "dead": [0] * E, "band": [0] * E, "tot": [0] * E,
          "wtop1": Stat(), "went": Stat(), "wmin": Stat()} for _ in experts]

    cap = {}
    def mk(i):
        def hook(_m, args):
            cap[i] = (args[0].detach(), args[1].detach(), args[2].detach())
        return hook
    handles = [m.register_forward_pre_hook(mk(i)) for i, (_n, m) in enumerate(experts)]
    gen = token_batches(a.batch, a.seq_len, dev, dataset=a.dataset)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)

    # ---------------- SELFCHECK: does the model compute what act_of() says? ----------------
    got = {}
    h = experts[0][1].register_forward_hook(lambda _m, _i, o: got.__setitem__("y", o.detach()))
    with torch.no_grad(), amp:
        model.model(input_ids=next(gen)[:, :-1], use_cache=False)
    h.remove()
    mod0 = experts[0][1]
    al0 = scal(mod0)
    _h, _idx, _w = cap[0]
    _h = _h.reshape(-1, _h.shape[-1]).float()
    _idx = _idx.reshape(_h.shape[0], -1)
    _w = _w.reshape(_h.shape[0], -1).float()
    ref = torch.zeros_like(_h)
    for e in range(E):
        msk = (_idx == e)
        sel = msk.any(-1)
        if not bool(sel.any()):
            continue
        g_, u_ = (_h[sel] @ mod0.gate_up_proj[e].float().T).chunk(2, dim=-1)
        _z, _gn, a_, _r = act_of(a.arm, g_, (al0[e] if al0 is not None else None))
        ref[sel] += ((a_ * u_) @ mod0.down_proj[e].float().T) * (_w[sel] * msk[sel].float()).sum(-1, keepdim=True)
    y = got["y"].reshape(-1, got["y"].shape[-1]).float()
    err = (y - ref).abs().max().item() / max(y.abs().max().item(), 1e-9)
    print(f"[{a.arm}] SELFCHECK layer0 rel err = {err:.3e}", flush=True)
    if err > 5e-2:
        raise SystemExit(f"SELFCHECK FAILED ({err:.3e}) for arm={a.arm}: the forward is not "
                         f"computing what act_of() assumes. Stats would be meaningless.")

    # ---------------- main pass ----------------
    with torch.no_grad():
        for b in range(a.batches):
            ids = next(gen)
            with amp:
                model.model(input_ids=ids[:, :-1], use_cache=False)
            for i, (_n, mod) in enumerate(experts):
                hh, idx, w = cap[i]
                hh = hh.reshape(-1, hh.shape[-1]).float()
                idx = idx.reshape(hh.shape[0], -1)
                w = w.reshape(hh.shape[0], -1).float()
                st = ST[i]
                pw = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                st["wtop1"].add(pw.max(-1).values)
                st["wmin"].add(pw.min(-1).values)
                st["went"].add(-(pw * pw.clamp_min(1e-9).log()).sum(-1))
                al = scal(mod)
                for e in range(E):
                    sel = (idx == e).any(-1)
                    ntok = int(sel.sum())
                    st["load"][e] += ntok
                    if ntok == 0:
                        continue
                    g, u = (hh[sel] @ mod.gate_up_proj[e].float().T).chunk(2, dim=-1)
                    z, gain, act, r = act_of(a.arm, g, (al[e] if al is not None else None))
                    prod = act * u
                    st["gate"][e].add(g); st["up"][e].add(u); st["z"][e].add(z)
                    st["act"][e].add(act); st["prod"][e].add(prod)
                    st["gain"][e].add(gain); st["tokrms"][e].add(r)
                    st["outv"][e].add(prod @ mod.down_proj[e].float().T)
                    st["lin"][e] += int((z > 3).sum()); st["dead"][e] += int((z < -3).sum())
                    st["band"][e] += int((z.abs() <= 3).sum()); st["tot"][e] += z.numel()
            print(f"  batch {b+1}/{a.batches}", flush=True)
    for x in handles:
        x.remove()

    rep = dict(arm=a.arm, ckpt=a.ckpt, E=E, I=I, d=D,
               tokens=a.batches * a.batch * a.seq_len, layers=[])
    for i, (name, mod) in enumerate(experts):
        st = ST[i]
        tl = sum(st["load"]) or 1
        al = scal(mod)
        exs = []
        for e in range(E):
            tr = st["tokrms"][e].out()
            d = dict(expert=e, load_frac=st["load"][e] / tl,
                     cv_tok_rms_gate=(tr["std"] / tr["mean"]) if tr.get("mean") else float("nan"),
                     frac_linear=st["lin"][e] / max(st["tot"][e], 1),
                     frac_dead=st["dead"][e] / max(st["tot"][e], 1),
                     frac_band=st["band"][e] / max(st["tot"][e], 1))
            for k in KEYS:
                d[k] = st[k][e].out()
            if al is not None:
                d["scalar_raw"] = al[e].item()
                d["scalar_eff"] = (torch.sigmoid(al[e]).item() if a.arm == "radial" else al[e].item())
            exs.append(d)
        loads = torch.tensor([x["load_frac"] for x in exs])
        Wg = mod.gate_up_proj[:, :I, :].float().reshape(E, -1)
        Wn = Wg / Wg.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Wn @ Wn.T
        off = C[~torch.eye(E, dtype=torch.bool, device=C.device)]
        rep["layers"].append(dict(
            idx=i, module=name,
            router=dict(top1=st["wtop1"].out()["mean"], wmin=st["wmin"].out()["mean"],
                        entropy=st["went"].out()["mean"],
                        eff_experts=float(torch.tensor(st["went"].out()["mean"]).exp()),
                        load_max_over_mean=(loads.max() / loads.mean()).item(),
                        load_cv=(loads.std() / loads.mean()).item(),
                        dead_experts=int((loads < 1e-4).sum())),
            xcorr_mean_abs=off.abs().mean().item(),
            scalar=(dict(mean=_nm([x["scalar_eff"] for x in exs]),
                         min=min(x["scalar_eff"] for x in exs),
                         max=max(x["scalar_eff"] for x in exs),
                         cv=(torch.tensor([x["scalar_eff"] for x in exs]).std()
                             / abs(_nm([x["scalar_eff"] for x in exs]))).item())
                    if al is not None else None),
            experts=exs))
    with open(a.out, "w") as f:
        json.dump(rep, f)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
