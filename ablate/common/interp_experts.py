"""Per-expert / per-layer interp on a trained BiBo MoE checkpoint.

KERNEL-INDEPENDENT BY CONSTRUCTION. The only thing hooked is the expert module's forward_pre_hook,
whose args are (hidden_states, top_k_index, top_k_weights) -- the same point RouterTrace uses. The
gate/up pre-activations are RECOMPUTED from gate_up_proj, so nothing here depends on which dispatch
path (triton fused / eager) actually ran. gate_up_proj[e] is (2I, d) and chunk(2, -1) gives gate
first, up second (src/modeling/ffn/moe.py:100-101).

Everything accumulates as running moments -- no raw activations are retained -- so this runs
comfortably alongside a training job on the same GPU.

WHAT THE COLUMNS ARE FOR:
  gate_rms      the scale SiLU actually sees. The round measured 5.6-9.3 at 524M, which is why
                plain SiLU degenerates toward ReLU and why normsilu's /r matters at all.
  %lin/%dead    fraction of gate values above +3 / below -3, where SiLU is within ~5% of the
                identity / of zero. This is the degeneracy made quantitative.
  CV_tok        per-EXPERT coefficient of variation of the per-TOKEN rms(gate). This is the exact
                quantity radial's learned p tracked inversely (p->0 where r is noisy per token).
  CV_exp        spread of mean rms(gate) ACROSS experts in a layer -- the output-scale spread that
                radial's r^p was found to amplify (spearman(p, gate scale) = +0.46..0.66).
  xcorr         mean |cos| between experts' gate blocks: redundancy / collapse.

Usage:
  python -m ablate.common.interp_experts --ckpt run_final.pt --out interp.json --act silu
"""
from . import _paths  # noqa: F401
import argparse
import contextlib
import json

import torch
import torch.nn.functional as F


class Acc:
    """Streaming moments. Kurtosis needs the 4th power, so all four sums are carried."""

    def __init__(self):
        self.n = 0
        self.s1 = self.s2 = self.s3 = self.s4 = 0.0
        self.mn, self.mx = float("inf"), float("-inf")

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

    # An expert that received ZERO tokens is a real outcome, not an error -- return a full dict of
    # NaNs so every downstream key exists and the deadness propagates visibly instead of KeyError-ing.
    _KEYS = ("mean", "std", "rms", "min", "max", "skew", "kurtosis", "absmax_over_rms")

    def out(self):
        if not self.n:
            return dict(n=0, **{k: float("nan") for k in self._KEYS})
        m = self.s1 / self.n
        var = max(self.s2 / self.n - m * m, 0.0)
        sd = var ** 0.5
        rms = (self.s2 / self.n) ** 0.5
        skew = kurt = float("nan")
        if sd > 1e-12:
            skew = (self.s3 / self.n - 3 * m * var - m ** 3) / sd ** 3
            kurt = (self.s4 / self.n - 4 * m * self.s3 / self.n
                    + 6 * m * m * self.s2 / self.n - 3 * m ** 4) / var ** 2
        return dict(n=self.n, mean=m, std=sd, rms=rms, min=self.mn, max=self.mx, skew=skew,
                    kurtosis=kurt,
                    absmax_over_rms=(max(abs(self.mn), abs(self.mx)) / rms if rms > 1e-12 else float("nan")))


def _nanmean(xs):
    v = [x for x in xs if x == x]          # NaN != NaN; drops dead experts
    return sum(v) / len(v) if v else float("nan")


def _weight_stats(mod, e, I):
    W = mod.gate_up_proj[e].float()
    Wg, Wu = W[:I], W[I:]
    Wd = mod.down_proj[e].float()
    sv = torch.linalg.svdvals(Wg)
    q = sv / sv.sum().clamp_min(1e-12)
    return dict(fro_gate=Wg.norm().item(), fro_up=Wu.norm().item(), fro_down=Wd.norm().item(),
                sv_max=sv[0].item(), sv_min=sv[-1].item(),
                cond=(sv[0] / sv[-1].clamp_min(1e-12)).item(),
                # spectral entropy: high = the expert uses many directions, low = near rank-1
                sv_entropy=(-(q * q.clamp_min(1e-12).log()).sum()).item(),
                stable_rank=(sv.pow(2).sum() / sv[0].pow(2).clamp_min(1e-12)).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--experts", type=int, default=64)   # TOTAL routed (GLU + specials)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--dataset", default="/home/marimo/work/data/bip2")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = a.device

    from ablate.common import patches as patchmod
    from ablate.common.models import build_arm
    from ablate.common.data import token_batches
    # The activation is radial NormSiLU on BOTH paths now -- src's eager expert loop and the Triton
    # patch read the same radial_theta parameter, so they cannot disagree. They USED to: the eager
    # path indexed a hardcoded ("silu","relu2","normsilu")[e%3] tuple while ACT_CYCLE steered only
    # the patched forward, which ran 2/3 of the experts on the wrong activation and silently
    # corrupted every downstream layer's routing. The selfcheck below still verifies it end-to-end.

    model, _cfg = build_arm("bibo_min", device=dev, dtype=torch.float32, attn_impl="sdpa",
                            num_experts=a.experts, top_k=a.top_k)
    # load to CPU first: on a shared GPU the checkpoint copy is a 2.6 GB spike that can OOM a
    # training run sharing the device. load_state_dict copies onto the (already-placed) model.
    sd = torch.load(a.ckpt, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt}  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > 20:
        print(f"  WARNING first missing: {missing[:5]}", flush=True)
    model.eval()

    experts = [(n, m) for n, m in model.named_modules() if hasattr(m, "gate_up_proj")]
    E, twoI, D = experts[0][1].gate_up_proj.shape
    I = twoI // 2
    print(f"{len(experts)} MoE layers | E={E} I={I} d={D}", flush=True)

    ST = [dict(gate=[Acc() for _ in range(E)], up=[Acc() for _ in range(E)],
               act=[Acc() for _ in range(E)], outn=[Acc() for _ in range(E)],
               zin=[Acc() for _ in range(E)],
               tokrms=[Acc() for _ in range(E)],
               load=[0] * E, lin=[0] * E, dead=[0] * E, band=[0] * E, tot=[0] * E,
               w_top1=Acc(), w_ent=Acc())
          for _ in experts]

    cap = {}

    def mk(i):
        def hook(_mod, args):
            cap[i] = (args[0].detach(), args[1].detach(), args[2].detach())
        return hook

    handles = [m.register_forward_pre_hook(mk(i)) for i, (_n, m) in enumerate(experts)]
    gen = token_batches(a.batch, a.seq_len, dev, dataset=a.dataset)

    # ---- END-TO-END SELF-CHECK -------------------------------------------------------------
    # Assert the MODEL computes what this script assumes, by recomputing one expert layer's output
    # from the weights and comparing to what the module actually returned. A stats script that
    # merely runs without error proves nothing about which activation the forward used -- that is
    # exactly how the _POLYGLU_ACTIVATIONS mismatch above went unnoticed.
    got = {}
    h_out = experts[0][1].register_forward_hook(lambda _m, _i, o: got.__setitem__("y", o.detach()))
    ids0 = next(gen)
    with torch.no_grad():
        model.model(input_ids=ids0[:, :-1], use_cache=False)
    h_out.remove()
    _mod = experts[0][1]
    _h, _idx, _w = cap[0]
    _h = _h.reshape(-1, _h.shape[-1]).float()
    _idx = _idx.reshape(_h.shape[0], -1)
    _w = _w.reshape(_h.shape[0], -1).float()
    ref = torch.zeros_like(_h)
    for e in range(E):
        m_ = (_idx == e)
        sel = m_.any(-1)
        if not bool(sel.any()):
            continue
        he = _h[sel]
        g_, u_ = (he @ _mod.gate_up_proj[e].float().T).chunk(2, dim=-1)
        r_ = g_.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        p_ = torch.sigmoid(_mod.radial_theta[e].float())
        y_ = (F.silu(g_ / r_) * r_.pow(p_) * u_) @ _mod.down_proj[e].float().T
        ref[sel] += y_ * (_w[sel] * m_[sel].float()).sum(-1, keepdim=True)
    y = got["y"].reshape(-1, got["y"].shape[-1]).float()
    err = (y - ref).abs().max().item() / max(y.abs().max().item(), 1e-9)
    print(f"[selfcheck] layer0 expert-output rel err = {err:.3e}  (act=radial)", flush=True)
    if err > 5e-2:
        raise SystemExit(f"SELFCHECK FAILED (rel err {err:.3e}): the forward is NOT computing "
                         f"radial NormSiLU on every expert -- stats would be meaningless.")

    # fp32 on CPU (no autocast) -- more accurate for statistics, and bf16 autocast is CUDA-only
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext())
    with torch.no_grad():
        for b in range(a.batches):
            ids = next(gen)
            with amp:
                model.model(input_ids=ids[:, :-1], use_cache=False)
            for i, (_n, mod) in enumerate(experts):
                h, idx, w = cap[i]
                h = h.reshape(-1, h.shape[-1]).float()
                idx = idx.reshape(h.shape[0], -1)
                w = w.reshape(h.shape[0], -1).float()
                st = ST[i]
                pw = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
                st["w_top1"].add(pw.max(-1).values)
                st["w_ent"].add(-(pw * pw.clamp_min(1e-9).log()).sum(-1))
                for e in range(E):
                    sel = (idx == e).any(-1)
                    n_tok = int(sel.sum())
                    st["load"][e] += n_tok
                    if n_tok == 0:
                        continue
                    he = h[sel]
                    W = mod.gate_up_proj[e].float()
                    g, u = (he @ W.T).chunk(2, dim=-1)
                    st["gate"][e].add(g)
                    st["up"][e].add(u)
                    st["tokrms"][e].add(g.pow(2).mean(-1).sqrt())     # ONE value per token
                    # z = what SiLU ACTUALLY SEES. For radial that is g/r, not g, so the regime
                    # split below has to be counted on z -- raw g says nothing about where on the
                    # curve a normalized model sits. gain = r^p is the magnitude radial puts BACK.
                    _r = g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                    z = g / _r
                    gain = _r.pow(torch.sigmoid(mod.radial_theta[e].float()))
                    act = F.silu(z) * gain
                    st["zin"][e].add(z)
                    st["act"][e].add(act)
                    st["outn"][e].add(((act * u) @ mod.down_proj[e].float().T).pow(2).mean(-1).sqrt())
                    # SiLU degeneracy: >+3 is within ~5% of identity, <-3 within ~5% of zero
                    st["lin"][e] += int((z > 3).sum())
                    st["dead"][e] += int((z < -3).sum())
                    st["band"][e] += int((z.abs() <= 3).sum())
                    st["tot"][e] += z.numel()
            print(f"  batch {b + 1}/{a.batches}", flush=True)
    for hh in handles:
        hh.remove()

    report = dict(ckpt=a.ckpt, act="radial", E=E, I=I, d=D,
                  tokens_seen=a.batches * a.batch * a.seq_len, layers=[])
    for i, (name, mod) in enumerate(experts):
        st = ST[i]
        tot_load = sum(st["load"]) or 1
        exs = []
        for e in range(E):
            tr = st["tokrms"][e].out()
            cv = (tr["std"] / tr["mean"]) if tr.get("mean") else float("nan")
            exs.append(dict(expert=e, load_frac=st["load"][e] / tot_load,
                            gate=st["gate"][e].out(), up=st["up"][e].out(),
                            silu_input=st["zin"][e].out(),
                            act=st["act"][e].out(), out_rms=st["outn"][e].out(),
                            tok_rms_gate=tr, cv_tok_rms_gate=cv,
                            frac_linear=st["lin"][e] / max(st["tot"][e], 1),
                            frac_dead=st["dead"][e] / max(st["tot"][e], 1),
                            frac_band=st["band"][e] / max(st["tot"][e], 1),
                            weights=_weight_stats(mod, e, I)))
        Wg = mod.gate_up_proj[:, :I, :].float().reshape(E, -1)
        Wn = Wg / Wg.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Wn @ Wn.T
        off = C[~torch.eye(E, dtype=torch.bool, device=C.device)]
        loads = torch.tensor([x["load_frac"] for x in exs])
        # nan-safe: dead experts contribute no scale, and averaging them in as 0 would understate
        # the live experts' spread. Report them separately via router.dead_experts instead.
        rms_ac = torch.tensor([x["gate"]["rms"] for x in exs])
        rms_ac = rms_ac[~torch.isnan(rms_ac)]
        report["layers"].append(dict(
            idx=i, module=name,
            router=dict(top1_weight=st["w_top1"].out().get("mean"),
                        entropy=st["w_ent"].out().get("mean"),
                        load_max=loads.max().item(), load_min=loads.min().item(),
                        load_max_over_mean=(loads.max() / loads.mean()).item(),
                        load_cv=(loads.std() / loads.mean()).item(),
                        dead_experts=int((loads < 1e-4).sum())),
            gate_rms_across_experts=dict(mean=rms_ac.mean().item(), std=rms_ac.std().item(),
                                         cv=(rms_ac.std() / rms_ac.mean()).item(),
                                         min=rms_ac.min().item(), max=rms_ac.max().item()),
            cv_tok_rms_gate_layer=_nanmean([x["cv_tok_rms_gate"] for x in exs]),
            frac_linear_layer=_nanmean([x["frac_linear"] for x in exs]),
            frac_dead_layer=_nanmean([x["frac_dead"] for x in exs]),
            frac_band_layer=_nanmean([x["frac_band"] for x in exs]),
            xcorr=dict(mean_abs=off.abs().mean().item(), max_abs=off.abs().max().item(),
                       rms=off.pow(2).mean().sqrt().item()),
            experts=exs))

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}", flush=True)

    print("\n" + "=" * 108)
    print(f"{'L':>2} {'gate_rms':>9} {'z_rms':>7} {'CV_tok':>7} {'CV_exp':>7} {'%lin':>6} {'%dead':>6} {'%band':>6} "
          f"{'kurt':>6} {'top1w':>6} {'rent':>6} {'ldmx/mn':>8} {'xcorr':>7} {'out_rms':>8} {'dead_e':>6}")
    print("=" * 108)
    for L in report["layers"]:
        ex = L["experts"]
        k = _nanmean([x["gate"]["kurtosis"] for x in ex])
        orms = _nanmean([x["out_rms"]["mean"] for x in ex])
        zr = _nanmean([x["silu_input"]["rms"] for x in ex])
        print(f"{L['idx']:>2} {L['gate_rms_across_experts']['mean']:>9.3f} {zr:>7.3f} "
              f"{L['cv_tok_rms_gate_layer']:>7.3f} {L['gate_rms_across_experts']['cv']:>7.3f} "
              f"{100 * L['frac_linear_layer']:>6.2f} {100 * L['frac_dead_layer']:>6.2f} "
              f"{100 * L['frac_band_layer']:>6.2f} {k:>6.2f} {L['router']['top1_weight']:>6.3f} "
              f"{L['router']['entropy']:>6.3f} {L['router']['load_max_over_mean']:>8.2f} "
              f"{L['xcorr']['mean_abs']:>7.4f} {orms:>8.3f} {L['router']['dead_experts']:>6d}")


if __name__ == "__main__":
    main()
