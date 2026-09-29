"""8-bit recipes on a REAL trained MoE layer: outliers per phase, underflow / overflow per GEMM operand,
and forward / backward parity of every intermediate and gradient.

    python -m ablate.tools.quant_layer_study <result.json> [--layers 0,1,5,9] [--n_seqs 8]

The layer's real input AND its real upstream gradient are captured from a CE forward + backward on
holdout text. Then the layer runs as an explicit fp32 chain -- RMSNorm -> sigmoid router (fp32,
never quantized) -> per expert: GU = x @ Wgu^T (F1), G/U split, inter = radial(G) * U, O = inter @
Wdn^T (F3), y += w * O -- with each recipe's fake quantization inserted at the GEMM operands:
  fwd    F1: x, Wgu along H          F3: inter, Wdn along I
  dgrad  B3: dO, Wdn along H         B6: dGU, Wgu along 2I
  wgrad  B2: dO^T, inter^T along tokens (32-token blocks per expert)   B5: dGU^T, x^T along tokens
Everything else (norm, router, activation, combine) stays fp32. The routing (top-k set) is computed
once and shared, so every recipe is compared on the same tokens-to-experts assignment.

Scales: e8m0 = power of two rounded up (the native MX scale); fp32 / bf16 = amax / max; "2L-e5m2" /
"2L-e4m3" = a TWO-LEVEL scale: one fp32 per tensor x an 8-bit float per block (rounded up), the
NVFP4 construction applied to 8-bit elements; "e5m2-only" = an e5m2 block scale with no fp32 tensor
scale (what an 8-bit float scale can cover on its own).
"""
from ablate.common import _paths  # noqa: F401
import argparse
import importlib
import json
import math

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from ablate.common import patches as P

K75 = importlib.import_module("kernels.sm75.moe")
DEV = "cuda"


# ------------------------------------------------------------------ number formats
def emu(r, E, M, bias, maxv, ceil=False):
    """(1, E, M) float with subnormals, saturating at maxv; round-to-nearest-even (or UP)."""
    a = r.abs().clamp(max=maxv)
    e = (torch.frexp(a)[1] - 1).float().clamp(min=1 - bias)
    step = torch.exp2(e - M)
    q = (torch.ceil(a / step) if ceil else torch.round(a / step)) * step
    return torch.copysign(q.clamp(max=maxv), r)


ELEM = {"e4m3": (lambda r: r.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float(), 448.0, 2.0 ** -6),
        "e5m2": (lambda r: r.clamp(-57344.0, 57344.0).to(torch.float8_e5m2).float(), 57344.0, 2.0 ** -14),
        "e3m4": (lambda r: emu(r, 3, 4, 3, 30.0), 30.0, 2.0 ** -2),
        # FP4: 1 sign / 2 exp / 1 mantissa, bias 1 -> {0, .5, 1, 1.5, 2, 3, 4, 6}
        "e2m1": (lambda r: emu(r, 2, 1, 1, 6.0), 6.0, 1.0)}
SCALE8 = {"e5m2": (5, 2, 15, 57344.0), "e4m3": (4, 3, 7, 448.0)}

STATS = {}


def block_scale(amax, scale, emax):
    b = amax / emax
    if scale == "e8m0":
        return torch.exp2(torch.ceil(torch.log2(b)).clamp(-127, 127))
    if scale == "fp32":
        return b
    if scale == "bf16":
        return (b * (1 + 2 ** -7)).to(torch.bfloat16).float()
    if scale.startswith("2L-"):
        E, M, bias, mx = SCALE8[scale[3:]]
        S = b.amax() / mx                                   # one fp32 per tensor: top block -> top of the format
        return S * emu(b / S, E, M, bias, mx, ceil=True).clamp_min(2.0 ** (1 - bias - M))
    if scale == "e5m2-only":
        E, M, bias, mx = SCALE8["e5m2"]
        return emu(b, E, M, bias, mx, ceil=True).clamp_min(2.0 ** (1 - bias - M))
    raise ValueError(scale)


def fq(t, dim, recipe, tag):
    """Fake-quantize t in 1D blocks along `dim`; records flushed / subnormal / saturated counts."""
    fmt, scale, blk = recipe[:3]
    if fmt == "fp32":
        return t
    if fmt == "bf16":
        return t.to(torch.bfloat16).float()
    qf, emax, mn = ELEM[fmt]
    x = t.float().movedim(dim, -1)
    K = x.shape[-1]
    pad = (-K) % blk
    if pad:
        x = F.pad(x, (0, pad))
    xb = x.reshape(*x.shape[:-1], -1, blk)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = block_scale(amax, scale, emax)
    r = xb / s
    y = qf(r) * s
    if tag is not None:
        nz = xb != 0
        st = STATS.setdefault(tag, [0.0, 0.0, 0.0, 0.0])
        st[0] += ((y == 0) & nz).sum().item()
        st[1] += ((r.abs() < mn) & nz).sum().item()
        st[2] += (r.abs() > emax * 1.0001).sum().item()
        st[3] += nz.sum().item()
    y = y.reshape(*x.shape)
    if pad:
        y = y[..., :K]
    return y.movedim(-1, dim)


_HAD = {}


def hadamard(t, dim, n):
    """Rotate t along `dim` in blocks of n by the normalized Sylvester Hadamard (orthogonal, so
    applying it to BOTH operands along a GEMM's reduction dim leaves the product unchanged)."""
    if not n:
        return t
    H = _HAD.get(n)
    if H is None:
        H = torch.ones(1, 1, device=DEV)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = H / math.sqrt(n)
        _HAD[n] = H
    x = t.float().movedim(dim, -1)
    K = x.shape[-1]
    pad = (-K) % n
    if pad:
        x = F.pad(x, (0, pad))
    # stays PADDED: both GEMM operands pad the same reduction dim with zeros, and cutting the padding
    # back off after rotating would drop real mass (the rotation spreads it into the pad positions)
    y = (x.reshape(*x.shape[:-1], -1, n) @ H).reshape(*x.shape)
    return y.movedim(-1, dim)


class QMM(torch.autograd.Function):
    """y = a @ b^T with a (m, K) activations, b (n, K) weights; every GEMM operand quantized along its
    own reduction dim. name = 'F1' or 'F3' (fwd); its dgrad / wgrad are tagged B6/B5 or B3/B2."""

    @staticmethod
    def forward(ctx, a, b, recipe, name):
        ctx.save_for_backward(a, b)
        ctx.r, ctx.n = recipe, name
        h = recipe[3] if len(recipe) > 3 else 0
        wr = recipe[4] if len(recipe) > 4 else recipe[:3]      # weights may use their own format (W4A8)
        return fq(hadamard(a, 1, h), 1, recipe[:3], f"{name}.act") @ fq(hadamard(b, 1, h), 1, wr, f"{name}.W").t()

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        r, n = ctx.r, ctx.n
        dg, wg = ("B6", "B5") if n == "F1" else ("B3", "B2")
        h = r[3] if len(r) > 3 else 0
        wr = r[4] if len(r) > 4 else r[:3]
        r = r[:3]
        H = lambda t, d: hadamard(t, d, h)
        da = fq(H(dy, 1), 1, r, f"{dg}.grad") @ fq(H(b, 0), 0, wr, f"{dg}.W^T")
        # wgrad reduces over TOKENS: rotate along tokens (NVIDIA's NVFP4 recipe puts its RHT exactly here)
        db = fq(H(dy, 0), 0, r, f"{wg}.grad^T").t() @ fq(H(a, 0), 0, r, f"{wg}.act^T")
        return da, db, None, None


# ------------------------------------------------------------------ the layer chain
def run_layer(L, x0, gy, idx, recipe):
    moe = L.mlp
    ex = moe.experts
    E = moe.gate.num_routed_experts
    b, s, h = x0.shape
    params = {"norm_w": L.post_attention_layernorm.weight, "router_w": moe.gate.gate_proj.weight,
              "gate_up": ex.gate_up_proj, "down": ex.down_proj, "theta": ex.radial_theta}
    for p in params.values():
        p.grad = None
    x = x0.float().clone().requires_grad_(True)
    hn = params["norm_w"] * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + L.post_attention_layernorm.variance_epsilon))
    hn.retain_grad()
    flat = hn.reshape(-1, h)
    scores = torch.sigmoid(flat @ params["router_w"].t())
    tw = scores.gather(-1, idx)
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)
    codes = P._expert_codes(ex, DEV, torch.float32)
    codes_l = codes.tolist() if torch.is_tensor(codes) else list(codes)
    ap = P._act_params(ex)
    I = params["down"].shape[2]
    y = torch.zeros(b * s, h, device=DEV)
    keep = {"G": [], "U": [], "inter": [], "O": []}
    for e in range(E):
        sel = (idx == e)
        rows = sel.any(-1)
        if not bool(rows.any()):
            continue
        tok = rows.nonzero().squeeze(-1)
        w = (tw * sel).sum(-1)[rows]
        GU = QMM.apply(flat[tok], params["gate_up"][e], recipe, "F1")
        G, U = GU[:, :I], GU[:, I:]
        G.retain_grad(); U.retain_grad()
        inter = K75._act_eager(G, codes_l[e], ap[e, 0] if codes_l[e] in (8, 10) else 1.0) * U
        inter.retain_grad()
        O = QMM.apply(inter, params["down"][e], recipe, "F3")
        O.retain_grad()
        y = y.index_add(0, tok, O * w.unsqueeze(-1))
        for k_, v in (("G", G), ("U", U), ("inter", inter), ("O", O)):
            keep[k_].append(v)
    y = y.reshape(b, s, h)
    (y * gy).sum().backward()
    cat = lambda k_: torch.cat([t.detach() for t in keep[k_]])
    catg = lambda k_: torch.cat([t.grad.detach() for t in keep[k_]])
    out = {"fwd G (gate)": cat("G"), "fwd U (up)": cat("U"), "fwd act(G)*U": cat("inter"),
           "fwd O (down out)": cat("O"), "fwd y": y.detach(),
           "bwd dO": catg("O"), "bwd d act-out": catg("inter"), "bwd dG": catg("G"), "bwd dU": catg("U"),
           "bwd d normed-x": hn.grad.detach(), "bwd dx": x.grad.detach()}
    gu_g = params["gate_up"].grad
    out.update({"dW gate": gu_g[:, :I].clone(), "dW up": gu_g[:, I:].clone(), "dW down": params["down"].grad.clone(),
                "d theta": params["theta"].grad.clone() if params["theta"].grad is not None else None,
                "d router": params["router_w"].grad.clone(), "d norm_w": params["norm_w"].grad.clone()})
    return out


def outliers(name, t):
    a = t.float().abs().reshape(-1)
    a = a[a > 0]
    rms = a.pow(2).mean().sqrt()
    kurt = (a.pow(4).mean() / a.pow(2).mean().pow(2)).item()
    blk = t.float().abs().reshape(-1, 32) if t.shape[-1] % 32 == 0 else None
    if blk is not None:
        bmax = blk.amax(-1)
        bmed = blk.median(-1).values.clamp_min(1e-30)
        ratio = (bmax / bmed)
        sub = ((blk < bmax[:, None] / 28672) & (blk > 0)).float().mean().item()
        rs = f"{ratio.median().item():8.1f} {ratio.quantile(0.99).item() if ratio.numel() < 16_000_000 else float('nan'):9.1f} {100 * sub:9.4f}"
    else:
        rs = f"{'-':>8s} {'-':>9s} {'-':>9s}"
    samp = a if a.numel() < 16_000_000 else a[torch.randperm(a.numel(), device=a.device)[:8_000_000]]
    print(f"   {name:18s} {(a.max() / samp.median()).item():10.0f} {(a.max() / rms).item():8.1f} {kurt:8.1f} "
          f"{100 * (a > 10 * rms).float().mean().item():8.4f} {100 * (a > 100 * rms).float().mean().item():8.5f} {rs}")


MX4, NV4 = ("e2m1", "e8m0", 32), ("e2m1", "2L-e4m3", 16)
RECIPES4 = [("bf16", None, 0), ("e4m3", "e8m0", 32),
            ("e4m3", "e8m0", 32, 0, MX4), ("e4m3", "e8m0", 32, 0, NV4),            # W4A8 (QAT: 4-bit weights)
            ("e4m3", "e8m0", 32, 32, MX4), ("e4m3", "e8m0", 32, 16, NV4),          # W4A8 + Hadamard
            MX4, NV4,                                                              # W4A4
            (*MX4, 32), (*NV4, 16)]                                                # W4A4 + Hadamard
RECIPES = [("bf16", None, 0),
           ("e4m3", "e8m0", 32, 32), ("e4m3", "e8m0", 32, 128), ("e4m3", "e8m0", 128, 128),
           ("e4m3", "e8m0", 32), ("e4m3", "e8m0", 64), ("e4m3", "e8m0", 128),
           ("e4m3", "fp32", 32), ("e4m3", "fp32", 128), ("e4m3", "bf16", 32),
           ("e4m3", "2L-e5m2", 32), ("e4m3", "2L-e4m3", 32), ("e4m3", "e5m2-only", 32),
           ("e5m2", "e8m0", 32), ("e3m4", "e8m0", 32), ("e3m4", "fp32", 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--layers", default="0,1,5,9")
    ap.add_argument("--n_seqs", type=int, default=8)
    ap.add_argument("--four", action="store_true", help="the 4-bit grid (W4A8 / W4A4, MXFP4 / NVFP4, +-Hadamard)")
    a = ap.parse_args()
    c0 = json.load(open(a.result))["config"]
    model, c = load_from_result(a.result)
    hold = _val.build_holdout(c0["dataset"], c0["seq_len"], a.n_seqs, DEV)
    layers = [int(v) for v in a.layers.split(",")]
    cap = {}
    orig = {}
    for li in layers:
        Lx = model.model.layers[li]
        orig[li] = Lx._attn_res_mlp_forward

        def grab(x, li=li, f0=orig[li]):
            cap[(li, "x")] = x.detach().clone()
            out = f0(x)
            if out.requires_grad:
                out.register_hook(lambda g: cap.__setitem__((li, "gy"), g.detach().float().clone()))
            return out
        Lx._attn_res_mlp_forward = grab
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hN = model.model(input_ids=hold[:, :-1], use_cache=False).last_hidden_state
        loss = F.cross_entropy(model.lm_head(hN).float().reshape(-1, model.lm_head.weight.shape[0]),
                               hold[:, 1:].reshape(-1), ignore_index=0)
    loss.backward()
    for li in layers:
        model.model.layers[li]._attn_res_mlp_forward = orig[li]
    model.zero_grad(set_to_none=True)
    print(f"[qstudy] holdout {tuple(hold.shape)}, CE {loss.item():.4f}; captured real input + upstream grad for layers {layers}",
          flush=True)

    for li in layers:
        L = model.model.layers[li]
        moe = L.mlp
        x0, gy = cap[(li, "x")], cap[(li, "gy")]
        b, s, h = x0.shape
        with torch.no_grad():
            xf = x0.float()
            hn = L.post_attention_layernorm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6))
            sc = torch.sigmoid(hn.reshape(-1, h) @ moe.gate.gate_proj.weight.t())
            sel = sc + (moe.gate.bias if moe.gate.bias is not None else 0)
            idx = sel.topk(moe.gate.top_k, -1).indices
        E, k = moe.gate.num_routed_experts, moe.gate.top_k
        print(f"\n================ layer {li}: {E} experts top-{k}, {b * s} tokens, {b * s * k} expert rows ================")
        STATS.clear()
        ref = run_layer(L, x0, gy, idx, ("fp32", None, 0))
        print("OUTLIERS per phase (fp32 reference). amax/med = max |v| over median |v|; kurt = E[v^4]/E[v^2]^2 "
              "(3 = Gaussian); blk ratio = per-32-block max/median; subn% = values below blockmax/28672 "
              "(the e4m3 subnormal zone)")
        print(f"   {'tensor':18s} {'amax/med':>10s} {'amax/rms':>8s} {'kurt':>8s} {'>10rms%':>8s} {'>100rms%':>8s} "
              f"{'blk p50':>8s} {'blk p99':>9s} {'subn%':>9s}")
        outliers("x (layer in)", x0)
        outliers("normed x (F1 in)", hn)
        for kk in ("fwd G (gate)", "fwd U (up)", "fwd act(G)*U", "fwd O (down out)", "fwd y",
                   "bwd dO", "bwd d act-out", "bwd dG", "bwd dU", "bwd d normed-x"):
            outliers(kk.replace("fwd ", "").replace("bwd ", "d:"), ref[kk])
        outliers("upstream grad", gy)

        print("\nPARITY: relative Frobenius error vs the fp32 chain (same routing)")
        keys = [k_ for k_ in ref if ref[k_] is not None]
        res, stats = {}, {}
        grid = RECIPES4 if a.four else RECIPES
        for rc in grid:
            STATS.clear()
            out = run_layer(L, x0, gy, idx, rc)
            res[rc] = {k_: ((out[k_].float() - ref[k_].float()).norm() / ref[k_].float().norm()).item()
                       for k_ in keys if out[k_] is not None}
            stats[rc] = {t: tuple(100 * v / max(st[3], 1) for v in st[:3]) for t, st in STATS.items()}
            del out
        def name(rc):
            if rc[1] is None:
                return rc[0]
            fmt = lambda f: {"e2m1 e8m0 32": "MXFP4", "e2m1 2L-e4m3 16": "NVFP4", "e4m3 e8m0 32": "MXFP8"}.get(
                f"{f[0]} {f[1]} {f[2]}", f"{f[0]} {f[1]} {f[2]}")
            base = fmt(rc) if len(rc) < 5 else f"W:{fmt(rc[4])} A:{fmt(rc)}"
            return base + (f" +H{rc[3]}" if len(rc) > 3 and rc[3] else "")
        short = {"fwd G (gate)": "G", "fwd U (up)": "U", "fwd act(G)*U": "act", "fwd O (down out)": "O",
                 "fwd y": "y", "bwd dO": "dO", "bwd d act-out": "d act", "bwd dG": "dG", "bwd dU": "dU",
                 "bwd d normed-x": "d hn", "bwd dx": "dx", "dW gate": "dW g", "dW up": "dW u",
                 "dW down": "dW d", "d theta": "d th", "d router": "d rtr", "d norm_w": "d nw"}
        print(f"   {'recipe':22s} " + " ".join(f"{short[k_]:>7s}" for k_ in keys))
        for rc in grid:
            print(f"   {name(rc):26s} " + " ".join(f"{res[rc].get(k_, float('nan')):7.1e}" for k_ in keys))

        print("\nUNDERFLOW / OVERFLOW per GEMM operand, % of non-zero values: flushed-to-0 / subnormal / saturated")
        tags = ["F1.act", "F1.W", "F3.act", "F3.W", "B3.grad", "B3.W^T", "B6.grad", "B6.W^T",
                "B2.grad^T", "B2.act^T", "B5.grad^T", "B5.act^T"]
        print(f"   {'recipe':22s} " + " ".join(f"{t:>21s}" for t in tags))
        for rc in grid:
            if rc[1] is None:
                continue
            cells = []
            for t in tags:
                v = stats[rc].get(t)
                cells.append(f"{v[0]:6.3f}/{v[1]:6.2f}/{v[2]:6.3f}" if v else f"{'-':>21s}")
            print(f"   {name(rc):26s} " + " ".join(f"{c_:>21s}" for c_ in cells))
    print("\nQUANT_LAYER_STUDY_DONE")


if __name__ == "__main__":
    main()
