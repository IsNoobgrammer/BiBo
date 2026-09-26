"""Quantization-friendliness of finished checkpoints: weight / activation outliers + fake-quant loss.

    python -m ablate.tools.quant_probe <run_a_result.json> [<run_b_result.json> ...] [--seqs 32]

Per checkpoint:
  1. weight outliers per group: per-row max|w|/rms (mean, max) and excess kurtosis. Absolute norm is
     reported but is NOT the quant signal -- per-channel / per-group scales absorb it; the shape
     inside a row (outliers) is what costs bits.
  2. activation outliers per layer, residual stream after each decoder layer: rms, per-token
     max|x|/rms, channel kurtosis, and the largest single activation.
  3. fake-quant (weight-only, symmetric, round-to-nearest) held-out CE delta vs the bf16 baseline:
     int8 per-channel, fp8-e4m3 per-channel, int4 group-128 (group-64 where the input width
     is not a multiple of 128: expert down_proj, in=576). Every weight matrix is quantized
     (attention, experts, router, attn-res projections); embed_tokens and lm_head stay full
     precision, as every deployment recipe does.

All weight tensors are (..., out, in), experts included, so rows / groups run along the last dim.
Same frozen held-out batch for every checkpoint (validation.build_holdout, first --seqs rows).
"""
from ablate.common import _paths  # noqa: F401
import argparse
import os
import torch

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from ablate.common.tensor_health import _group
from kernels.sm120.cross_entropy import fused_linear_cross_entropy
import importlib
_moe = importlib.import_module("kernels.sm75.moe")   # the package re-exports a function named moe

SKIP = ("embed_tokens", "lm_head")


@torch.no_grad()
def _set(model, weights):
    # p.copy_ (not p.data.copy_) so _version bumps, and drop the MoE bf16 cast cache: it is keyed
    # on data_ptr + _version, so a later checkpoint at a freed address, or a .data write, would
    # silently keep serving the OLD expert weights to the forward.
    for n, p in _mats(model):
        p.copy_(weights[n])
    _moe._CAST_CACHE.clear()


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1(x):
    """Round to the nearest FP4 (e2m1) value, saturating at 6."""
    g = _E2M1.to(x.device)
    a = x.abs().clamp(max=6.0)
    i = torch.bucketize(a, (g[1:] + g[:-1]) / 2)
    return torch.sign(x) * g[i]


def block_fq(x, fmt):
    """Fake-quantize along the LAST dim (the GEMM reduction dim for weights and activations).
    mxfp8/mxfp4: OCP MX, 32-wide blocks, power-of-two E8M0 scale 2^(floor(log2 amax) - emax_elem).
    nvfp4: 16-wide blocks, e4m3 block scale times a per-tensor fp32 scale."""
    x32 = x.float()
    bs = 16 if fmt == "nvfp4" else 32
    *lead, n = x32.shape
    b = x32.reshape(*lead, n // bs, bs)
    amax = b.abs().amax(-1, keepdim=True)
    if fmt in ("mxfp8", "mxfp4"):
        emax = 8 if fmt == "mxfp8" else 2
        sc = torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0 ** -126))) - emax)
        y = b / sc
        y = y.clamp(-448, 448).to(torch.float8_e4m3fn).float() if fmt == "mxfp8" else _e2m1(y)
        return (y * sc).reshape(x32.shape).to(x.dtype)
    if fmt == "nvfp4":
        sg = (x32.abs().amax() / (448.0 * 6.0)).clamp_min(1e-30)
        sb = (amax / 6.0 / sg).clamp(max=448).to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9) * sg
        return (_e2m1(b / sb) * sb).reshape(x32.shape).to(x.dtype)
    raise ValueError(fmt)


BLOCK_FMTS = ("mxfp8", "mxfp4", "nvfp4")


def _quant(w, kind):
    if kind in BLOCK_FMTS:
        return block_fq(w, kind).float()
    w32 = w.float()
    if kind == "int4_g128":
        *lead, n = w32.shape
        gs = 128 if n % 128 == 0 else 64          # expert down_proj has in=576 = 9 x 64
        g = w32.reshape(*lead, n // gs, gs)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7
        return (torch.round(g / s).clamp(-8, 7) * s).reshape(w32.shape)
    s = w32.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    if kind == "int8_ch":
        s = s / 127
        q = torch.round(w32 / s).clamp(-127, 127) * s
        assert (q - w32).abs().max() <= s.max() * 0.5 + 1e-6      # rtn bound
        return q
    if kind == "fp8_ch":
        s = s / 448
        return (w32 / s).to(torch.float8_e4m3fn).float() * s
    raise ValueError(kind)


def _mats(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.ndim >= 2 and p.numel() >= 4096 and not any(k in n for k in SKIP)]


def weight_stats(model):
    out = {}
    for n, p in _mats(model):
        w = p.detach().float().reshape(-1, p.shape[-1])
        rms = w.pow(2).mean(-1).sqrt().clamp_min(1e-12)
        mr = w.abs().amax(-1) / rms
        z = (w - w.mean(-1, keepdim=True)) / w.std(-1, keepdim=True).clamp_min(1e-12)
        k = z.pow(4).mean(-1) - 3
        out.setdefault(_group(n), []).append((p.detach().float().norm().item(), mr.mean().item(),
                                              mr.max().item(), k.mean().item()))
    return {g: [sum(x[i] for x in v) / len(v) if i != 2 else max(x[2] for x in v) for i in range(4)]
            for g, v in out.items()}


class ActQuant:
    """Forward pre-hooks that fake-quantize the INPUT of every attention projection (q/k/v/o).
    Expert GEMM inputs are inside the fused MoE kernel and not reachable from Python; the
    residual-stream table covers their distribution (the same post-norm stream feeds both)."""
    def __init__(self, model):
        self.fmt, self.calls, self.hooks = None, 0, []
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self.hooks.append(m.register_forward_pre_hook(self._pre))

    def _pre(self, _m, args):
        if self.fmt is None:
            return None
        self.calls += 1
        return (block_fq(args[0], self.fmt),) + tuple(args[1:])

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def act_snr(model, holdout, amp):
    """Relative quantization error ||Q(x)-x||^2/||x||^2 of the RMS-normalised residual stream
    after each layer, per block format -- how quantizable the GEMM inputs are."""
    xs, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        def hook(_m, _inp, out, i=i):
            x = out[0] if isinstance(out, tuple) else out
            x = x.float().reshape(-1, x.shape[-1])
            xs[i] = x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        hooks.append(layer.register_forward_hook(hook))
    try:
        holdout_loss(model, holdout[:4], amp)
    finally:
        for h in hooks:
            h.remove()
    return {f: [((block_fq(x, f) - x).pow(2).sum() / x.pow(2).sum()).item() for _, x in sorted(xs.items())]
            for f in BLOCK_FMTS}


@torch.no_grad()
def holdout_loss(model, holdout, amp):
    return _val.losses(model, holdout, None, fused_linear_cross_entropy, amp, pad_id=0)[0]


@torch.no_grad()
def act_stats(model, holdout, amp):
    rows, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        def hook(_m, _inp, out, i=i):
            x = out[0] if isinstance(out, tuple) else out
            x = x.float().reshape(-1, x.shape[-1])
            rms = x.pow(2).mean(-1).sqrt().clamp_min(1e-12)
            z = (x - x.mean(0)) / x.std(0).clamp_min(1e-12)           # per-channel standardise
            ch_k = z.pow(4).mean(0) - 3                                  # kurtosis of each channel
            rows[i] = (rms.mean().item(), (x.abs().amax(-1) / rms).mean().item(),
                       ch_k.max().item(), x.abs().max().item())
        hooks.append(layer.register_forward_hook(hook))
    try:
        holdout_loss(model, holdout[:4], amp)
    finally:
        for h in hooks:
            h.remove()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--seqs", type=int, default=32)
    a = ap.parse_args()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    holdout = None
    summary = {}
    for rj in a.results:
        _moe._CAST_CACHE.clear()
        model, c = load_from_result(rj)
        tag = c.run_tag or os.path.basename(rj)
        if holdout is None:
            holdout = _val.build_holdout(c.dataset, c.seq_len, a.seqs, "cuda")
            print(f"[quant] holdout {tuple(holdout.shape)}", flush=True)
        print(f"\n===== {tag}", flush=True)
        ws = weight_stats(model)
        print(f"  {'weight group':34s} {'norm':>9s} {'max/rms':>8s} {'max/rms':>8s} {'kurt':>7s}")
        print(f"  {'':34s} {'':>9s} {'mean':>8s} {'max':>8s} {'mean':>7s}")
        for g, (nm, mrm, mrx, k) in ws.items():
            print(f"  {g:34s} {nm:9.2f} {mrm:8.2f} {mrx:8.2f} {k:7.2f}")
        acts = act_stats(model, holdout, amp)
        print(f"  {'layer':>5s} {'resid rms':>10s} {'tok max/rms':>12s} {'max ch kurt':>12s} {'max |x|':>9s}")
        for i, (r, mr, ck, mx) in sorted(acts.items()):
            print(f"  {i:5d} {r:10.3f} {mr:12.2f} {ck:12.1f} {mx:9.2f}")
        base = holdout_loss(model, holdout, amp)
        import json
        _fr = json.load(open(rj)).get("final_report", {})
        print(f"  sanity: probe bf16 CE {base:.4f} | in-training final_report ctx1024 "
              f"{_fr.get('ctxabl/ctx1024', float('nan')):.4f} (different text, should be close)", flush=True)
        orig = {n: p.detach().clone() for n, p in _mats(model)}
        res = {"bf16": base}
        for kind in ("int8_ch", "fp8_ch", "int4_g128") + BLOCK_FMTS:
            _set(model, {n: _quant(w, kind) for n, w in orig.items()})
            res[kind] = holdout_loss(model, holdout, amp)
            _set(model, orig)
        aq = ActQuant(model)                      # W + A: weights in the format AND attention inputs
        for f in BLOCK_FMTS:
            _set(model, {n: _quant(w, f) for n, w in orig.items()})
            aq.fmt, aq.calls = f, 0
            res[f"W+A_{f}"] = holdout_loss(model, holdout, amp)
            assert aq.calls > 0, "activation hooks never fired"
            _set(model, orig)
        aq.fmt = None
        aq.remove()
        snr = act_snr(model, holdout, amp)
        print("  residual-stream block-quant rel err % per layer (post-RMS-norm = GEMM-input quantizability):")
        for f, v in snr.items():
            print(f"    {f:6s} mean {100 * sum(v) / len(v):6.3f}  worst {100 * max(v):6.3f}  | "
                  + " ".join(f"{100 * e:.2f}" for e in v))
        res["snr"] = {f: sum(v) / len(v) for f, v in snr.items()}
        assert abs(holdout_loss(model, holdout, amp) - base) < 1e-6          # weights restored
        print("  held-out CE: " + "  ".join(f"{k}={v:.4f}" + ("" if k == "bf16" else f" ({v - base:+.4f})")
                                           for k, v in res.items() if k != "snr"), flush=True)
        summary[tag] = res
        del model, orig
        torch.cuda.empty_cache()
    print("\n===== SUMMARY  (fake-quant CE increase vs bf16, same holdout; W = every weight matrix,"
          " W+A = plus attention-projection inputs)")
    for t, r in summary.items():
        print(f"  {t:14s} bf16={r['bf16']:.4f}  " + "  ".join(f"{k} {r[k] - r['bf16']:+.4f}"
                                                            for k in r if k not in ("bf16", "snr")))
        print(f"  {'':14s} act rel err % " + "  ".join(f"{f} {100 * v:.3f}" for f, v in r["snr"].items()))
    print("QUANT_FINISHED")


if __name__ == "__main__":
    main()
