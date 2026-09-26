"""Expert GEMM-input quantizability + full W4A16 / W4A8 / W4A4 (+Hadamard-16) PTQ on checkpoints.

    python -m ablate.tools.quant_probe_experts <result.json> [...] [--seqs 32] [--stat_seqs 2]

quant_probe.py cannot see the expert GEMM inputs: they live inside the fused Triton MoE kernel.
Here the checkpoint is loaded with the `moe` / `megakernel` patches OFF, so experts run the eager
src loop, whose two GEMMs go through `F.linear` -- the module's `F` is swapped for a shim that
records or fake-quantizes the GEMM input (gate_up input = post-norm hidden; down_proj input =
act(gate) * up). Attention q/k/v/o inputs are fake-quantized with forward pre-hooks. The router
and lm_head stay bf16.

Per layer, per tensor: amax / median|x| (the thread's table) and the UNDERFLOW rate -- fraction of
nonzero inputs that quantize to exactly 0 -- in NVFP4 and MXFP4, with and without Hadamard-16.

Hadamard: with H = blockdiag(H16), symmetric orthogonal, x W^T = (xH)(WH)^T, and
q(xH)H^T (q(WH)H^T)^T = q(xH) q(WH)^T, so "rotate, quantize, rotate back" applied to x and W
separately is exactly a Hadamard-rotated quantized GEMM.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import importlib
import json
import os
import tempfile
import torch
import torch.nn.functional as TF

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from ablate.tools.quant_probe import block_fq, _mats, _set, holdout_loss

_moe_src = importlib.import_module("src.modeling.ffn.moe")
_moe_k = importlib.import_module("kernels.sm75.moe")


def _h16(device):
    h = torch.ones(1, 1)
    while h.shape[0] < 16:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / 4.0).to(device)                     # 16^-1/2: orthogonal AND symmetric


def make_q(fmt, had):
    """Fake-quantizer along the last dim; had=True rotates 16-blocks by H16 first and back after."""
    if fmt is None:
        return None
    if not had:
        return lambda t: block_fq(t, fmt)

    def q(t):
        H = _h16(t.device)
        s = t.shape
        r = (t.float().reshape(*s[:-1], s[-1] // 16, 16) @ H).reshape(s)
        r = block_fq(r, fmt).float()
        return (r.reshape(*s[:-1], s[-1] // 16, 16) @ H).reshape(s).to(t.dtype)
    return q


class Shim:
    """Stands in for torch.nn.functional inside src.modeling.ffn.moe."""
    def __init__(self, hidden, layers):
        self.hidden, self.layer, self.aq, self.rec = hidden, 0, None, None
        self.calls = 0
        for i, l in enumerate(layers):
            l.register_forward_pre_hook(lambda _m, _a, i=i: setattr(self, "layer", i))

    def __getattr__(self, k):
        return getattr(TF, k)

    def linear(self, x, w, b=None):
        kind = "gate_up" if w.shape[-1] == self.hidden else "down"
        if self.rec is not None:
            self.rec.setdefault((self.layer, kind), []).append(x.detach().float())
        if self.aq is not None:
            self.calls += 1
            x = self.aq(x)
        return TF.linear(x, w, b)


def eager_result(rj):
    """Copy of the result json with the fused MoE patches stripped, so experts run the src loop."""
    res = json.load(open(rj))
    pl = [p for p in res["config"]["patches"].split(",") if p.strip() not in ("moe", "megakernel")]
    res["config"]["patches"] = ",".join(pl)
    f = tempfile.NamedTemporaryFile("w", suffix="_result.json", delete=False)
    json.dump(res, f)
    f.close()
    return f.name


def attn_hooks(model, st):
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m.register_forward_pre_hook(lambda _m, a: None if st["aq"] is None
                                        else (st["aq"](a[0]),) + tuple(a[1:]))


@torch.no_grad()
def input_stats(model, shim, holdout, amp):
    """{(layer, 'gate_up'|'down'): {amax/med, uf_<fmt>[_h]}} over every expert call of the batch.
    Underflow = nonzero input that quantizes to exactly 0, measured in the domain the quantizer
    sees (the rotated one under Hadamard)."""
    shim.rec = {}
    holdout_loss(model, holdout, amp)
    rec, shim.rec = shim.rec, None
    out = {}
    for (L, kind), xs in sorted(rec.items()):
        x = torch.cat(xs)                                   # (tokens routed, K)
        a = x.abs().flatten()
        idx = torch.randperm(a.numel(), device=a.device)[:1_000_000]
        row = {"amax/med": (a.max() / a[idx].median().clamp_min(1e-30)).item()}
        for had in (False, True):
            src = _rot(x) if had else x
            m = src != 0
            for fmt in ("nvfp4", "mxfp4"):
                q = block_fq(src, fmt)
                row[f"uf_{fmt}{'_h' if had else ''}"] = ((q == 0) & m).sum().item() / max(m.sum().item(), 1)
        out[(L, kind)] = row
    return out


def _rot(t):
    s = t.shape
    return (t.float().reshape(*s[:-1], s[-1] // 16, 16) @ _h16(t.device)).reshape(s)


CONFIGS = [  # name, weight quantizer spec, activation quantizer spec
    ("W4A16 nvfp4", ("nvfp4", False), None),
    ("W4A8 nvfp4/mxfp8", ("nvfp4", False), ("mxfp8", False)),
    ("W4A4 nvfp4", ("nvfp4", False), ("nvfp4", False)),
    ("W4A4 nvfp4 +H16", ("nvfp4", True), ("nvfp4", True)),
    ("W4A4 mxfp4", ("mxfp4", False), ("mxfp4", False)),
    ("W4A4 mxfp4 +H16", ("mxfp4", True), ("mxfp4", True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--stat_seqs", type=int, default=2)
    a = ap.parse_args()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    holdout, summary = None, {}
    for rj in a.results:
        _moe_k._CAST_CACHE.clear()
        model, c = load_from_result(eager_result(rj))
        tag = c.run_tag or os.path.basename(rj)
        if holdout is None:
            holdout = _val.build_holdout(c.dataset, c.seq_len, a.seqs, "cuda")
        shim = Shim(model.config.hidden_size, model.model.layers)
        _moe_src.F = shim
        st = {"aq": None}
        attn_hooks(model, st)
        print(f"\n===== {tag}  (eager experts)", flush=True)
        base = holdout_loss(model, holdout, amp)
        _fr = json.load(open(rj)).get("final_report", {})
        print(f"  sanity: eager bf16 CE {base:.4f} | fused-path probe was ~3.39-3.40, in-training "
              f"ctx1024 {_fr.get('ctxabl/ctx1024', float('nan')):.4f}", flush=True)
        stats = input_stats(model, shim, holdout[:a.stat_seqs], amp)
        print(f"  {'L':>2s} {'tensor':8s} {'amax/med':>9s} {'uf nvfp4':>9s} {'+H16':>7s} {'uf mxfp4':>9s} {'+H16':>7s}")
        for (L, kind), r in stats.items():
            print(f"  {L:2d} {kind:8s} {r['amax/med']:9.1f} {100 * r['uf_nvfp4']:8.2f}% {100 * r['uf_nvfp4_h']:6.2f}%"
                  f" {100 * r['uf_mxfp4']:8.2f}% {100 * r['uf_mxfp4_h']:6.2f}%", flush=True)
        orig = {n: p.detach().clone() for n, p in _mats(model)}
        res = {"bf16": base}
        for name, wq, aq in CONFIGS:
            fw = make_q(*wq)
            _set(model, {n: fw(w).float() for n, w in orig.items()})
            fa = make_q(*aq) if aq else None
            shim.aq, st["aq"], shim.calls = fa, fa, 0
            res[name] = holdout_loss(model, holdout, amp)
            assert fa is None or shim.calls > 0, "expert activation shim never fired"
            shim.aq = st["aq"] = None
            _set(model, orig)
            print(f"  {name:20s} CE {res[name]:.4f} ({res[name] - base:+.4f})", flush=True)
        summary[tag] = res
        _moe_src.F = TF
        del model, orig
        torch.cuda.empty_cache()
    print("\n===== SUMMARY  (CE increase vs eager bf16; W = every weight matrix except embed/lm_head;"
          " A = attention q/k/v/o + expert gate_up/down inputs; router bf16)")
    names = [n for n, _, _ in CONFIGS]
    print(f"  {'':14s} {'bf16':>7s} " + " ".join(f"{n:>18s}" for n in names))
    for t, r in summary.items():
        print(f"  {t:14s} {r['bf16']:7.4f} " + " ".join(f"{r[n] - r['bf16']:+18.4f}" for n in names))
    print("QEXP_FINISHED")


if __name__ == "__main__":
    main()
