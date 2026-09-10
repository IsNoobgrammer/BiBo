"""Why does a SPARSE layer 0 extrapolate differently from a dense or all-active one?

    python -m ablate.tools.length_probe <result.json> [--lens 1024,2048,4095] [--seqs 8]
                                        [--zero_bias 0] [--dataset DIR]

Runs a FROZEN checkpoint on held-out text at several context lengths and reports, per layer, both
the loss and what the router did. Three questions, one pass:

  1. DOES ROUTING ADAPT TO LENGTH.  Compare each layer's expert-load distribution at 1024 against
     4095 (rank correlation, total-variation distance, boundary gap, entropy). A router whose
     output is a function of the INPUT should route differently when the input distribution
     shifts; one that has degenerated into a function of the training distribution should not.

  2. DOES AN ALL-ACTIVE LAYER MOVE AT ALL.  With top_k == E the load is flat by construction, so
     the same comparison is run on the mixing WEIGHTS. The seed-23 ensemble ended with weight
     entropy 0.987 -- near-uniform -- so the prediction is that its weights barely move with
     length, i.e. it behaves like the dense layer it matched on extrapolation.

  3. IS THE BALANCING BIAS THE LIABILITY.  `--zero_bias L` zeroes layer L's router bias for the
     eval only and re-measures. The bias is fitted to the TRAIN token distribution and cannot
     respond to anything at eval; the coarse-L0 arm, whose router logits never left log(E) so the
     bias placed every token, had delta_ctx4095 0.757 against the baseline's 0.054. If a stale
     bias is what costs extrapolation, zeroing it should IMPROVE long-context CE.

WHY THE MECHANISM IS OPEN. Capacity does not explain the ordering: L0 holds 75.5M params in BOTH
all-MoE (64x768 top-6) and coarse (32x1536 top-3), and those two sit at opposite extremes of
delta_ctx4095. What tracks the ordering is how the selection is made -- input-driven, absent, or
bias-driven -- and that is what this measures rather than assumes.

Nothing here trains. It loads, evaluates, and prints.
"""
from ablate.common import _paths  # noqa: F401

import argparse
import json
import re

import torch

from ablate.common.report_ckpt import load_from_result

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Probe:
    """Per-layer router capture. Keyed by the MODEL's layer index, not discovery order."""

    def __init__(self, model):
        self.acc, self._h = {}, []
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "BiBoMoERouter":
                m = re.search(r"layers\.(\d+)\.", name)
                self._h.append(mod.register_forward_hook(self._mk(int(m.group(1)))))

    def _mk(self, i):
        @torch.no_grad()
        def hook(mod, args, out):
            E, k = mod.num_routed_experts, mod.top_k
            scores = torch.sigmoid(mod.router_logits(args[0]))
            idx = out[0].reshape(-1, out[0].shape[-1])
            wgt = out[1].reshape(-1).float()
            a = self.acc.setdefault(i, {"c": torch.zeros(E, dtype=torch.float64),
                                        "w": torch.zeros(E, dtype=torch.float64),
                                        "gap": 0.0, "n": 0, "E": E, "k": k})
            a["c"] += torch.bincount(idx.reshape(-1).cpu(), minlength=E).double()
            a["w"] += torch.bincount(idx.reshape(-1).cpu(), weights=wgt.cpu(),
                                     minlength=E).double()
            if k < E:
                tk = scores.topk(k + 1, dim=-1).values
                a["gap"] += (tk[..., k - 1] - tk[..., k]).mean().item()
            a["n"] += 1
        return hook

    def take(self):
        out = {}
        for i, a in self.acc.items():
            c = a["c"] / a["c"].sum().clamp_min(1)
            w = a["w"] / a["w"].sum().clamp_min(1e-12)
            out[i] = {"load": c, "weight": w, "E": a["E"], "k": a["k"],
                      "gap": a["gap"] / max(a["n"], 1)}
        self.acc = {}
        return out

    def close(self):
        for h in self._h:
            h.remove()


def _entropy(p):
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


@torch.no_grad()
def _run(model, probe, dataset, C, n_seqs, target_window=512, row_tokens=4096, pad_id=0, chunk=4):
    """CE on the SAME targets with C tokens of context, plus the router state on those inputs.

    This is `final_report.context_ablation` verbatim in construction, which matters: ctxabl is the
    metric the whole extrapolation claim rests on, and it scores identical target tokens while
    varying only the visible history. An earlier version of this probe scored DIFFERENT targets at
    each length -- CE then falls with length for the trivial reason that later tokens have more
    context, and it is not comparable to delta_ctx4095 at all.
    """
    from ablate.common.final_report import _holdout_rows
    batch = _holdout_rows(dataset, row_tokens, n_seqs, DEV)
    assert batch is not None, "no held-out shard: this needs a local corpus directory"
    C = min(C, row_tokens - 1)
    a = row_tokens - 1 - C
    tot = cnt = 0.0
    for i in range(0, batch.shape[0], chunk):
        b = batch[i:i + chunk]
        inp, tgt = b[:, a:a + C], b[:, a + 1:a + C + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            logits = model(inp).logits
        w = min(target_window, logits.shape[1])
        ce = torch.nn.functional.cross_entropy(
            logits[:, -w:].float().reshape(-1, logits.shape[-1]),
            tgt[:, -w:].reshape(-1), ignore_index=int(pad_id))
        tot += float(ce) * b.shape[0] * w
        cnt += b.shape[0] * w
        del logits
    return tot / max(cnt, 1), probe.take()


def _compare(a, b, key):
    """Rank correlation and total-variation distance between two per-expert distributions."""
    x, y = a[key], b[key]
    tv = float((x - y).abs().sum()) / 2.0                  # 0 = identical, 1 = disjoint
    xr = x.argsort().argsort().double()
    yr = y.argsort().argsort().double()
    xr, yr = xr - xr.mean(), yr - yr.mean()
    rho = float((xr * yr).sum() / (xr.norm() * yr.norm()).clamp_min(1e-12))
    return rho, tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--lens", default="1024,2048,4095")
    ap.add_argument("--seqs", type=int, default=32)   # ctxabl converges at 32 rows
    ap.add_argument("--dataset", default="/home/marimo/work/data/bibo_mix")
    ap.add_argument("--zero_bias", type=int, default=-1,
                    help="layer whose router bias to zero for the eval (-1 = leave it alone)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    lens = [int(v) for v in a.lens.split(",")]
    model, cfg = load_from_result(a.result, device=DEV)
    model.eval()

    if a.zero_bias >= 0:
        b = model.model.layers[a.zero_bias].mlp.gate.bias
        print(f"[probe] zeroing layer {a.zero_bias} router bias for the eval "
              f"(was |b|={b.norm().item():.4f}, spread {b.max().item() - b.min().item():.4f})")
        b.data.zero_()

    probe = _Probe(model)
    res = {}
    for n in lens:
        ce, st = _run(model, probe, a.dataset, n, a.seqs)
        res[n] = (ce, st)
        print(f"\n=== ctx {n}: CE {ce:.4f}")
        print(f"{'layer':>6}{'E':>5}{'k':>4}{'gap':>9}{'load_ent':>10}{'wgt_ent':>9}{'max_wgt':>9}")
        for i in sorted(st):
            s = st[i]
            print(f"{i:>6}{s['E']:>5}{s['k']:>4}{s['gap']:>9.4f}"
                  f"{_entropy(s['load']) / __import__('math').log(s['E']):>10.4f}"
                  f"{_entropy(s['weight']) / __import__('math').log(s['E']):>9.4f}"
                  f"{float(s['weight'].max()):>9.4f}")
    probe.close()

    base, far = lens[0], lens[-1]
    print(f"\n=== DOES ROUTING MOVE WITH LENGTH?  {base} vs {far}")
    print(f"  CE {res[base][0]:.4f} -> {res[far][0]:.4f}   delta {res[far][0] - res[base][0]:+.4f}")
    print(f"{'layer':>6}{'load_rho':>10}{'load_tv':>9}{'wgt_rho':>9}{'wgt_tv':>8}{'d_gap':>9}")
    for i in sorted(res[base][1]):
        lo, hi = res[base][1][i], res[far][1][i]
        lr, lt = _compare(lo, hi, "load")
        wr, wt = _compare(lo, hi, "weight")
        print(f"{i:>6}{lr:>10.3f}{lt:>9.4f}{wr:>9.3f}{wt:>8.4f}{hi['gap'] - lo['gap']:>+9.4f}")
    print("\n  rho 1.0 / tv 0.0 = the router does the SAME thing at both lengths.")
    print("  A layer whose routing is input-driven should show tv well above 0 at 4x the length.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({str(n): {"ce": res[n][0],
                                "layers": {str(i): {"load": res[n][1][i]["load"].tolist(),
                                                    "weight": res[n][1][i]["weight"].tolist(),
                                                    "gap": res[n][1][i]["gap"]}
                                           for i in res[n][1]}} for n in res}, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
