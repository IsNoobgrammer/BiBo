"""Self-check for the new router gates: ranges, monotonicity, finite grads, and that the
ablate patch and the model agree (they were separate implementations until now)."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.configuration_bibo import GATE_TYPES, SIGNED_GATES
from src.modeling.ffn.router import gate_scores
from ablate.common import patches

x = torch.linspace(-30, 30, 20001, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
EXPECT = {"sigmoid": (0.0, 1.0), "tsig": (0.0, 0.7616), "sigtanh": (0.2689, 0.7311),
          "sqsp": (0.0, None), "situ": (-0.2786, 1.0)}

for g in GATE_TYPES:
    if g == "softmax":
        continue                       # cross-expert, not a pointwise map; skip the pointwise checks
    y = gate_scores(x, g)
    (gr,) = torch.autograd.grad(y.sum(), x, retain_graph=True)
    lo, hi = EXPECT[g]
    assert torch.isfinite(y).all(), f"{g}: non-finite score"
    assert torch.isfinite(gr).all(), f"{g}: non-finite grad"          # sqsp's sqrt is the risk here
    assert y.min() >= lo - 1e-4, f"{g}: min {y.min():.4f} < {lo}"
    if hi is not None:
        assert y.max() <= hi + 1e-4, f"{g}: max {y.max():.4f} > {hi}"
    if g not in SIGNED_GATES:
        assert (y >= 0).all(), f"{g}: negative score breaks div-sum normalization"
        d = y[0, 1:] - y[0, :-1]
        assert (d >= -1e-7).all(), f"{g}: NOT monotonic -> top-k ordering would be scrambled"
    print(f"  {g:<9} range [{y.min():+.4f}, {y.max():+.4f}]  max|grad| {gr.abs().max():.4f}  ok")

# the patch must produce EXACTLY what the model produces, for every gate
z = torch.randn(64, 8)
for g in GATE_TYPES:
    patches.ROUTER_GATE = g
    assert torch.equal(patches._gate_scores(z, "sigmoid"), gate_scores(z, g)), f"{g}: patch != model"
print(f"  patch == model for all {len(GATE_TYPES)} gates")

# div-sum flatness ordering claimed in the analysis: sigmoid is the most peaked of the bounded gates
lg = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
r = {g: (lambda w: (w.max() / w.min()).item())(gate_scores(lg, g) / gate_scores(lg, g).sum())
     for g in ("sigmoid", "tsig", "sigtanh")}
assert r["sigmoid"] > r["tsig"] > r["sigtanh"], f"flatness ordering broke: {r}"
print("  div-sum weight ratio  " + "  ".join(f"{k}={v:.2f}" for k, v in r.items()))
print("PASS")
