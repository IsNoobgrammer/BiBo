"""Eager radial NormSiLU: correct value AND a live gradient on theta.

The value check alone is not enough. `situ_alpha` is a separate nn.Parameter that the eager forward
has to reach into explicitly, and the failure mode is that it silently doesn't -- the layer then
computes normsilu (radial's p->0 floor), produces plausible numbers, trains fine, and the exponent
never moves. That is exactly how per-expert alpha shipped inert once before, so this asserts the
gradient reaches theta and that changing theta changes the output.

CPU-only, no Triton: this is the path that has to work where the kernel cannot run.
    python -m ablate.common.test_eager_radial
"""
from . import _paths  # noqa: F401

import torch
import torch.nn.functional as F

from src.modeling.ffn.moe import _NORMSILU_EPS, _POLYGLU_ACTIVATIONS


def _ref(gate, up, theta_e):
    """r^p * SiLU(gate/r) * up, straight from the definition, in fp64."""
    g = gate.double()
    r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
    p = torch.sigmoid(theta_e.double())
    return (r.pow(p) * F.silu(g / r)) * up.double()


def main():
    assert set(_POLYGLU_ACTIVATIONS) == {"radial"}, \
        f"expected the radial-only menu, got {_POLYGLU_ACTIVATIONS}"
    torch.manual_seed(0)
    E, I, D, N = 4, 64, 32, 40

    class Tiny(torch.nn.Module):
        """Just the pieces the eager expert branch touches."""
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.randn(E, 2 * I, D) * 0.3)
            self.situ_alpha = torch.nn.Parameter(torch.zeros(E))   # theta; p = sigmoid(0) = 0.5

        def one_expert(self, x, e):
            gate, up = F.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
            g = gate.float()
            r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
            act = F.silu(g / r) * r.pow(torch.sigmoid(self.situ_alpha[e].float()))
            return act.to(gate.dtype) * up

    m = Tiny()
    with torch.no_grad():
        m.situ_alpha.copy_(torch.tensor([-2.0, 0.0, 1.5, 3.0]))     # p = .12 / .50 / .82 / .95
    x = torch.randn(N, D)

    worst = 0.0
    for e in range(E):
        out = m.one_expert(x, e)
        gate, up = F.linear(x, m.gate_up_proj[e]).chunk(2, dim=-1)
        rel = ((out.double() - _ref(gate, up, m.situ_alpha[e])).abs().max()
               / _ref(gate, up, m.situ_alpha[e]).abs().max()).item()
        worst = max(worst, rel)
        print(f"  expert {e}  p={torch.sigmoid(m.situ_alpha[e]).item():.3f}  rel err {rel:.2e}")
    assert worst < 1e-6, f"value mismatch vs the fp64 reference: {worst:.2e}"

    # (1) theta must RECEIVE gradient -- the inert-parameter check
    m.zero_grad()
    sum(m.one_expert(x, e).square().sum() for e in range(E)).backward()
    gt = m.situ_alpha.grad
    print(f"\n  d(loss)/d(theta) = {[round(v, 4) for v in gt.tolist()]}")
    assert gt is not None and (gt.abs() > 1e-8).all(), \
        f"theta got NO gradient -- eager is silently running normsilu: {gt}"

    # (2) theta must CHANGE the output -- guards against a gradient that leads nowhere
    with torch.no_grad():
        base = m.one_expert(x, 0).clone()
        m.situ_alpha[0] += 2.0
        moved = (m.one_expert(x, 0) - base).abs().max().item()
    print(f"  moving theta[0] by +2.0 changes the output by {moved:.4f}")
    assert moved > 1e-3, "theta does not affect the output -- radial is plumbed but INERT"

    # (3) p -> -inf must reproduce normsilu exactly (radial's floor is a real special case)
    with torch.no_grad():
        m.situ_alpha[0] = -40.0                       # sigmoid(-40) = 0 -> r^0 = 1
        got = m.one_expert(x, 0)
        gate, up = F.linear(x, m.gate_up_proj[0]).chunk(2, dim=-1)
        g = gate.float()
        ns = F.silu(g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)) * up
    err = (got - ns).abs().max().item()
    print(f"  p->0 vs normsilu: max abs diff {err:.2e}")
    assert err < 1e-5, f"radial at p=0 is not normsilu ({err:.2e})"

    print("\nPASS -- eager radial: values match fp64, theta gets gradient, theta moves the output, "
          "and p->0 is exactly normsilu.")


if __name__ == "__main__":
    main()
