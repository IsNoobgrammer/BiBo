"""Eager radial NormSiLU: correct value AND a live gradient on theta, on the REAL module.

The value check alone is not enough. `radial_theta` is a separate nn.Parameter that the eager
forward has to reach into explicitly, and the failure mode is that it silently doesn't -- the layer
then computes normsilu (radial's p->0 floor), produces plausible numbers, trains fine, and the
exponent never moves. That is exactly how per-expert alpha shipped inert once before, so this
asserts the gradient reaches theta and that changing theta changes the output.

This drives BiBoFusedExperts itself rather than a hand-written replica: a replica can agree with the
reference while the real module does something else entirely.

CPU-only, no Triton: this is the path that has to work where the kernel cannot run.
    python -m ablate.common.test_eager_radial
"""
from . import _paths  # noqa: F401

import torch
import torch.nn.functional as F

from src.configuration_bibo import BiBoConfig
from src.modeling.ffn.moe import BiBoFusedExperts, _NORMSILU_EPS


def _ref(gate, up, theta_e):
    """r^p * SiLU(gate/r) * up, straight from the definition, in fp64."""
    g = gate.double()
    r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)
    p = torch.sigmoid(theta_e.double())
    return (r.pow(p) * F.silu(g / r)) * up.double()


def main():
    torch.manual_seed(0)
    E, I, H, N = 4, 64, 32, 40
    cfg = BiBoConfig(hidden_size=H, num_attention_heads=4, num_key_value_heads=2,
                     moe_intermediate_size=I, num_routed_experts=E, special_expert_pairs=0,
                     pos_identity_expert=False, neg_identity_expert=False,
                     num_experts_per_tok=2)
    m = BiBoFusedExperts(cfg)
    assert m.num_glu_experts == E, f"expected {E} GLU experts, got {m.num_glu_experts}"
    assert hasattr(m, "radial_theta"), "BiBoFusedExperts has no radial_theta -- radial cannot learn"

    with torch.no_grad():
        m.radial_theta.copy_(torch.tensor([-2.0, 0.0, 1.5, 3.0]))     # p = .12 / .50 / .82 / .95
    x = torch.randn(N, H)

    def one_expert(e):
        """Route every token to expert e with weight 1 -- isolates that expert's activation."""
        idx = torch.full((N, 1), e, dtype=torch.long)
        w = torch.ones(N, 1)
        return m(x, idx, w)

    worst = 0.0
    for e in range(E):
        out = one_expert(e)
        gate, up = F.linear(x, m.gate_up_proj[e]).chunk(2, dim=-1)
        ref = _ref(gate, up, m.radial_theta[e]) @ m.down_proj[e].double().T
        rel = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, rel)
        print(f"  expert {e}  p={torch.sigmoid(m.radial_theta[e]).item():.3f}  rel err {rel:.2e}")
    assert worst < 1e-5, f"value mismatch vs the fp64 reference: {worst:.2e}"

    # (1) theta must RECEIVE gradient -- the inert-parameter check
    m.zero_grad()
    sum(one_expert(e).square().sum() for e in range(E)).backward()
    gt = m.radial_theta.grad
    print(f"\n  d(loss)/d(theta) = {[round(v, 4) for v in gt.tolist()]}")
    assert gt is not None and (gt.abs() > 1e-8).all(), \
        f"theta got NO gradient -- eager is silently running normsilu: {gt}"

    # (2) theta must CHANGE the output -- guards against a gradient that leads nowhere
    with torch.no_grad():
        base = one_expert(0).clone()
        m.radial_theta[0] += 2.0
        moved = (one_expert(0) - base).abs().max().item()
    print(f"  moving theta[0] by +2.0 changes the output by {moved:.4f}")
    assert moved > 1e-4, "theta does not affect the output -- radial is plumbed but INERT"

    # (3) p -> 0 must reproduce normsilu exactly (radial's floor is a real special case)
    with torch.no_grad():
        m.radial_theta[0] = -40.0                     # sigmoid(-40) = 0 -> r^0 = 1
        got = one_expert(0)
        gate, up = F.linear(x, m.gate_up_proj[0]).chunk(2, dim=-1)
        g = gate.float()
        ns = F.silu(g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NORMSILU_EPS)) * up
        ns = ns @ m.down_proj[0].T
    err = (got - ns).abs().max().item()
    print(f"  p->0 vs normsilu: max abs diff {err:.2e}")
    assert err < 1e-4, f"radial at p=0 is not normsilu ({err:.2e})"

    print("\nPASS -- eager radial on BiBoFusedExperts: values match fp64, theta gets gradient, "
          "theta moves the output, and p->0 is exactly normsilu.")


if __name__ == "__main__":
    main()
