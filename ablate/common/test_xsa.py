"""XSA gate: the fused kernel matches eager, AND turning use_xsa on changes the model's output.

Two separate failures this catches, both of which would silently corrupt the A/B:
  1. fused != eager -> the XSA arm measures a broken kernel and XSA looks like it hurts quality.
  2. use_xsa=True does nothing -> both arms are the same model and the A/B reports a noise draw
     as "XSA is neutral". That is the per-expert-alpha failure mode again (see parity-vs-plumbing):
     a feature that is plumbed but inert produces plausible numbers and no error.

    python -m ablate.common.test_xsa
"""
from . import _paths  # noqa: F401

import torch
import torch.nn.functional as F

from src.modeling.attn.xsa import apply_xsa


def ref_xsa(y, v):
    """Y - (Y.Vn)Vn per head, in fp64, from the definition. GQA: V broadcasts over the query group."""
    B, H, S, D = y.shape
    Hkv = v.shape[1]
    g = H // Hkv
    yg = y.double().view(B, Hkv, g, S, D)
    vn = F.normalize(v.double()[:, :, -S:, :], dim=-1).unsqueeze(2)
    return (yg - (yg * vn).sum(-1, keepdim=True) * vn).reshape(B, H, S, D)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, Hkv, S, D = 2, 4, 2, 256, 128        # BiBo shape: 4 q-heads / 2 kv-heads, head_dim 128
    y = torch.randn(B, H, S, D, device=dev)
    v = torch.randn(B, Hkv, S, D, device=dev)

    r = ref_xsa(y, v)
    e = apply_xsa(y, v, enable_gqa=True)
    rel_e = ((e.double() - r).norm() / r.norm()).item()
    print(f"  eager  vs fp64 reference : rel {rel_e:.3e}")
    assert rel_e < 1e-6, f"src's eager apply_xsa does not match the definition ({rel_e:.2e})"

    # the rejection must be REAL: output orthogonal to V, and different from the input
    vn = F.normalize(v[:, :, -S:, :], dim=-1)
    resid = (e.view(B, Hkv, H // Hkv, S, D) * vn.unsqueeze(2)).sum(-1).abs().max().item()
    moved = (e - y).abs().max().item()
    print(f"  residual along V         : {resid:.3e}   (0 = fully removed)")
    print(f"  output moved vs no-XSA   : {moved:.4f}   (0 would mean XSA is INERT)")
    assert resid < 1e-4, "XSA left a component along V -- the rejection did not happen"
    assert moved > 1e-2, "XSA did not change the output -- it is plumbed but INERT"

    if dev != "cuda":
        print("\nPASS (eager only -- no CUDA, fused kernel not checked)")
        return

    from kernels.sm120.xsa import fused_xsa
    f = fused_xsa(y, v[:, :, -S:, :])
    rel_f = ((f.double() - r).norm() / r.norm()).item()
    rel_fe = ((f - e).norm() / e.norm()).item()
    print(f"  fused  vs fp64 reference : rel {rel_f:.3e}")
    print(f"  fused  vs eager          : rel {rel_fe:.3e}")
    assert rel_f < 1e-5, f"fused XSA does not match the definition ({rel_f:.2e})"
    assert rel_fe < 1e-5, f"fused XSA does not match src's eager path ({rel_fe:.2e})"

    # gradients too -- the arm trains through this
    ge, gf = [], []
    for fn, out in ((apply_xsa, ge), (lambda a, b, **k: fused_xsa(a, b[:, :, -a.shape[2]:, :]), gf)):
        yy = y.clone().requires_grad_(True); vv = v.clone().requires_grad_(True)
        fn(yy, vv, enable_gqa=True).square().sum().backward()
        out += [yy.grad, vv.grad]
    gy = ((gf[0] - ge[0]).norm() / ge[0].norm()).item()
    gv = ((gf[1] - ge[1]).norm() / ge[1].norm()).item()
    print(f"  grad_y fused vs eager    : rel {gy:.3e}")
    print(f"  grad_v fused vs eager    : rel {gv:.3e}")
    assert max(gy, gv) < 1e-4, f"fused XSA gradients differ from eager ({gy:.2e}, {gv:.2e})"

    print("\nPASS -- fused == eager == definition, forward and backward, and XSA is not inert.")


if __name__ == "__main__":
    main()
