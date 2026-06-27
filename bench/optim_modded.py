"""
Turbo-Muon ("modded" Muon) — opt-in NS variant, enabled via `--modded-muon`.

Turbo-Muon (arXiv:2512.04632, Boissin et al.; ref impl github.com/thib-s/flash-newton-schulz):
AOL-preconditioned Newton-Schulz with per-iteration Polar-Express coefficients. 4 iters + AOL
≈ 5-iter standard NS (val-loss Δ±0.002 on NanoGPT 144M), ~20% fewer NS GEMMs.

This SUBCLASSES the production Muon (bench/optim.py) and overrides ONLY the Newton-Schulz
function. Everything else is inherited unchanged: momentum→orthogonalize order, the Moonlight
0.2·√max(A,B) consistent-RMS scaling, decoupled weight decay, per-expert 3D batching, compile_ns.
So this keeps BiBo's Moonlight recipe + muon_lr=3e-4 (the "option-1 graft"), NOT the paper's
Jordan aspect-ratio scale + lr~0.02.

⚠ UNVALIDATED COMBO: the paper pairs these coeffs with Jordan's scaling, not Moonlight's. Turbo's
output deliberately spreads singular values over ~[0.5,1.5] (mean ~1), so the 0.2·√max(A,B) scale
(which assumes SVs≈1) stays correct only to ~4%. The paper never tested MoE / 3D experts. Run the
__main__ self-check (SV distribution + RMS(|Δp|)/lr attribution must stay ~0.2, flat across shapes)
AND a few real training steps before trusting this.

T4 (sm_75): fp32 NS (no bf16 tensor cores). compile_ns still applies. PyTorch fallback only — the
paper's 2.8× needs the custom Triton kernel; the iteration-drop alone is the ~20% algorithmic slice.
"""
import torch

from optim import Muon  # lazy at call sites; safe (optim is fully loaded before create_optimizer runs)


# Per-iteration Polar-Express coefficients (Turbo-Muon ref, verbatim). 5 tuples; "turbo" takes the
# LAST `num_iters`. Aggressive early (expand small SVs fast), settling late. NOT the fixed
# (3.4445,-4.7750,2.0315) quintic — fewer iters needs these per-step-varying coeffs to converge.
_TURBO_NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def turbo_newton_schulz(G, num_iters=4, eps=1e-7):
    """AOL-preconditioned Newton-Schulz with per-iteration coeffs (Turbo-Muon).

    AOL = per-row rescale by rsqrt(|XXᵀ|.sum(rows)); FUSED into iteration 0 (reuses that iteration's
    Gram matmul → no extra GEMM) and REPLACES Frobenius normalization. Drives SVs below 1 cheaply so
    NS converges in one fewer step. Batches over leading dims: 2D weights (unsqueezed) + 3D stacked
    experts (per-slice) — same convention as optim.py's newton_schulz_iteration. fp32 for stability.
    """
    coeffs = _TURBO_NS_COEFFS[-num_iters:]
    X = G.float()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)                      # (1,A,B) — unify 2D + batched (E,A,B)
    transposed = X.size(1) > X.size(2)          # iterate on the smaller Gram
    if transposed:
        X = X.transpose(1, 2)
    # NO Frobenius normalization here — AOL (inside iter 0) conditions X instead.
    for i, (a, b, c) in enumerate(coeffs):
        A = X @ X.transpose(1, 2)
        if i == 0:                              # AOL preconditioning, reuses A (no extra matmul)
            s = torch.rsqrt(A.abs().sum(dim=-1).clamp_min(eps))   # per-row (per-expert) scale
            X = X * s.unsqueeze(-1)
            A = A * s.unsqueeze(-1) * s.unsqueeze(-2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(G.dtype)


class ModdedMuon(Muon):
    """Muon with the Turbo-Muon AOL Newton-Schulz (see module docstring). Keeps Moonlight scaling."""

    def __init__(self, *args, ns_steps=4, compile_ns=False, **kwargs):
        super().__init__(*args, compile_ns=compile_ns, **kwargs)
        # Override the inherited standard-quintic NS with the turbo AOL NS. Super may have wrapped the
        # standard NS in torch.compile, but we replace the handle before any call — and torch.compile
        # is lazy (compiles on first invocation) — so the standard NS never actually compiles.
        base = lambda g: turbo_newton_schulz(g, num_iters=ns_steps)
        self._ns = torch.compile(base, dynamic=False) if compile_ns else base
        self.ns_steps = ns_steps


if __name__ == "__main__":
    # Self-check (Rules 1-2): runs locally, plain torch fp32, no compile needed.
    #   (1) NaN-free + SV distribution of the turbo NS output (expect mean~1, spread ~[0.5,1.5]).
    #   (2) Moonlight scaling survives the graft: RMS(|Δp|)/lr ~0.2, FLAT across shapes (the property
    #       that makes a single AdamW-band muon_lr correctly attributed). A deviating shape => the
    #       graft broke the RMS-consistency the 0.2·√max(A,B) scale relies on.
    torch.manual_seed(0)
    shapes = [(512, 512), (256, 512), (1024, 512), (9, 1536, 512), (9, 512, 768)]

    print("turbo_newton_schulz — SV distribution per slice (expect mean~1, spread ~0.5-1.5):")
    sv_ok = True
    for sh in shapes:
        G = torch.randn(*sh)
        Y = turbo_newton_schulz(G, num_iters=4)
        assert not Y.isnan().any(), f"NaN in NS output for {sh}"
        sv = torch.linalg.svdvals(Y if Y.ndim == 3 else Y.unsqueeze(0))
        sv_ok &= 0.85 <= sv.mean().item() <= 1.15
        print(f"  {str(sh):16s} SV mean={sv.mean():.3f} min={sv.min():.3f} max={sv.max():.3f}")

    print("\nLR attribution — RMS(|Δp|)/lr (expect ~0.20, flat across shapes; wd=0 to isolate):")
    lr, attr_ok = 1e-3, True
    for sh in shapes:
        p = torch.nn.Parameter(torch.randn(*sh))
        opt = ModdedMuon([p], lr=lr, momentum=0.95, weight_decay=0.0, ns_steps=4)
        p.grad = torch.randn(*sh)
        before = p.detach().clone()
        opt.step()
        rms = ((p.detach() - before).abs() / lr).pow(2).mean().sqrt().item()
        attr_ok &= 0.17 <= rms <= 0.23
        print(f"  {str(sh):16s} RMS(|Δp|)/lr = {rms:.4f}")

    print(f"\n{'PASS' if (sv_ok and attr_ok) else 'FAIL'}: "
          f"SVs mean~1 ({sv_ok}), attribution flat ~0.2 ({attr_ok}).")
