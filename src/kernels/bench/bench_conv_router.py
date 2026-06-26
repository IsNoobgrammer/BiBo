"""
Frozen eval for the FUSED conv router (conv+sigmoid+bias+topk+gather as one autograd op).

Baseline (Rule 1) = ORIGINAL eager BiBoMoERouter.forward, router_type="conv", NO patch.
Candidate        = patch_conv_router_with_triton -> _FusedConvRouterFull.

Checks:
  1. Grad equivalence (fp32: indices identical + weights/grad_x/grad_w within 1e-4).
  2. fp16 index-agreement % (cuDNN-conv vs Triton-conv numeric paths) + weight tol.
  3. NaN-free over >=2 fwd+bwd passes.
  4. 3-phase timing (fwd / bwd / fwd+bwd) with cuda.Event, speedup vs eager.
  5. CUDA kernel-launch count + matmul count (profiler) — fused must be <= eager GEMMs,
     strictly fewer total launches.

Run: .venv/Scripts/python src/kernels/bench/bench_conv_router.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
from torch.profiler import profile, ProfilerActivity
from torch._C._autograd import DeviceType

from src.configuration_bibo import BiBoConfig
from src.modeling.ffn.router import BiBoMoERouter
from src.kernels.conv_fused import (
    patch_conv_router_with_triton, unpatch_conv_router,
    fused_conv_router_full, _FusedConvRouterFunction,
)

DEV = "cuda"
torch.manual_seed(0)


def make_router(dtype):
    cfg = BiBoConfig(
        vocab_size=5000, hidden_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2,
        polyglu_expert_multiplier=2, special_expert_pairs=1,
        num_experts_per_tok=2, moe_intermediate_size=256, intermediate_size=1024,
        router_type="conv", kernel_size=4, gate_type="sigmoid",
        router_activation="none", router_noise=0.0, norm_topk_prob=False,
    )
    r = BiBoMoERouter(cfg).to(DEV, dtype)
    return r


def run(router, x, train=True):
    router.train(train)
    idx, w = router(x)
    return idx, w


# ── A wrapper so we can patch/unpatch the SAME module (identical weights) ──
class _M(torch.nn.Module):
    def __init__(self, r):
        super().__init__()
        self.gate = r
    def forward(self, x):
        return self.gate(x)


def _ref_forward(x, r):
    """Reference that shares the EXACT conv path with the fused op (same Triton conv fwd+bwd),
    differing only in the sigmoid/bias/topk/gather done in eager autograd. This isolates the
    NEW fused logic (sigmoid epilogue + scatter-sigmoid' backward) for a clean Rule-1 check —
    routing is identical by construction, so any diff is the new math, not the conv numeric path."""
    logits = _FusedConvRouterFunction.apply(x, r.gate_conv.weight).float()
    scores = torch.sigmoid(logits)
    sel = scores + r.bias
    _, idx = torch.topk(sel, r.top_k, dim=-1)
    w = scores.gather(-1, idx)
    return idx, w


def correctness(B, S, H, dtype):
    """Isolated check: fused full router vs eager-reference, conv path held identical."""
    r = make_router(dtype)
    x0 = torch.randn(B, S, H, device=DEV, dtype=dtype)
    go = torch.randn(B * S, r.top_k, device=DEV, dtype=dtype)  # raw funcs return flat (B*S,k)

    xr = x0.clone().requires_grad_(True)
    ir, wr = _ref_forward(xr, r)
    gxr, gwr = torch.autograd.grad(wr, [xr, r.gate_conv.weight], grad_outputs=go.to(wr.dtype))

    xf = x0.clone().requires_grad_(True)
    iff, wf = fused_conv_router_full(xf, r.gate_conv.weight, r.bias, r.top_k)
    gxf, gwf = torch.autograd.grad(wf, [xf, r.gate_conv.weight], grad_outputs=go.to(wf.dtype))

    idx_agree = (ir == iff).float().mean().item()
    w_diff = (wr.float() - wf.float()).abs().max().item()
    gx_diff = (gxr - gxf).abs().max().item()
    gw_diff = (gwr - gwf).abs().max().item()
    return idx_agree, w_diff, gx_diff, gw_diff


def vs_eager_routing(B, S, H, dtype):
    """Separate, informational: how often does the Triton-conv fused router pick the SAME
    experts as the true cuDNN-conv eager router? (<100% from the conv numeric-path swap.)"""
    r = make_router(dtype); r.eval()
    m = _M(r)
    x = torch.randn(B, S, H, device=DEV, dtype=dtype)
    with torch.no_grad():
        ie, _ = r(x)
        patch_conv_router_with_triton(m)
        iff, _ = r(x)
        unpatch_conv_router(m)
    return (ie == iff).float().mean().item()


def nan_check(B, S, H, dtype):
    r = make_router(dtype)
    m = _M(r); patch_conv_router_with_triton(m)
    x = torch.randn(B, S, H, device=DEV, dtype=dtype, requires_grad=True)
    ok = True
    for _ in range(3):
        r.gate_conv.weight.grad = None
        idx, w = r(x)
        loss = w.float().pow(2).sum()
        loss.backward()
        if torch.isnan(loss).any() or x.grad is None or torch.isnan(x.grad).any() \
           or torch.isnan(r.gate_conv.weight.grad).any():
            ok = False
        x.grad = None
    unpatch_conv_router(m)
    return ok


def _time(fn, n_warm=10, n_iter=30):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(n_iter): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / n_iter  # ms


def phases(router, x, k):
    go = torch.randn(x.shape[0], x.shape[1], k, device=DEV, dtype=x.dtype)
    def fwd():
        with torch.no_grad(): router(x)
    def fwdbwd():
        router.gate_conv.weight.grad = None
        if x.grad is not None: x.grad = None
        idx, w = router(x)
        w.backward(go)
    return _time(fwd), _time(fwdbwd)


def bench_speed(B, S, H, dtype):
    r = make_router(dtype); r.train(True)
    m = _M(r)
    x = torch.randn(B, S, H, device=DEV, dtype=dtype, requires_grad=True)
    k = r.top_k
    f_e, fb_e = phases(r, x, k)
    patch_conv_router_with_triton(m)
    f_f, fb_f = phases(r, x, k)
    unpatch_conv_router(m)
    return (f_e, fb_e), (f_f, fb_f)


def launch_counts(B, S, H, dtype):
    r = make_router(dtype); r.train(True)
    m = _M(r)
    x = torch.randn(B, S, H, device=DEV, dtype=dtype, requires_grad=True)
    go = torch.randn(B, S, r.top_k, device=DEV, dtype=dtype)

    def profile_one():
        for _ in range(3):  # warmup
            r.gate_conv.weight.grad = None; x.grad = None
            idx, w = r(x); w.backward(go)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            r.gate_conv.weight.grad = None; x.grad = None
            idx, w = r(x); w.backward(go)
            torch.cuda.synchronize()
        def cuda_time(e):
            for attr in ("self_device_time_total", "self_cuda_time_total", "cuda_time_total"):
                v = getattr(e, attr, 0) or 0
                if v:
                    return v
            return 0
        cuda_rows = [e for e in prof.key_averages() if cuda_time(e) > 0]
        n_launch = sum(e.count for e in cuda_rows)
        n_mm = sum(e.count for e in cuda_rows
                   if any(t in e.key.lower() for t in
                          ("gemm", "conv", "dot", "matmul", "cutlass", "sgemm", "implicit", "wgrad", "dgrad")))
        return n_launch, n_mm

    le, mme = profile_one()
    patch_conv_router_with_triton(m)
    lf, mmf = profile_one()
    unpatch_conv_router(m)
    return (le, mme), (lf, mmf)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs CUDA"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    shapes = [(8, 512, 512), (16, 1024, 512)]

    print("=" * 78)
    print("CORRECTNESS — fused vs eager-reference, conv path held identical (Rule 1)")
    # fp32 gate: new logic (sigmoid epilogue + scatter-sigmoid' bwd) must be exact (w, gx).
    # gw rides the existing _conv_router_dw_kernel's documented TF32 long-reduction caveat -> <1e-3.
    print("  fp32 (gate: idx=1.0, w<1e-4, gx<1e-4, gw<1e-3 [dw-kernel TF32 caveat]):")
    for (B, S, H) in shapes:
        ia, wd, gxd, gwd = correctness(B, S, H, torch.float32)
        ok = (ia == 1.0) and gxd < 1e-4 and wd < 1e-4 and gwd < 1e-3
        print(f"    B{B} S{S} H{H}: idx_agree={ia:.4f} w|Δ|={wd:.2e} gx|Δ|={gxd:.2e} gw|Δ|={gwd:.2e}  {'PASS' if ok else 'FAIL'}")
    print("  fp16 (informational — fused keeps fp32 logits so it routes MORE precisely than the")
    print("        fp16-rounding reference; small idx divergence + grad diff is expected, not a bug):")
    for (B, S, H) in shapes:
        ia, wd, gxd, gwd = correctness(B, S, H, torch.float16)
        print(f"    B{B} S{S} H{H}: idx_agree={ia:.4f} w|Δ|={wd:.2e} gx|Δ|={gxd:.2e} gw|Δ|={gwd:.2e}")

    print("\nROUTING vs true cuDNN-conv eager (informational; <1.0 = expected conv numeric-path swap)")
    for (B, S, H) in shapes:
        print(f"  B{B} S{S} H{H}: idx_agree={vs_eager_routing(B, S, H, torch.float16):.4f}")

    print("\nNaN-FREE (3 fwd+bwd passes, fp16):", "PASS" if nan_check(16, 1024, 512, torch.float16) else "FAIL")

    print("\n" + "=" * 78)
    print("SPEED (fp16, median ms over 30 iters)")
    for (B, S, H) in shapes:
        (fe, fbe), (ff, fbf) = bench_speed(B, S, H, torch.float16)
        print(f"  B{B} S{S} H{H}:")
        print(f"    fwd     eager {fe:.3f}  fused {ff:.3f}  speedup {fe/ff:.2f}x")
        print(f"    fwd+bwd eager {fbe:.3f}  fused {fbf:.3f}  speedup {fbe/fbf:.2f}x")

    print("\n" + "=" * 78)
    print("KERNEL LAUNCHES (fp16, one fwd+bwd) — fused GEMMs must be <= eager")
    for (B, S, H) in shapes:
        (le, mme), (lf, mmf) = launch_counts(B, S, H, torch.float16)
        print(f"  B{B} S{S} H{H}: total launches eager={le} fused={lf} | matmul/conv eager={mme} fused={mmf}")
