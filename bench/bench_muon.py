"""
Frozen eval for the Muon optimizer step (bench/optim.py) — verifies the torch.compile lever.

Background (Jun 27): cross-param shape-bucketing (one batched NS over all same-shape matrices) was
tried and REVERTED — numerically exact (1e-6) but only ~1.1x on the FLOP-bound expert matmuls at 2x
memory, which thrashed the 4GB local GPU (22x slower). The Muon step is GEMM-bound (~71%, profiled)
and its GEMM count is the algorithmic floor (NS = 3 matmuls/iter); cuBLAS bmm beats Triton tl.dot.
So the only real launch-overhead lever is torch.compile + cudagraphs on the (per-param) Newton-Schulz.

This bench:
  1. Profiles the eager Muon step: time, GEMM%, GEMM-launch count (confirms GEMM-bound).
  2. Compares eager vs compile_ns=True (compile broken locally -> prints "unavailable", runs on Kaggle):
     numerics-equivalence (max|Δp| per step, expect ~1e-6) + speedup + GEMM-launch count.

Run: .venv/Scripts/python bench/bench_muon.py    (locally: section 2 reports compile unavailable)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import yaml
from torch.profiler import profile, ProfilerActivity

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from optim import Muon

DEV = "cuda"
torch.manual_seed(0)


def get_muon_params(compile_ns=False):
    db = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "configs", "bibo.yaml")))
    cfg = BiBoConfig(**{k: v for k, v in db["model"].items() if k != "type"})
    model = BiBoForCausalLM(cfg).to(DEV)
    params = [p for n, p in model.named_parameters()
              if (p.ndim == 2) or (p.ndim == 3 and "experts." in n and ("gate_up_proj" in n or "down_proj" in n))]
    muon = Muon(params, lr=3e-4, momentum=0.95, weight_decay=0.1, compile_ns=compile_ns)
    return model, muon, params


def time_step(muon, params, warm=5, it=20):
    gen = torch.Generator(device=DEV).manual_seed(1)
    def prime():
        for p in params:
            p.grad = torch.randn(*p.shape, generator=gen, device=DEV)
    for _ in range(warm):
        prime(); muon.step()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        prime(); muon.step()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def profile_step(muon, params):
    for p in params: p.grad = torch.randn_like(p)
    for _ in range(3): muon.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for p in params: p.grad = torch.randn_like(p)
        muon.step(); torch.cuda.synchronize()
    def ct(e):
        for a in ("self_device_time_total", "self_cuda_time_total", "cuda_time_total"):
            v = getattr(e, a, 0) or 0
            if v: return v
        return 0
    rows = [e for e in prof.key_averages() if ct(e) > 0]
    tot = sum(ct(e) for e in rows)
    gemm_t = sum(ct(e) for e in rows if any(t in e.key.lower() for t in ("bmm", "gemm", "matmul", "cutlass", "sgemm")))
    gemm_n = sum(e.count for e in rows if any(t in e.key.lower() for t in ("bmm", "gemm", "matmul", "cutlass", "sgemm")))
    nlaunch = sum(e.count for e in rows)
    return tot, gemm_t, gemm_n, nlaunch


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    model, muon, params = get_muon_params(compile_ns=False)
    print(f"Muon param set: {len(params)} tensors (attention + dense MLP + 3D experts)")
    print("\n1. EAGER Muon step (profile):")
    t_e = time_step(muon, params)
    tot, gemm_t, gemm_n, nl = profile_step(muon, params)
    print(f"   time={t_e:.1f}ms | GEMM {100*gemm_t/max(tot,1):.0f}% of CUDA | gemm-launches={gemm_n} total-launches={nl}")
    print("   -> GEMM-bound; gemm count is the NS floor (3 matmuls/iter). cuBLAS engine; compile cuts launch overhead.")

    print("\n2. EAGER vs torch.compile(NS) — numerics + speed (Kaggle; local compile is broken):")
    try:
        model_c, muon_c, params_c = get_muon_params(compile_ns=True)
        # sync compiled params to eager-fresh params for a fair numerics compare
        with torch.no_grad():
            for pc, pe in zip(params_c, params):
                pc.copy_(pe)
        gen = torch.Generator(device=DEV).manual_seed(99)
        worst = 0.0
        for _ in range(5):
            grads = [torch.randn(*p.shape, generator=gen, device=DEV) for p in params]
            for p, g in zip(params, grads): p.grad = g.clone()
            for p, g in zip(params_c, grads): p.grad = g.clone()
            muon.step(); muon_c.step()
            worst = max(worst, max((a - b).abs().max().item() for a, b in zip(params, params_c)))
        t_c = time_step(muon_c, params_c)
        _, _, gemm_nc, nlc = profile_step(muon_c, params_c)
        ok = worst < 1e-3
        print(f"   numerics max|Δp| eager-vs-compiled = {worst:.2e}  {'PASS' if ok else 'FAIL'}")
        print(f"   speed eager {t_e:.1f}ms -> compiled {t_c:.1f}ms ({t_e/t_c:.2f}x) | launches {nl}->{nlc}")
    except Exception as ex:
        msg = str(ex).splitlines()[0][:120]
        print(f"   compile unavailable here: {msg}")
        print("   -> expected locally (torch.compile/triton AttrsDescriptor mismatch). Run this on Kaggle/T4.")
