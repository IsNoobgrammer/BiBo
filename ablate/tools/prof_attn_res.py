"""Full profile of the AttnRes path at the board shape (T=65536, H=512, bf16 stream, b3 sites=1 carry).

A. kernel level, per candidate count N: fused Triton vs eager torch vs torch.compile (Inductor),
   fwd and fwd+bwd, peak memory, achieved bandwidth. Same for the carry residual add.
B. topology level: the real 10-layer exp model with attention and MLP replaced by a per-channel
   scale, so ONLY the residual plumbing is left (depth mix, block cat, carry, prefix adds, final
   mix, and the autograd accumulation of block grads). AR vs control, eager-fused / eager-torch /
   Inductor, per-kernel breakdown.
Inductor's generated code lands in $TORCHINDUCTOR_CACHE_DIR (set it to a fresh dir).

    TORCHINDUCTOR_CACHE_DIR=/home/marimo/work/ardump python -m ablate.tools.prof_attn_res
"""
from ablate.common import _paths  # noqa: F401
import collections, glob, os, re, sys, types

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

import exp.modeling_bibo as xm
from kernels.sm120.attn_res import attn_res as fused_ar
from kernels.sm120.residual_add import make_mlp_input
from ablate.common.models import build_arm
from ablate.common.configs import swa_block_pattern, SHARED

DEV, BF = "cuda", torch.bfloat16
T, H = 65536, 512
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 64


def timed(fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def peak(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def kernels(fn, it=3):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
    k = collections.defaultdict(lambda: [0.0, 0])
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0:
            n = re.sub(r"\(.*", "", e.name)
            n = re.sub(r"<.*", "", n) if n.startswith("void") else n
            k[n[:70]][0] += e.device_time_total / 1e3 / it
            k[n[:70]][1] += 1
    return {n: (t, c // it) for n, (t, c) in k.items()}


def show(k, top=12, ind="      "):
    tot = sum(t for t, _ in k.values())
    print(f"{ind}kernel total {tot:.3f} ms, {sum(c for _, c in k.values())} launches")
    for n, (t, c) in sorted(k.items(), key=lambda x: -x[1][0])[:top]:
        print(f"{ind}  {t:7.3f} ms  x{c:<4d} {n}")


# ---------------------------------------------------------------- A. kernel level
def eager_ar(br, ps, w_norm, w_proj):
    norm = types.SimpleNamespace(weight=w_norm, variance_epsilon=1e-6)
    proj = types.SimpleNamespace(weight=w_proj)
    old = xm._HAS_FUSED_AR
    xm._HAS_FUSED_AR = False
    try:
        return xm.apply_attention_residual(ps, br, proj, norm, 0, 0)
    finally:
        xm._HAS_FUSED_AR = old


def fused_ar_call(br, ps, w_norm, w_proj):
    return fused_ar(br, ps, w_norm.float() * w_proj.squeeze(0).float(), 1e-6, 0, 0)


def part_a():
    print("== A. depth mix, T=65536 H=512, block_residual + prefix_sum bf16 (board layout)")
    print("      N    impl                 fwd ms   fwd+bwd ms   peak MiB   fwd GB/s  f+b GB/s")
    comp = torch.compile(eager_ar, dynamic=False)
    comp_ma = torch.compile(eager_ar, dynamic=False, mode="max-autotune-no-cudagraphs")
    for N in (2, 3, 4, 5):
        g = torch.Generator(device=DEV).manual_seed(N)
        br = (torch.randn(T, N - 1, H, device=DEV, generator=g) * 3).to(BF).requires_grad_()
        ps = (torch.randn(T, H, device=DEV, generator=g) * 20).to(BF).requires_grad_()
        wn = torch.ones(H, device=DEV).requires_grad_()
        wp = (torch.randn(1, H, device=DEV, generator=g) * 0.05).requires_grad_()
        go = torch.randn(T, H, device=DEV, generator=g).to(BF)
        # ideal bytes: fwd reads V once + writes out; bwd reads V + dout, writes dV (+ fwd recompute)
        bv = N * T * H * 2
        b_f, b_fb = bv + T * H * 2, (bv + T * H * 2) + (2 * bv + T * H * 2)
        for name, fn in (("fused triton", fused_ar_call), ("eager torch", eager_ar),
                         ("inductor", comp), ("inductor max-at", comp_ma)):
            try:
                f = lambda: fn(br, ps, wn, wp)
                fb = lambda: fn(br, ps, wn, wp).backward(go)
                tf, tfb, pm = timed(f), timed(fb), peak(fb)
                print(f"      {N}    {name:18s} {tf:8.3f}   {tfb:9.3f}   {pm:8.0f}   "
                      f"{b_f / tf / 1e6:8.0f}  {b_fb / tfb / 1e6:8.0f}")
                if name in ("fused triton", "inductor") and N == 4:
                    show(kernels(fb), top=8)
            except Exception as ex:
                print(f"      {N}    {name:18s} FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:80]}")
            br.grad = ps.grad = wn.grad = wp.grad = None

    print("== A2. carry add h = read + c*attn_out, per-dim c, bf16")
    read = torch.randn(T, H, device=DEV).to(BF).requires_grad_()
    ao = torch.randn(T, H, device=DEV).to(BF).requires_grad_()
    th = torch.zeros(H, device=DEV).requires_grad_()
    go = torch.randn(T, H, device=DEV).to(BF)
    fused_c = lambda r, t, a: make_mlp_input(r, 2 * torch.sigmoid(t), a, modes=("none",))
    eager_c = lambda r, t, a: r + (2 * torch.sigmoid(t)).to(a.dtype) * a
    comp_c = torch.compile(eager_c, dynamic=False)
    for name, fn in (("fused triton", fused_c), ("eager torch", eager_c), ("inductor", comp_c)):
        fb = lambda: fn(read, th, ao).backward(go)
        print(f"           {name:18s} {timed(lambda: fn(read, th, ao)):8.3f}   {timed(fb):9.3f}   {peak(fb):8.0f}"
              f"   ideal f+b bytes {(3 + 4) * T * H * 2 / 2**20:.0f} MiB")
        show(kernels(fb), top=5)


# ---------------------------------------------------------------- B. topology level
class MockAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(H, device=DEV))

    def forward(self, hidden_states, **kw):
        return hidden_states * self.w.to(hidden_states.dtype), None


class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(H, device=DEV))

    def forward(self, x):
        return x * self.w.to(x.dtype)


def build(ar):
    torch.manual_seed(0)
    kw = dict(attn_res=ar, attn_res_sites=1, attn_res_carry=True, attn_res_carry_per_dim=True,
              attn_res_carry_scale="sigmoid") if ar != "control" else dict(attn_res="control")
    model, _ = build_arm("bibo_min", device=DEV, dtype=torch.float32, attn_impl="sdpa",
                         num_experts=8, top_k=2, special_pairs=0, use_xsa=True,
                         hybrid_layer_pattern=swa_block_pattern(SHARED["num_hidden_layers"]),
                         sliding_window=128, bf16_residual_stream=True, **kw)
    for L in model.model.layers:
        L.self_attn, L.mlp = MockAttn(), MockMLP()
    return model.model.train()


def part_b():
    print("== B. residual plumbing only (attention/MLP mocked), 10 layers, B64 x S1024, fwd+bwd")
    ids = torch.randint(0, 1000, (64, 1024), device=DEV)

    def step(m):
        with torch.autocast("cuda", dtype=BF):
            h = m(input_ids=ids, use_cache=False)[0]
        h.float().square().mean().backward()

    res = {}
    for tag, ar, fused, comp in (("control eager", "control", True, False),
                                 ("AR fused (production)", "3", True, False),
                                 ("AR eager torch", "3", False, False),
                                 ("control inductor", "control", True, True),
                                 ("AR inductor", "3", False, True)):
        xm._HAS_FUSED_AR = xm._HAS_FUSED_RES_ADD = fused
        m = build(ar)
        mm = torch.compile(m, dynamic=False) if comp else m
        try:
            t = timed(lambda: step(mm), it=6)
            pm = peak(lambda: step(mm))
            res[tag] = t
            print(f"   {tag:24s} {t:8.2f} ms   peak {pm:7.0f} MiB")
            show(kernels(lambda: step(mm), it=2), top=14)
        except Exception as ex:
            print(f"   {tag}: FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:100]}")
        m.zero_grad(set_to_none=True)
        del m, mm
        torch.cuda.empty_cache()
    xm._HAS_FUSED_AR = xm._HAS_FUSED_RES_ADD = True
    if "control eager" in res:
        for k, v in res.items():
            print(f"   AR tax vs control eager: {k:24s} {v - res['control eager']:+8.2f} ms per micro-step "
                  f"(x4 micro = {4 * (v - res['control eager']):+.1f} ms/step)")


def dump_list():
    dd = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not dd:
        return
    print(f"== Inductor sources under {dd}")
    for f in sorted(glob.glob(os.path.join(dd, "**", "*.py"), recursive=True)):
        txt = open(f, errors="ignore").read()
        names = re.findall(r"def (triton_\w+)\(", txt)
        if names:
            print(f"   {f}: {', '.join(names)[:300]}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "a" in which:
        part_a()
    if "b" in which:
        part_b()
    dump_list()
    print("ALLDONE_PROF_AR")
