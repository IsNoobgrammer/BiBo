"""One-step GPU timeline: how much of a training step the GPU is actually busy.

    python -m ablate.common.train <board flags> --profile_step 30

train.py profiles that one step (zero_grad .. optimizer/scheduler step), prints this report and
exits. busy = union of all CUDA kernel/memcpy intervals; idle = step wall time - busy. Gaps are
the idle stretches between consecutive GPU activity, attributed to the CPU op running before them.
"""
import collections
import torch

# CPU-side calls that block the host until the GPU catches up. Each one drains the launch queue, so
# everything launched after it starts with an empty GPU.
SYNCS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


def _site(e):
    """First BiBo/kernels frame of the op that issued this runtime call (walks up the CPU parents:
    runtime events such as cudaStreamSynchronize carry no Python stack themselves)."""
    p = e
    while p is not None:
        fr = [f for f in (p.stack or []) if any(k in f for k in ("ablate", "src/", "kernels/", "modeling"))]
        if fr:
            return f"{p.name} @ {fr[0]}"
        p = p.cpu_parent
    return f"{e.cpu_parent.name if e.cpu_parent else '?'} @ ?"


# kernel-name -> component, first match wins. Coarse on purpose: it answers "where do the ms go".
COMPONENTS = (
    ("moe gemm (Triton grouped)", ("_grouped_gemm", "_grouped_mm_kernel", "_gate_up_glu", "_grouped_wgrad", "_wg_kernel", "_wg_reduce")),
    ("moe act row-ops (radial)", ("_glu_fwd_rowloop", "_glu_bwd_rowloop", "_glu_fwd", "_glu_bwd")),
    ("router / megakernel", ("router", "_mk_", "topk", "megakernel")),
    ("attn residual (depth mix + carry)", ("_attn_res_", "residual_add", "_fwd_kernel", "_bwd_kernel")),
    ("attention (fused attn_xsa)", ("_attn_prep", "_attn_fwd", "_attn_bwd"),),
    ("attention (flex/xsa/rope)", ("flex_attention", "flash", "xsa", "rope", "attn")),
    ("fused CE", ("_grad_logits", "cross_entropy", "_ce_", "logsumexp")),
    ("optimizer + foreach (clip, AdamW)", ("_bgemm_epi", "_bmmt", "_bssm", "_pre_kernel", "_post_kernel", "_norm_t", "multi_tensor_apply")),
    ("dense cuBLAS/cutlass GEMM", ("cutlass", "gemm", "nvjet", "xmma", "sm90", "sm100")),
    ("norms (liger)", ("rms_norm", "_layer_norm")),
    ("copy / cast / memcpy", ("copy", "Memcpy", "Memset", "cast", "CatArray")),
    ("elementwise / reduce / other", ("",)),
)


def component(name):
    n = name.lower()
    for label, keys in COMPONENTS:
        if any(k.lower() in n for k in keys):
            return label
    return "elementwise / reduce / other"


def report(prof, wall_ms, top=12):
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0]
    iv = sorted((e.time_range.start, e.time_range.end, e.name) for e in ev)
    busy, gaps, cur_s, cur_e, last = 0.0, [], None, None, None
    for s, e, n in iv:
        if cur_e is None:
            cur_s, cur_e, last = s, e, n
        elif s > cur_e:
            busy += cur_e - cur_s
            gaps.append((s - cur_e, last, n))
            cur_s, cur_e, last = s, e, n
        elif e > cur_e:
            cur_e, last = e, n
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (iv[-1][1] - iv[0][0]) / 1e3 if iv else 0.0
    busy /= 1e3
    ev = [e for e in ev if not e.name.startswith(("Optimizer.", "ProfilerStep"))]   # annotations, not work
    kern = {}
    for e in ev:
        k = kern.setdefault(e.name, [0, 0.0]); k[0] += 1; k[1] += e.device_time_total / 1e3
    gap_ms = [g[0] / 1e3 for g in gaps]
    print(f"[profile] step wall {wall_ms:.1f} ms | GPU span {span:.1f} ms | GPU busy {busy:.1f} ms "
          f"({100 * busy / wall_ms:.1f}% of wall) | idle {wall_ms - busy:.1f} ms | {len(ev)} GPU ops, "
          f"{len(gaps)} gaps", flush=True)
    for lo, hi in ((0, 0.005), (0.005, 0.02), (0.02, 0.1), (0.1, 1), (1, 1e9)):
        sel = [g for g in gap_ms if lo <= g < hi]
        print(f"[profile]   gaps {lo * 1e3:>6.0f}-{hi * 1e3 if hi < 1e8 else float('inf'):>6.0f} us: "
              f"{len(sel):6d} gaps, {sum(sel):8.2f} ms", flush=True)
    print("[profile] largest gaps (ms, after -> before):", flush=True)
    for g, a, b in sorted(gaps, reverse=True)[:top]:
        print(f"[profile]   {g / 1e3:8.3f}  {str(a)[:60]} -> {str(b)[:60]}", flush=True)
    sy = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CPU and e.name in SYNCS:
            sy[(e.name, _site(e))] += 1
    mc = collections.Counter(e.name for e in ev if e.name.startswith("Memcpy"))
    print(f"[profile] GPU copies: {dict(mc)}", flush=True)
    print(f"[profile] host->GPU waits (stream/device/event sync) in the step: {sum(sy.values())}", flush=True)
    for (n, site), c in sy.most_common(25):
        print(f"[profile]   {c:5d}  {n:22s} {site[:150]}", flush=True)
    comp = collections.defaultdict(lambda: [0, 0.0])
    for n, (c, t) in kern.items():
        k = comp[component(n)]; k[0] += c; k[1] += t
    tot = sum(t for _, t in comp.values())
    print(f"[profile] GPU time by component (sum {tot:.1f} ms; overlapping streams can exceed busy):", flush=True)
    for label, (c, t) in sorted(comp.items(), key=lambda x: -x[1][1]):
        print(f"[profile]   {t:8.2f} ms {100 * t / tot:5.1f}% {c:6d} ops  {label}", flush=True)
    print("[profile] top GPU ops by time (ms, count):", flush=True)
    for n, (c, t) in sorted(kern.items(), key=lambda x: -x[1][1])[:40]:
        print(f"[profile]   {t:8.2f} {c:6d}  {n[:100]}", flush=True)
    # WHO launched the glue: each elementwise / copy / reduce kernel -> the first repo frame of its
    # CPU op (forward) or the autograd node that ran it (backward). This is the waste map.
    # device events carry no CPU parent here, so walk the CPU ops and read the kernels each launched
    glue = collections.defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CPU or not e.kernels:
            continue
        for kk in e.kernels:
            if component(kk.name) in ("elementwise / reduce / other", "copy / cast / memcpy"):
                k = glue[_origin(e) + " | " + kk.name[:40]]; k[0] += 1; k[1] += kk.duration / 1e3
    gt = sum(t for _, t in glue.values())
    print(f"[profile] elementwise + copy kernels by origin ({gt:.1f} ms):", flush=True)
    for o, (c, t) in sorted(glue.items(), key=lambda x: -x[1][1])[:45]:
        print(f"[profile]   {t:8.2f} {c:6d}  {o[:170]}", flush=True)


def _origin(e):
    """'fwd <aten op> @ file:line' or 'bwd <autograd node> <aten op>' for the CPU op that launched."""
    op, p, node = None, e, None
    while p is not None:
        if op is None and p.name.startswith("aten::"):
            op = p.name
        if p.name.startswith("autograd::engine::evaluate_function:"):
            node = p.name.split(":", 4)[-1].strip()
            break
        fr = [f for f in (p.stack or []) if any(k in f for k in ("ablate", "src/", "kernels/", "exp/"))]
        if fr:
            return f"fwd {op or p.name} @ {fr[0]}"
        p = p.cpu_parent
    return f"bwd {node} {op or ''}" if node else f"? {op or e.name[:60]}"


_SITES = collections.Counter()


def sync_audit_start():
    """torch's sync debug mode warns on every host-device synchronization; record each warning's
    call site (the innermost BiBo / kernels frames) instead of printing it."""
    import traceback
    import warnings

    def _rec(message, category, filename, lineno, file=None, line=None):
        fr = [f for f in traceback.extract_stack()[:-1]
              if any(k in f.filename for k in ("ablate", "/src/", "kernels")) and "step_profile" not in f.filename]
        _SITES[" <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}" for f in fr[-3:][::-1]) or "?"] += 1

    warnings.showwarning = _rec
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")


def sync_audit_report():
    torch.cuda.set_sync_debug_mode(0)
    print(f"[sync] host syncs in one full step: {sum(_SITES.values())} (innermost frame first)", flush=True)
    for site, c in _SITES.most_common(30):
        print(f"[sync]   {c:5d}  {site[:200]}", flush=True)
