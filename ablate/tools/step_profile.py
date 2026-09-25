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
    print("[profile] top GPU ops by time (ms, count):", flush=True)
    for n, (c, t) in sorted(kern.items(), key=lambda x: -x[1][1])[:top]:
        print(f"[profile]   {t:8.2f} {c:6d}  {n[:100]}", flush=True)
