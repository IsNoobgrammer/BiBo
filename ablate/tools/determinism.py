"""Run-to-run determinism of one training micro-batch: same weights, same batch, forward+backward
N times, then compare the loss and EVERY parameter gradient bit-for-bit against the first repeat.

    python -m ablate.common.train <board flags> --determinism_check 3

Repeat 0 is a warm-up (Triton autotune and cuBLAS heuristics settle there) and is not compared.
Differing gradients are grouped by parameter name with layer indices stripped, so one
nondeterministic kernel shows up as one row (e.g. `layers.*.mlp.experts.gate_up_proj`), not ten.
"""
import re
import collections

import torch


def run(model, gen, loss_fn, amp, repeats):
    ids = next(gen)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ref_loss, ref_g = None, None
    rows = collections.OrderedDict()
    for r in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        with amp:
            loss = loss_fn(ids)
        loss.backward()
        torch.cuda.synchronize()
        g = [None if p.grad is None else p.grad.detach().clone() for _, p in named]
        if r == 0:
            continue                                   # warm-up: autotune picks configs here
        if ref_g is None:
            ref_loss, ref_g = loss.detach().clone(), g
            continue
        same_loss = torch.equal(loss.detach(), ref_loss)
        print(f"[det] repeat {r}: loss {float(loss):.8f} vs {float(ref_loss):.8f} -> "
              f"{'bit-identical' if same_loss else 'DIFFERENT (forward is nondeterministic)'}", flush=True)
        n_diff = 0
        for (n, _), a, b in zip(named, g, ref_g):
            if a is None or b is None or torch.equal(a, b):
                continue
            n_diff += 1
            key = re.sub(r"\.\d+\.", ".*.", n)
            rel = float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
            k = rows.setdefault(key, [0, 0.0])
            k[0] += 1
            k[1] = max(k[1], rel)
        print(f"[det] repeat {r}: {n_diff}/{len(named)} parameter grads differ", flush=True)
    print(f"[det] nondeterministic gradient groups (count, max rel diff):", flush=True)
    for key, (c, rel) in sorted(rows.items(), key=lambda x: -x[1][1]):
        print(f"[det]   {c:4d}  {rel:.2e}  {key}", flush=True)
    if not rows:
        print("[det]   none -- backward is bit-deterministic for this batch", flush=True)
    print("[det] DONE", flush=True)
