"""Multi-step optimizer parity on REAL BiBo gradients: does low-precision error accumulate?

    python -m ablate.common.train <the usual board flags> --optim_parity_steps 15 --batch 16 --grad_accum 1

Single-step parity says nothing about accumulation: momentum is a bf16 EMA and every step's NS
output is rounded, so small per-step errors can compound or overflow. Here train.py builds the
exact board model, patches and data stream, then instead of training it runs this:

  REFERENCE   the real model, stepped by fp32 FusedMuon (fp32 momentum, fp32 NS on cuBLAS, eager
              tail) + the usual AdamW for everything else.
  CANDIDATES  each holds its OWN copy of the Muon parameters and its own optimizer state, and every
              step applies the SAME gradients -- the ones the real forward/backward produced at the
              reference weights. Gradients are shared on purpose: the MoE backward is not bitwise
              deterministic (atomics), so free-running arms would diverge from that alone and hide
              what the optimizer contributes. With a shared gradient stream, any drift from the
              reference is optimizer error, accumulated.

Reported per step: weight error rel = ||W - W_ref|| / ||W_ref - W0||, momentum error vs the fp32
momentum, momentum max|m| and non-finite count (overflow), and whether the weights are still
bit-identical to "old" (the pre-refactor bf16 optimizer).
"""
import torch

from kernels.sm120.muon import FusedMuon

LR, MU = 1e-2, 0.95
CANDIDATES = [
    ("old", dict(ns_backend="cublas", fused_tail=False)),
    ("cublas+tail", dict(ns_backend="cublas")),
    ("epi", dict(ns_backend="epi")),
    ("symmul", dict(ns_backend="symmul")),
    ("symepi", dict(ns_backend="symepi")),
    ("auto", dict(ns_backend="auto")),
    ("gram(4,6)", dict(ns_backend="gram")),
]
REPORT = (1, 2, 3, 5, 10, 12, 15)


def _groups(model):
    """The Muon param set exactly as build_optimizers routes it: 3D stacks, then 2D matrices."""
    stacks, mats = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or "embed" in n or p.ndim not in (2, 3):
            continue
        if p.ndim == 2 and 1 in p.shape:           # vec_matrices_adamw: (1, H) vectors go to AdamW
            continue
        (stacks if p.ndim == 3 else mats).append(p)
    return stacks, mats


def _muon(stacks, mats, dt, wd, **kw):
    return FusedMuon([{"params": stacks}, {"params": mats}], lr=LR, momentum=MU, weight_decay=wd,
                     ns_coeffs="ns8", ns_dtype=dt, variant="aurora", **kw)


def _moms(opt):
    """Momentum buffers in plan order (one per shape bucket), comparable across optimizers built the
    same way from the same param order."""
    out = []
    for g in opt.param_groups:
        for b in opt._plan_cache.get(id(g), []):
            out.append(opt.state[b["anchor"]]["muon_mom"])
    return out


@torch.no_grad()
def run(model, gen, loss_fn, amp, steps, wd, adam_lr):
    stacks, mats = _groups(model)
    ref_ps = stacks + mats
    W0 = [p.detach().clone() for p in ref_ps]
    print(f"[parity] {len(stacks)} stacks + {len(mats)} mats = "
          f"{sum(p.numel() for p in ref_ps) / 1e6:.1f}M Muon params; {len(CANDIDATES)} candidates; "
          f"lr {LR} constant, wd {wd}, aurora, ns8", flush=True)
    ref_opt = _muon(stacks, mats, torch.float32, wd, ns_backend="cublas", fused_tail=False)
    muon_ids = {id(p) for p in ref_ps}
    adamw = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad and id(p) not in muon_ids],
                              lr=adam_lr, weight_decay=wd)
    cands = []
    for name, kw in CANDIDATES:
        cs = [torch.nn.Parameter(p.detach().clone()) for p in stacks]
        cm = [torch.nn.Parameter(p.detach().clone()) for p in mats]
        wd_c = 0.0 if name == "muown" else wd
        cands.append((name, cs + cm, _muon(cs, cm, torch.bfloat16, wd_c, **kw)))
    hist = {}
    for step in range(1, steps + 1):
        for p in model.parameters():
            p.grad = None
        ids = next(gen)
        with torch.enable_grad():
            with amp:
                loss = loss_fn(ids)
            loss.backward()
        bad_g = sum(int((~torch.isfinite(p.grad)).sum()) for p in ref_ps if p.grad is not None)
        for name, ps, opt in cands:
            for c, r in zip(ps, ref_ps):
                c.grad = r.grad
            opt.step()
        ref_opt.step()
        adamw.step()
        if step not in REPORT:
            continue
        total = torch.sqrt(sum(((r.float() - w) ** 2).sum() for r, w in zip(ref_ps, W0)))
        ref_m = _moms(ref_opt)
        m_norm = torch.sqrt(sum(m.float().pow(2).sum() for m in ref_m))
        old_ps = cands[0][1]
        print(f"\n[parity] step {step}  loss {float(loss):.4f}  non-finite grads {bad_g}  "
              f"fp32 total change ||W-W0|| {float(total):.4e}", flush=True)
        print(f"   {'optimizer':<12} {'W rel err':>10} {'W max|d|':>10} {'mom rel err':>12} {'max|mom|':>9} "
              f"{'nonfinite':>9} {'=old bits':>9}", flush=True)
        for name, ps, opt in cands:
            wr = float(torch.sqrt(sum(((c - r) ** 2).sum() for c, r in zip(ps, ref_ps))) / total)
            wm = max(float((c - r).abs().max()) for c, r in zip(ps, ref_ps))
            cm = _moms(opt)
            mr = float(torch.sqrt(sum(((a.float() - b.float()) ** 2).sum() for a, b in zip(cm, ref_m))) / m_norm)
            mmax = max(float(a.float().abs().max()) for a in cm)
            nf = sum(int((~torch.isfinite(a)).sum()) for a in cm) + sum(int((~torch.isfinite(c)).sum()) for c in ps)
            same = sum(torch.equal(c, o) for c, o in zip(ps, old_ps))
            hist.setdefault(name, []).append((step, wr, mr))
            print(f"   {name:<12} {wr:>10.3e} {wm:>10.3e} {mr:>12.3e} {mmax:>9.3f} {nf:>9d} "
                  f"{same:>4d}/{len(ps)}", flush=True)
        print(f"   {'fp32 ref':<12} {'0':>10} {'0':>10} {'0':>12} {max(float(m.abs().max()) for m in ref_m):>9.3f}",
              flush=True)
    print("\n[parity] GROWTH of weight error (rel err at the last reported step / at step 1):")
    for name, rows in hist.items():
        print(f"   {name:<12} {rows[0][1]:.3e} -> {rows[-1][1]:.3e}   x{rows[-1][1] / max(rows[0][1], 1e-30):.1f}")
    print("[parity] DONE", flush=True)
