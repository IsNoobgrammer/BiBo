"""Whole-model parity of our kernels: one training step (same batch, dropout 0, no SpecAugment, one look-ahead) dumped by
train_asr.py --parity_dump three times -- NeMo eager bf16 (no kernel flags), ours (kernel flags, bf16), and the fp32
reference (--parity_fp32, no kernels, fp32 matmuls at 'highest') -- then compared here.

Gate: for the loss, the global gradient and every parameter, ours vs fp32 must be within max(1.25 x eager's error,
1e-3 absolute-relative floor for tiny grads). Also reports ours vs eager directly.

    python voice/asr/parity_e2e.py REF.pt EAGER.pt OURS.pt
"""
import sys

import torch


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


def main():
    ref, eag, our = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:4])
    names = [n for n in ref["grads"] if n in eag["grads"] and n in our["grads"]]
    miss = sorted(set(ref["grads"]) ^ set(our["grads"])) + sorted(set(ref["grads"]) ^ set(eag["grads"]))
    print(f"loss  ref {ref['loss']:.6f}  eager {eag['loss']:.6f} ({abs(eag['loss'] - ref['loss']) / abs(ref['loss']):.1e})"
          f"  ours {our['loss']:.6f} ({abs(our['loss'] - ref['loss']) / abs(ref['loss']):.1e})")
    print(f"{len(names)} grads compared; missing on one side: {miss[:6] if miss else 'none'}")
    flat = lambda d: torch.cat([d["grads"][n].flatten() for n in names])
    gr, ge, go = flat(ref), flat(eag), flat(our)
    print(f"global grad  eager {rel(ge, gr):.2e}  ours {rel(go, gr):.2e}  ours-vs-eager {rel(go, ge):.2e}  "
          f"|g| ref {gr.norm():.4f} eager {ge.norm():.4f} ours {go.norm():.4f}")
    rows, bad = [], []
    for n in names:
        e, o = rel(eag["grads"][n], ref["grads"][n]), rel(our["grads"][n], ref["grads"][n])
        rows.append((o / max(e, 1e-12), n, e, o))
        if o > max(1.25 * e, 1e-3):
            bad.append((n, e, o))
    rows.sort(reverse=True)
    print("worst 12 params by ours/eager error ratio (vs fp32):")
    for r, n, e, o in rows[:12]:
        print(f"  {r:6.2f}x  eager {e:.2e}  ours {o:.2e}  {n}")
    groups = {}
    for _, n, e, o in rows:
        k = n.split(".")[0] if not n.startswith("encoder.layers.") else "encoder.layers"
        g = groups.setdefault(k, [0, 0.0, 0.0])
        g[0] += 1
        g[1] = max(g[1], e)
        g[2] = max(g[2], o)
    print("per module (count, worst eager err, worst ours err): " +
          "  ".join(f"{k} {c} {e:.1e}/{o:.1e}" for k, (c, e, o) in groups.items()))
    loss_ok = abs(our["loss"] - ref["loss"]) <= max(1.25 * abs(eag["loss"] - ref["loss"]), 1e-4 * abs(ref["loss"]))
    glob_ok = rel(go, gr) <= max(1.25 * rel(ge, gr), 1e-3)
    print(f"[{'PASS' if loss_ok else 'FAIL'}] loss   [{'PASS' if glob_ok else 'FAIL'}] global grad   "
          f"[{'PASS' if not bad else 'FAIL'}] per-param ({len(bad)} over the gate"
          + (": " + ", ".join(f"{n} {o:.1e}>{e:.1e}" for n, e, o in bad[:6]) if bad else "") + ")")
    print("E2E_PARITY " + ("PASS" if loss_ok and glob_ok and not bad else "FAIL"))


if __name__ == "__main__":
    main()
