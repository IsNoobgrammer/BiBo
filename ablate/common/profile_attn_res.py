"""Kernel-level profile: why is AttnRes carry FASTER than no AttnRes?

Measured repeatedly at 60 steps, eager (no --compile), on an idle box:

    src model, no AttnRes    1518 ms/step   60.9 GB
    exp model, AttnRes OFF   1519 ms/step   60.9 GB
    exp model, b3 carry      1486 ms/step   60.2 GB     <- 2.2% FASTER, 0.7 GB LIGHTER

Ruled out by measurement: run ordering, torch.compile (it is off), the src-vs-exp code path
(the two controls agree to 1 ms), and MoE load imbalance. The carry path does strictly MORE
arithmetic -- three residual adds per layer to the control's two, plus the depth mix -- so a
speedup has no business existing. This names the op instead of guessing.

    python -m ablate.common.profile_attn_res
"""
from . import _paths  # noqa: F401

import argparse
import torch
from torch.profiler import profile, ProfilerActivity

from .models import build_arm
from .configs import swa_block_pattern, SHARED
from . import patches as patchmod
from kernels.sm120.cross_entropy import fused_linear_cross_entropy

DEV = "cuda"


def _build(attn_res, sites, carry):
    torch.manual_seed(42069)
    model, _ = build_arm(
        "bibo_min", device=DEV, dtype=torch.float32, attn_impl="sdpa",
        num_experts=64, top_k=6, special_pairs=0,
        use_xsa=True, xsa_alpha_init=0.0,
        hybrid_layer_pattern=swa_block_pattern(SHARED["num_hidden_layers"]),
        sliding_window=128, swa_qk_norm=True,
        attn_res=attn_res, attn_res_sites=sites, attn_res_carry=carry,
    )
    return model.train()


def _step(model, ids):
    """One micro-step, same shape as the trainer: bf16 autocast, fused linear CE, backward."""
    inp, tgt = ids[:, :-1], ids[:, 1:].reshape(-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.model(input_ids=inp, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        loss = fused_linear_cross_entropy(h.reshape(-1, h.shape[-1]), model.lm_head.weight, tgt)
    loss.backward()
    model.zero_grad(set_to_none=True)


def _profile(model, ids, warmup, iters):
    for _ in range(warmup):
        _step(model, ids)
    torch.cuda.synchronize()
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]   # CPU too, so keys are aten:: ops
    with profile(activities=acts, record_shapes=False) as prof:
        for _ in range(iters):
            _step(model, ids)
        torch.cuda.synchronize()
    tot = {}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            tot[e.key] = tot.get(e.key, 0.0) + e.self_device_time_total / 1e3 / iters
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)     # trainer micro-batch: 64 / grad_accum 4
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--top", type=int, default=22)
    args = ap.parse_args()

    patchmod.apply(["liger_norm", "liger_rope", "moe", "xsa"])
    torch.manual_seed(0)
    ids = torch.randint(0, SHARED["vocab_size"], (args.batch, args.seq_len + 1), device=DEV)

    runs = {}
    for tag, (ar, sites, carry) in (("ctlexp", ("control", 2, False)),
                                    ("b3carry", ("3", 1, True))):
        model = _build(ar, sites, carry)
        runs[tag] = _profile(model, ids, args.warmup, args.iters)
        del model
        torch.cuda.empty_cache()

    a, b = runs["ctlexp"], runs["b3carry"]
    sa, sb = sum(a.values()), sum(b.values())
    print(f"\ntotal CUDA time per micro-step: ctlexp {sa:.2f} ms | b3carry {sb:.2f} ms "
          f"| delta {sb - sa:+.2f} ms ({(sb / sa - 1) * 100:+.1f}%)")
    print(f"  (trainer does {4} micro-steps per optimizer step -> "
          f"~{(sb - sa) * 4:+.1f} ms/step, vs -33 ms measured)")

    keys = sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, 0.0) - a.get(k, 0.0)))
    print(f"\n{'op':<52}{'ctlexp':>10}{'b3carry':>10}{'delta':>10}")
    print("-" * 82)
    for k in keys[: args.top]:
        x, y = a.get(k, 0.0), b.get(k, 0.0)
        print(f"{k[:69]:<70}{x:>9.2f}{y:>9.2f}{y - x:>+9.2f}")


if __name__ == "__main__":
    main()
