"""WHOLE-MODEL forward + backward audit: every parameter gradient of the production stack (all tkf
kernels, bf16) against an fp32-eager reference (no kernels), with bf16-eager (no kernels) as the
floor. Same weights (same seed), same first batch, three separate processes:

    ref    python -m ablate.common.train <board> --precision fp32 --patches "" --attn_kernel flex \\
               --global_attn sdpa --fused_res_add false --bf16_residual_stream false --grad_audit ref.pt
    floor  python -m ablate.common.train <board> --patches "" --attn_kernel flex --global_attn sdpa \\
               --fused_res_add false --grad_audit floor.pt
    prod   python -m ablate.common.train <board> --grad_audit prod.pt
    python -m ablate.tools.grad_audit ref.pt floor.pt prod.pt

--patches "" also turns the AttnRes depth-mix kernel off (the eager torch path), so "no kernels" means
none of ours at all. The balancing-bias update is disabled for the audit step (it mutates router
state inside forward). Top-k routing is discrete: a token whose expert choice flips vs fp32 moves
that expert's gradient by a whole contribution in EITHER bf16 stack, so the routing agreement with
the reference is printed next to the gradient errors. The comparison that matters is prod vs floor
against the same reference, per parameter group.
"""
import collections
import re
import sys

import torch


def dump(model, gen, loss_fn, amp, path, kernels_off):
    import exp.modeling_bibo as E
    if kernels_off:
        E._HAS_FUSED_AR = False
        E._HAS_FUSED_RES_ADD = False
    routes = {}
    for name, m in model.named_modules():
        if hasattr(m, "bias_update_factor"):
            m.bias_update_factor = 0.0
        if name.endswith(".gate") and hasattr(m, "gate_proj"):
            def _hook(mod, inp, out, _n=name):
                routes[_n] = out[0].detach().reshape(-1, out[0].shape[-1]).sort(-1).values.cpu()
            m.register_forward_hook(_hook)
    model.train()
    ids = next(gen)
    model.zero_grad(set_to_none=True)
    with amp:
        loss = loss_fn(ids)
    loss.backward()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    torch.save({"loss": float(loss), "grads": grads, "routes": routes,
                "ids_sum": int(ids.sum())}, path)
    print(f"[grad_audit] wrote {path}: loss {float(loss):.6f}, {len(grads)} grads, "
          f"{len(routes)} routers, batch checksum {int(ids.sum())}", flush=True)


_GROUPS = [
    ("attn q/k/v/o proj", r"self_attn\.(q|k|v|o)_proj"),
    ("attn qk-norm", r"self_attn\.(q|k)_norm"),
    ("attn xsa alpha", r"xsa"),
    ("moe experts gate_up (L1-9)", r"layers\.[1-9]\.mlp\.experts\.gate_up_proj"),
    ("moe experts down (L1-9)", r"layers\.[1-9]\.mlp\.experts\.down_proj"),
    ("moe radial theta", r"radial_theta"),
    ("L0 ensemble experts", r"layers\.0\.mlp\.experts\.(gate_up|down)_proj"),
    ("router gate_proj", r"gate\.gate_proj"),
    ("norms", r"(layernorm|_norm\.weight|\.norm\.weight)"),
    ("attn-res proj/norm", r"attention_res|res_proj|res_norm"),
    ("carry theta", r"carry_theta"),
    ("embed / lm_head", r"embed_tokens|lm_head"),
]


def _group(name):
    for g, pat in _GROUPS:
        if re.search(pat, name):
            return g
    return "other"


def compare(ref_p, floor_p, prod_p):
    R, F, P = (torch.load(p) for p in (ref_p, floor_p, prod_p))
    assert R["ids_sum"] == F["ids_sum"] == P["ids_sum"], "the three runs saw different batches"
    print(f"loss   ref {R['loss']:.6f}   floor(bf16 eager) {F['loss']:.6f} (d {F['loss'] - R['loss']:+.2e})"
          f"   prod(kernels) {P['loss']:.6f} (d {P['loss'] - R['loss']:+.2e})")
    print("routing agreement with fp32 (fraction of token top-k SETS identical), per MoE layer:")
    for n in sorted(R["routes"]):
        r, f, p = R["routes"][n], F["routes"].get(n), P["routes"].get(n)
        af = (f == r).all(-1).float().mean().item() if f is not None else float("nan")
        apr = (p == r).all(-1).float().mean().item() if p is not None else float("nan")
        print(f"   {n:24s} floor {af:.4f}   prod {apr:.4f}")
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0, []])
    worst = []
    missing = [n for n in R["grads"] if n not in P["grads"]]
    for n, g in R["grads"].items():
        if n not in P["grads"] or n not in F["grads"]:
            continue
        rn = g.norm().item()
        if rn == 0:
            continue
        ef = (F["grads"][n] - g).norm().item() / rn
        ep = (P["grads"][n] - g).norm().item() / rn
        a = agg[_group(n)]
        a[0] += ef; a[1] += ep; a[2] = max(a[2], ep / max(ef, 1e-12)); a[3] += 1
        worst.append((ep / max(ef, 1e-12), n, ef, ep))
    print(f"\nparameter gradients, rel err vs fp32 eager (mean over the group), prod/floor worst ratio:")
    print(f"   {'group':30s} {'n':>4s} {'floor (bf16 eager)':>19s} {'prod (kernels)':>15s} {'worst prod/floor':>17s}")
    for g, (sf, sp, wr, c, _) in sorted(agg.items()):
        print(f"   {g:30s} {c:4d} {sf / c:19.3e} {sp / c:15.3e} {wr:17.2f}")
    print("\nten worst parameters by prod/floor ratio:")
    for r_, n, ef, ep in sorted(worst, reverse=True)[:10]:
        print(f"   {r_:6.2f}  {n:60s} floor {ef:.3e}  prod {ep:.3e}")
    if missing:
        print(f"\nMISSING in prod (no grad!): {missing[:10]}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(compare(*sys.argv[1:4]))
