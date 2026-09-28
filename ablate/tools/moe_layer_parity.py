"""Real-layer parity of the MoE training path: a trained checkpoint's MoE layer, fed its REAL input
(captured on holdout text), through the path training runs (megakernel: fused norm + router +
experts, bf16 autocast) vs an fp32 eager chain (RMSNorm -> sigmoid router -> tkf moe_eager).

    python -m ablate.tools.moe_layer_parity <result.json> [--layer 1] [--n_seqs 8]

The reference reuses the kernel's top-k INDICES (only the weights are recomputed in fp32), so a
near-tie token that the bf16 router routes differently does not swamp the comparison; the routing
agreement is reported on its own. Grads compared: input, norm weight, router weight, gate_up, down,
radial theta. Model in eval mode, so the balancing bias never moves.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import importlib
import json

import torch

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from ablate.common import patches as P

K75 = importlib.import_module("kernels.sm75.moe")
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--n_seqs", type=int, default=8)
    a = ap.parse_args()
    c0 = json.load(open(a.result))["config"]
    model, c = load_from_result(a.result)
    L = model.model.layers[a.layer]
    moe = L.mlp
    hold = _val.build_holdout(c0["dataset"], c0["seq_len"], a.n_seqs, DEV)

    cap = {}
    f0 = L._attn_res_mlp_forward

    def grab(x):
        cap["x"] = x.detach().clone()
        return f0(x)
    L._attn_res_mlp_forward = grab
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.model(input_ids=hold[:, :-1], use_cache=False)
    L._attn_res_mlp_forward = f0
    x0 = cap["x"]
    b, s, h = x0.shape
    print(f"[parity] layer {a.layer}: real input {tuple(x0.shape)} {x0.dtype}, rms {x0.float().pow(2).mean().sqrt():.3f}; "
          f"E={moe.gate.num_routed_experts} top{moe.gate.top_k}", flush=True)

    params = {"norm_w": L.post_attention_layernorm.weight, "router_w": moe.gate.gate_proj.weight,
              "gate_up": moe.experts.gate_up_proj, "down": moe.experts.down_proj,
              "theta": moe.experts.radial_theta}
    gy = torch.randn(b, s, h, device=DEV, generator=torch.Generator(device=DEV).manual_seed(1))

    def zero():
        for p in params.values():
            p.grad = None

    # ---- training path (kernel)
    zero()
    route = {}
    # the megakernel path replays the experts' forward PRE-hooks with (flat, idx, weights)
    hk = moe.experts.register_forward_pre_hook(
        lambda m, args: route.update(idx=args[1].detach(), w=args[2].detach()))
    try:
        x = x0.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = L._attn_res_mlp_forward(x)
        (y.float() * gy).sum().backward()
    finally:
        hk.remove()
    assert "idx" in route, "megakernel path did not run (no routing captured)"
    kern = {"y (fwd)": y.detach().float(), "d_x": x.grad.float()}
    kern.update({f"d_{k}": p.grad.float().clone() for k, p in params.items()})

    # ---- fp32 eager reference, same top-k indices
    zero()
    xr = x0.float().clone().requires_grad_(True)
    hn = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + L.post_attention_layernorm.variance_epsilon)
    hn = params["norm_w"] * hn
    scores = torch.sigmoid(hn.reshape(-1, h) @ params["router_w"].t())
    idx = route["idx"].long().reshape(b * s, -1)
    tw = scores.gather(-1, idx)
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)
    codes = P._expert_codes(moe.experts, DEV, torch.float32)
    yr = K75.moe_eager(hn.reshape(-1, h), idx, tw, params["gate_up"], params["down"], codes,
                       act_params=P._act_params(moe.experts)).reshape(b, s, h)
    (yr * gy).sum().backward()
    ref = {"y (fwd)": yr.detach(), "d_x": xr.grad}
    ref.update({f"d_{k}": (p.grad.float() if p.grad is not None else None) for k, p in params.items()})

    # routing agreement of the fp32 router with the kernel's choice
    with torch.no_grad():
        sel = scores + (moe.gate.bias if moe.gate.bias is not None else 0)
        ridx = sel.topk(idx.shape[1], -1).indices.sort(-1).values
        agree = (ridx == idx.sort(-1).values).all(-1).float().mean().item()
        wdiff = ((route["w"].float().reshape(b * s, -1) - tw).abs().max()).item()
    rel = lambda u, v: ((u - v).norm() / v.norm()).item()
    print(f"[parity] routing: fp32 router picks the kernel's top-k set for {agree * 100:.3f}% of tokens; "
          f"max |w_kernel - w_fp32| {wdiff:.2e}")
    print(f"[parity] relative Frobenius error, training path (bf16 kernels) vs fp32 eager:")
    for k in kern:
        r = ref.get(k)
        print(f"   {k:12s} " + ("(no fp32 grad)" if r is None else f"{rel(kern[k], r):.3e}"))
    print("LAYER_PARITY_DONE")


if __name__ == "__main__":
    main()
