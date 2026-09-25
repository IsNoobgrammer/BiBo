"""MoE expert kernels vs the PURE-PYTORCH src eager experts (BiBoFusedExperts.forward: per-expert
loop, radial NormSiLU in torch), with routing FIXED (same indices/weights in every arm) so only the
kernel arithmetic is measured. Forward output and every gradient -- d_hidden, d_weights,
d_gate_up_proj, d_down_proj, d_radial_theta -- over two accumulated micro-batches.

    ref    src eager, fp32, autocast off              (reference)
    floor  src eager under bf16 autocast              (plain PyTorch bf16)
    prod   the patched tkf path (moe_per_expert) under bf16 autocast, exactly as training calls it

Cases: the board MoE layer (64 experts, top-6, I=768) and the L0 all-active ensemble (8/8, I=576).
PASS: prod error <= 1.05 x floor error for every tensor, and prod bitwise repeatable.

    python -m ablate.tools.parity_moe_vs_eager
"""
from ablate.common import _paths  # noqa: F401
import sys
import types

import torch

from src.modeling.ffn.moe import BiBoFusedExperts
from ablate.common import patches as P
from kernels.sm120.moe import moe_per_expert

dev = "cuda"
H, N = 512, 16384


def experts(E, I, seed=0):
    cfg = types.SimpleNamespace(num_glu_experts=E, special_expert_pairs=0, num_routed_experts=E,
                                hidden_size=H, moe_intermediate_size=I, initializer_range=0.02,
                                num_pos_identity_experts=0, num_neg_identity_experts=0)
    torch.manual_seed(seed)
    m = BiBoFusedExperts(cfg).to(dev)
    with torch.no_grad():
        m.radial_theta.normal_(0.0, 0.5)             # off init, so d_theta is a real test
    return m


def run(case, mode):
    E, K, I = case
    m = experts(E, I)
    out_all = {}
    for micro in range(2):
        g = torch.Generator(device=dev).manual_seed(10 + micro)
        # the model's layout: the MoE input is the bf16 stream and its upstream grad is bf16 too.
        # The fp32 reference gets the SAME bf16-valued tensors upcast, so input quantization is
        # shared and only the kernel arithmetic differs. (fp32 inputs made eager keep an fp32
        # output / d_hidden while the kernel emits the stream dtype -- a dtype, not an accuracy,
        # difference that read as 5-10%.)
        xb = (torch.randn(N, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        x = (xb.float() if mode == "ref" else xb).detach().requires_grad_()
        sc = torch.sigmoid(torch.randn(N, E, device=dev, generator=g))
        w_, idx = sc.topk(K, dim=-1)
        wt = (w_ / w_.sum(-1, keepdim=True)).detach().requires_grad_()   # sum-norm, as the router
        go = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16)
        if mode == "prod":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = moe_per_expert(x, idx, wt, m.gate_up_proj, m.down_proj,
                                   P._expert_codes(m, dev, torch.bfloat16), act_params=P._act_params(m))
        elif mode == "floor":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(x, idx, wt)
        else:
            y = m(x, idx, wt)
        y.backward(go.to(y.dtype))
        out_all[f"out[{micro}]"] = y.detach().float()
        out_all[f"d_hidden[{micro}]"] = x.grad.float()
        out_all[f"d_weights[{micro}]"] = wt.grad.float()
    out_all["d_gate_up_proj"] = m.gate_up_proj.grad.float()
    out_all["d_down_proj"] = m.down_proj.grad.float()
    out_all["d_radial_theta"] = m.radial_theta.grad.float()
    return out_all


def main():
    P.EXPERT_ACT, P.RADIAL_P = "radial", "sigmoid"      # the board: --act radial --radial_p sigmoid
    ok = True
    for case in ((64, 6, 768), (8, 8, 576)):
        ref, fl, pr, pr2 = run(case, "ref"), run(case, "floor"), run(case, "prod"), run(case, "prod")
        print(f"== E={case[0]} top-{case[1]} I={case[2]}  (fixed routing, 2 micro-batches, N={N})")
        for k in ref:
            rn = ref[k].norm().item() or 1.0
            ef = (fl[k] - ref[k]).norm().item() / rn
            ep = (pr[k] - ref[k]).norm().item() / rn
            rep = torch.equal(pr[k], pr2[k])
            good = rep and ep <= ef * 1.05 + 1e-7
            ok &= good
            print(f"   {k:16s} rel err vs fp32 eager: bf16 eager {ef:.3e}  tkf {ep:.3e}  "
                  f"repeat {'bitwise' if rep else 'DIFFERS'}  {'OK' if good else 'WORSE'}")
    print("MOE vs EAGER PASS" if ok else "MOE vs EAGER FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
