"""GPU check: the rank-balance aux loss (patches.RANK_AUX) is LIVE on the fused megakernel path and its gradient
reaches the router like the eager router's would.

    python -m ablate.common.test_rank_aux_gpu

Same weights, one routed layer's input captured from a forward: aux computed from the fused path's routing
weights (gradient through the norm-router kernel's backward) vs aux recomputed from the eager router math
(autograd). The router-weight gradients must match up to the fused backward's bf16 contraction.
"""
from ablate.common import _paths  # noqa: F401
import torch
import torch.nn.functional as F

from ablate.common.models import build_arm
from ablate.common import patches as P
from ablate.common.configs import resolve_swa


def main(rho=0.5):
    pat, win = resolve_swa("block3", 128, 10)
    torch.manual_seed(0)
    m, _ = build_arm("bibo_min", device="cuda", num_experts=64, top_k=6, hybrid_layer_pattern=pat, sliding_window=win,
                     moe_overrides={0: {"num_routed_experts": 8, "num_experts_per_tok": 8, "moe_intermediate_size": 576}},
                     mlp_only_layers=[], use_xsa=True, attn_res=3, attn_res_sites=1, attn_res_carry=True,
                     attn_res_carry_per_dim=True, attn_res_carry_scale="sigmoid", bf16_residual_stream=True)
    P.RADIAL_P, P.EXPERT_ACT = "sigmoid", "radial"
    P.apply(["liger_norm", "liger_rope", "moe", "megakernel", "xsa"])
    L = m.model.layers[3]
    g, ln = L.mlp.gate, L.post_attention_layernorm
    cap = {}
    L.mlp.experts.register_forward_pre_hook(lambda mod, a: cap.setdefault("x", a[0].detach()))
    ids = torch.randint(1, 81920, (2, 256), device="cuda")

    # fused: aux from the kernel's routing weights, backward through the fused router
    P.RANK_AUX["rho"], P.RANK_AUX["acc"] = rho, []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(ids)
    assert len(P.RANK_AUX["acc"]) == 9, f"expected one aux per routed layer, got {len(P.RANK_AUX['acc'])}"
    aux_f = P.RANK_AUX["acc"][2]                 # layer 3 = 3rd routed layer
    m.zero_grad(set_to_none=True)
    aux_f.backward()
    g_f = g.gate_proj.weight.grad.float().clone()
    P.RANK_AUX["rho"], P.RANK_AUX["acc"] = 0.0, []

    # eager reference on the same captured input
    x = cap["x"].reshape(-1, cap["x"].shape[-1]).float()
    W = g.gate_proj.weight.detach().float().requires_grad_(True)
    hn = F.rms_norm(x, (x.shape[-1],), ln.weight.detach().float(), eps=ln.variance_epsilon)
    s = torch.sigmoid(hn @ W.t())
    idx = (s + g.bias.float()).topk(g.top_k, -1).indices
    w = s.gather(1, idx)
    w = (w / w.sum(-1, keepdim=True)).sort(-1, descending=True).values
    aux_e = torch.relu(rho * w[:, :3].sum(-1) - w[:, 3:].sum(-1)).mean()
    aux_e.backward()
    g_e = W.grad
    rel = ((g_f - g_e).norm() / g_e.norm()).item()
    print(f"rho={rho}: aux fused {aux_f.item():.5f} eager {aux_e.item():.5f} | router grad rel err {rel:.2e} "
          f"| grad norm {g_e.norm().item():.3e}")
    assert abs(aux_f.item() - aux_e.item()) < 1e-3 and rel < 5e-2 and g_e.norm() > 0, "rank aux not live / wrong grad"
    print("RANK_AUX_GPU PASS")


if __name__ == "__main__":
    main()
