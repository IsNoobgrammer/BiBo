"""GPU check: router temperature on the FUSED megakernel path (W/T fed to the norm-router kernel) matches the
eager router (logits / T) -- same top-k selection, same normalised weights. Board geometry, bf16 residual stream.

    python -m ablate.common.test_router_temp_gpu
"""
from ablate.common import _paths
import torch
from ablate.common.models import build_arm
from ablate.common import patches as P
from ablate.common.configs import resolve_swa
pat, win = resolve_swa("block3", 128, 10)
kw = dict(device="cuda", num_experts=64, top_k=6, hybrid_layer_pattern=pat, sliding_window=win,
          moe_overrides={0: {"num_routed_experts": 8, "num_experts_per_tok": 8, "moe_intermediate_size": 576}},
          mlp_only_layers=[], use_xsa=True, attn_res=3, attn_res_sites=1, attn_res_carry=True,
          attn_res_carry_per_dim=True, attn_res_carry_scale="sigmoid", router_temperature=2.0, bf16_residual_stream=True)
torch.manual_seed(0)
m, c = build_arm("bibo_min", **kw)
P.RADIAL_P = "sigmoid"; P.EXPERT_ACT = "radial"
P.apply(["liger_norm", "liger_rope", "moe", "megakernel", "xsa"])
m.eval()
cap = {}
L = m.model.layers[3]
def hook(mod, args):
    cap["x"], cap["idx"], cap["w"] = args[0].detach(), args[1].detach(), args[2].detach()
L.mlp.experts.register_forward_pre_hook(hook)
ids = torch.randint(1, 81920, (2, 256), device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    m(ids)
x = cap["x"].reshape(-1, cap["x"].shape[-1]).float()
ln, g = L.post_attention_layernorm, L.mlp.gate
hn = torch.nn.functional.rms_norm(x, (x.shape[-1],), ln.weight.float(), eps=ln.variance_epsilon)
s = torch.sigmoid((hn @ g.gate_proj.weight.float().t()) / g.temperature)
sel = s + g.bias.float()
idx = sel.topk(g.top_k, -1).indices
ref = torch.zeros_like(s).scatter_(1, idx, s.gather(1, idx)); ref = ref / ref.sum(-1, keepdim=True)
got = torch.zeros_like(s).scatter_(1, cap["idx"].reshape(-1, g.top_k).long(), cap["w"].reshape(-1, g.top_k).float())
agree = (ref > 0).eq(got > 0).all(-1)
assert agree.all() and (ref - got)[agree].abs().max() < 1e-5, "fused router != eager at T=2"
print(f"T={g.temperature}: selection agree {100*agree.float().mean():.3f}% | weight max err (agreeing tokens) {(ref-got)[agree].abs().max():.2e} | top1 fused {got.max(-1).values.mean():.4f} eager {ref.max(-1).values.mean():.4f}")
