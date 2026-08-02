"""--act silu really reaches the kernel, and radial really goes inert when it does.

The failure this exists to catch is the quiet one: the flag lands in argparse, prints
`expert activation = silu`, and the experts keep computing radial. The run then looks fine, costs
an hour, and measures the control against itself. `--swa_qk_norm` shipped exactly that way earlier
today (config reached the module, never reached build_arm), so the assertions here are about
OBSERVABLE CONSEQUENCES, not about the flag having been set:

  1. the GLU experts get kernel act code 0, the specials keep 3/4
  2. the output equals SwiGLU  silu(gate) * up  computed independently
  3. it DIFFERS from what radial produces on identical weights and input
  4. radial_theta receives NO gradient -- the exponent cannot move, which is the same thing the
     training log shows as `p=0.500[0.50,0.50]` never budging

Needs a GPU (Triton):
    python -m ablate.common.test_act_silu
"""
from . import _paths  # noqa: F401

import torch
import torch.nn.functional as F

from . import patches as P
from src.configuration_bibo import BiBoConfig
from src.modeling.ffn.moe import BiBoFusedExperts


def main():
    assert torch.cuda.is_available(), "needs a GPU: this exercises the Triton expert kernel"
    P.apply(["moe"])
    torch.manual_seed(0)

    H, I, N = 64, 128, 96
    cfg = BiBoConfig(hidden_size=H, num_attention_heads=4, num_key_value_heads=2,
                     moe_intermediate_size=I, num_routed_experts=1, special_expert_pairs=0,
                     pos_identity_expert=False, neg_identity_expert=False,
                     num_experts_per_tok=1)
    m = BiBoFusedExperts(cfg).cuda().to(torch.bfloat16)
    assert m.num_glu_experts == 1, m.num_glu_experts

    x = torch.randn(N, H, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(N, 1, dtype=torch.long, device="cuda")
    wt = torch.ones(N, 1, device="cuda", dtype=torch.bfloat16)

    def run(act):
        P.EXPERT_ACT = act
        m._act_codes = None                      # cached per module; force a rebuild
        m.radial_theta.grad = None
        out = m(x, idx, wt)
        out.float().square().sum().backward()
        return out.detach().float(), m._act_codes.clone(), m.radial_theta.grad

    out_r, codes_r, grad_r = run("radial")
    out_s, codes_s, grad_s = run("silu")
    P.EXPERT_ACT = "radial"                       # leave the module as we found it

    # (1) the code the kernel is handed
    assert codes_r.tolist() == [8], f"radial should be act code 8, got {codes_r.tolist()}"
    assert codes_s.tolist() == [0], f"silu should be act code 0, got {codes_s.tolist()}"
    print(f"  [1] act codes: radial={codes_r.tolist()} silu={codes_s.tolist()}")

    # (2) silu mode == plain SwiGLU, computed independently of the kernel
    gate, up = F.linear(x.float(), m.gate_up_proj[0].float()).chunk(2, dim=-1)
    ref = F.linear(F.silu(gate) * up, m.down_proj[0].float())
    err = (out_s - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, f"silu output does not match SwiGLU reference: rel err {err:.3e}"
    print(f"  [2] matches silu(gate)*up reference, rel err {err:.2e} (bf16)")

    # (3) and it is genuinely a different function from radial
    d = (out_s - out_r).abs().max().item() / out_r.abs().max().item()
    assert d > 1e-2, f"silu and radial outputs are the same (rel diff {d:.3e}) -- arm is inert"
    print(f"  [3] differs from radial, rel diff {d:.2e}")

    # (4) the exponent is unreachable under silu. This is the one that catches a flag that set
    #     itself but changed nothing: radial still running would keep theta differentiable.
    assert grad_r is not None and grad_r.abs().sum() > 0, (
        "radial did not put a gradient on radial_theta -- the control itself is broken")
    assert grad_s is None or grad_s.abs().sum() == 0, (
        f"radial_theta still takes gradient under --act silu ({grad_s}) -- the exponent is live, "
        f"so the kernel is not running plain SwiGLU")
    print(f"  [4] radial_theta grad: radial={grad_r.abs().sum():.4g}, silu={grad_s}")
    print("PASS")


if __name__ == "__main__":
    main()
