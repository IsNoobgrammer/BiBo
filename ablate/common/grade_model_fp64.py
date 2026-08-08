"""WHOLE-MODEL grade: is the fused stack closer to fp64 truth than the eager stack?

Kernel-level grades prove each kernel beats eager in isolation. They do not prove the assembled
model does -- errors compose across 10 layers and a 64-expert MoE, and a kernel that is better
alone could still land the model further from truth. This measures the thing we actually care
about: same weights, same inputs, three forward passes.

    truth   fp64 everywhere, both kernels OFF (pure torch)
    eager   fp32, both kernels OFF
    fused   fp32, attn_res AND residual_add kernels ON

and reports relative error against truth for each. The fused stack must be closer.

Run in FP32 rather than under bf16 autocast on purpose: under autocast the error is dominated by
bf16 attention and MoE, which neither kernel touches, and that swamps the difference being
measured. The bf16 pass is reported too, as the realistic-setting sanity check.

    python -m ablate.common.grade_model_fp64
"""
from . import _paths  # noqa: F401

import torch

import exp.modeling_bibo as E
from .models import build_arm
from .configs import swa_block_pattern


def _run(model, ids, kernels, autocast=False):
    E._HAS_FUSED_AR = kernels
    E._HAS_FUSED_RES_ADD = kernels
    snap = {n: b.detach().clone() for n, b in model.named_buffers()}
    with torch.no_grad():
        for n, b in model.named_buffers():          # the MoE mutates gate.bias every forward
            b.copy_(snap[n])
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out = model.model(input_ids=ids, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return h.detach().double()


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    NL = 10
    pat = swa_block_pattern(NL)
    # No patches: liger / fused-CE / fused-MoE have no fp64 path, and they are not what is being
    # graded. This isolates the two AttnRes kernels, which is the whole point.
    cfg = dict(num_experts=8, top_k=2, special_pairs=0, use_xsa=True,
               hybrid_layer_pattern=pat, sliding_window=128, attn_res="3", attn_res_sites=1,
               attn_res_carry=True, attn_res_fp32_stream=True,
               attn_res_carry_scale="raw", attn_res_emb_term=True)

    torch.manual_seed(42069)
    m64, _ = build_arm("bibo_min", device="cuda", dtype=torch.float64, **cfg)
    m64.eval()
    with torch.no_grad():
        for n, p in m64.named_parameters():
            if "attn_res_carry_theta" in n:
                p.fill_(0.6)
            if "attn_res_emb_theta" in n:
                p.fill_(0.4)
    sd = {k: v.clone() for k, v in m64.state_dict().items()}

    torch.manual_seed(42069)
    m32, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32, **cfg)
    m32.eval()
    m32.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)

    ids = torch.randint(0, 1000, (2, 256), device="cuda")

    truth = _run(m64, ids, kernels=False)
    print(f"truth: fp64, kernels off   |h| max {truth.abs().max().item():.4f}")

    def rel(x):
        d = (x - truth).abs()
        den = max(truth.abs().max().item(), 1e-300)
        return d.mean().item() / den, d.max().item() / den

    for label, ac in (("fp32 (isolates the kernels)", False),
                      ("bf16 autocast (realistic)", True)):
        e_mu, e_mx = rel(_run(m32, ids, kernels=False, autocast=ac))
        k_mu, k_mx = rel(_run(m32, ids, kernels=True, autocast=ac))
        verdict = "FUSED CLOSER" if k_mu < e_mu else "EAGER CLOSER  <-- REGRESSION"
        print(f"\n{label}")
        print(f"  eager  mean {e_mu:.4e}   max {e_mx:.4e}")
        print(f"  fused  mean {k_mu:.4e}   max {k_mx:.4e}")
        print(f"  ratio  mean {k_mu / e_mu:.4f}      max {k_mx / e_mx:.4f}   {verdict}")
        if ac is False:
            assert k_mu <= e_mu, "fused stack is FURTHER from fp64 truth than eager in fp32"
    print("\nPASS: the fused stack is closer to fp64 truth than eager")


if __name__ == "__main__":
    main()
