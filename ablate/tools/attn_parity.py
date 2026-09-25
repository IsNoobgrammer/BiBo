"""Model-level parity of the fused attention path (--attn_kernel fused) inside real BiBoAttention
modules: one global layer (NoPE, qk-norm, XSA) and one sliding-window layer (RoPE, qk-norm, XSA).

    python -m ablate.tools.attn_parity

Three runs of the same module, same weights, same input:
  reference   eager path (flex / SDPA flavour modules) in fp32
  eager bf16  eager path under bf16 autocast (what training ran before)
  fused bf16  FUSED_ATTN=True under bf16 autocast
Pass = the fused path's output and EVERY parameter grad are within 1.5x of eager bf16's error
vs the fp32 reference, and the fused backward is bitwise repeatable.
"""
import copy
import importlib
import sys

import torch

sys.path.insert(0, ".")
from ablate.common import _paths  # noqa: F401,E402
from ablate.common.configs import make_bibo_min_config, swa_block_pattern  # noqa: E402

BASE = importlib.import_module("src.modeling.attn.base")
EMB = importlib.import_module("src.modeling.embed")


def run(attn, x, pe, fused, dtype):
    BASE.FUSED_ATTN = fused
    a = copy.deepcopy(attn)
    xi = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        out, _ = a(xi, position_embeddings=pe, attention_mask=None)
    torch.manual_seed(1)
    g = torch.randn(out.shape, device=out.device)
    out.float().backward(g)
    grads = {n: p.grad.detach().float().clone() for n, p in a.named_parameters() if p.grad is not None}
    grads["d_hidden"] = xi.grad.detach().float()
    return out.detach().float(), grads


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def main():
    torch.manual_seed(0)
    cfg = make_bibo_min_config(use_xsa=True, xsa_alpha_init=0.3, hybrid_layer_pattern=swa_block_pattern(10),
                               sliding_window=128, swa_qk_norm=True)
    ok = True
    for layer_idx, name in ((0, "global (NoPE)"), (1, "sliding window + rope")):
        attn = BASE.BiBoAttention(cfg, layer_idx=layer_idx).cuda().train()
        with torch.no_grad():
            for p in attn.parameters():
                if p.dim() == 1:
                    p.add_(0.1 * torch.randn_like(p))
        B, S = 4, 1024
        x = torch.randn(B, S, cfg.hidden_size, device="cuda")
        rot = EMB.BiBoRotaryEmbedding(cfg.head_dim, base=getattr(cfg, "rope_theta", 10000) or 10000).cuda()
        pos = torch.arange(S, device="cuda")[None].expand(B, S)
        pe = rot(x, pos)
        ref_o, ref_g = run(attn, x, pe, False, torch.float32)
        eag_o, eag_g = run(attn, x, pe, False, torch.bfloat16)
        fus_o, fus_g = run(attn, x, pe, True, torch.bfloat16)
        rep_o, rep_g = run(attn, x, pe, True, torch.bfloat16)
        repeat = torch.equal(fus_o, rep_o) and all(torch.equal(fus_g[k], rep_g[k]) for k in fus_g)
        print(f"== layer {layer_idx}: {name} | fused bwd repeatable {repeat}")
        ok &= repeat
        for k in ["out"] + sorted(ref_g):
            r = ref_o if k == "out" else ref_g[k]
            e = eag_o if k == "out" else eag_g.get(k)
            f = fus_o if k == "out" else fus_g.get(k)
            if f is None or e is None:
                print(f"   {k:32s} MISSING (fused {f is not None}, eager {e is not None})")
                ok = False
                continue
            ef, ee = rel(f, r), rel(e, r)
            good = ef <= 1.5 * ee + 1e-6
            ok &= good
            print(f"   {k:32s} fused {ef:.2e}   eager bf16 {ee:.2e}   {'OK' if good else 'WORSE'}")
    BASE.FUSED_ATTN = False
    print("ATTN MODEL PARITY", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
