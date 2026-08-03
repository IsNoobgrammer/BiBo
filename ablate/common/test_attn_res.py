"""exp/ Attention Residuals: the control is really a control, and the two block granularities are
really different models.

Three things have to hold before an AttnRes arm means anything:

  1. exp(attn_res=control) == src, EXACTLY -- same state_dict keys, same logits on the same
     weights. If it does, the existing SWA128+XSA run (D1, bpb 0.67312) already serves as the
     control and we do not spend an hour re-running it. If it does not, exp/ has drifted from src
     and every AttnRes number would be confounded by that drift instead of measuring AttnRes.

  2. AttnRes adds exactly 4*hidden per decoder layer (two sites x (RMSNorm gain + pseudo-query))
     plus 2*hidden at the trunk output -- and that count is IDENTICAL for block=1 and block=3,
     because block size changes memory and compute, not parameters. A param difference between the
     two would mean the block plumbing is allocating something it should not.

  3. block=1 (per-layer / Full) and block=3 (one block per [G,S,S]) both differ from the control
     AND from each other. Same params, different function.

CPU, fp32, no GPU:
    python -m ablate.common.test_attn_res
"""
from . import _paths  # noqa: F401

import torch
import torch.nn as nn

from .configs import swa_block_pattern, SHARED
from .models import build_arm
from src.modeling.norm import BiBoRMSNorm
from exp.modeling_bibo import apply_attention_residual

HID = SHARED["hidden_size"]
NL = SHARED["num_hidden_layers"]


def _build(attn_res, pattern, sites=2):
    torch.manual_seed(0)
    model, cfg = build_arm("bibo_min", device="cpu", dtype=torch.float32,
                           num_experts=6, special_pairs=0, use_xsa=True,
                           hybrid_layer_pattern=pattern, sliding_window=128,
                           attn_res=attn_res, attn_res_sites=sites)
    return model.eval(), cfg


def _naive_attn_res(prefix_sum, block_residual, projection, norm):
    """The ORIGINAL formula, written the obvious way: normalize the values, scale by the folded
    weight, sum. apply_attention_residual now factors the RMS out of the contraction to avoid
    materializing two (tokens, blocks+1, hidden) fp32 tensors per site; this is the reference that
    pins the rewrite to the same numbers."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    variance = values.square().mean(dim=-1, keepdim=True)
    keys = values * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    scores = (keys * score_weight).sum(dim=-1)
    return torch.matmul(scores.softmax(dim=-1).unsqueeze(1), values).squeeze(1)


def _check_score_math():
    """(0) the optimized contraction == the naive one."""
    torch.manual_seed(3)
    T, B, H = 512, 5, 512
    prefix = torch.randn(T, H)
    blocks = torch.randn(T, B, H)
    norm = BiBoRMSNorm(H, eps=1e-6)
    proj = nn.Linear(H, 1, bias=False)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    got = apply_attention_residual(prefix, blocks, proj, norm)
    want = _naive_attn_res(prefix, blocks, proj, norm)
    err = (got.float() - want).abs().max().item() / want.abs().max().item()
    assert err < 1e-5, f"optimized AttnRes diverges from the naive formula: rel err {err:.3e}"
    print(f"  [0] optimized score math == naive formula, rel err {err:.2e}")


def main():
    _check_score_math()
    pattern = swa_block_pattern(NL)
    ids = torch.randint(0, SHARED["vocab_size"], (2, 256))

    src_m, _ = _build("off", pattern)
    ctl_m, _ = _build("control", pattern)

    # (1) the control is the stable model, bit for bit
    ks, kc = set(src_m.state_dict()), set(ctl_m.state_dict())
    assert ks == kc, (f"exp control has drifted from src: only-in-src={sorted(ks - kc)[:6]}, "
                      f"only-in-exp={sorted(kc - ks)[:6]}")
    ctl_m.load_state_dict(src_m.state_dict(), strict=True)
    with torch.no_grad():
        y_src = src_m(input_ids=ids).logits
        y_ctl = ctl_m(input_ids=ids).logits
    d = (y_src - y_ctl).abs().max().item()
    assert d < 1e-5, (f"exp(control) != src, max abs logit diff {d:.3e}. The AttnRes control is "
                      f"NOT the stable model, so D1 cannot be used as the baseline.")
    print(f"  [1] exp(control) == src: {len(ks)} keys match, max logit diff {d:.2e}")

    # (2) parameter cost, and that it does not depend on block size
    n_ctl = sum(p.numel() for p in ctl_m.parameters())
    sizes = {}
    for bs in (1, 3):
        m, _ = _build(str(bs), pattern)
        m.load_state_dict(src_m.state_dict(), strict=False)   # AttnRes params keep their init
        sizes[bs] = (m, sum(p.numel() for p in m.parameters()))
    want = NL * 4 * HID + 2 * HID
    for bs, (_, n) in sizes.items():
        assert n - n_ctl == want, (
            f"block={bs}: AttnRes added {n - n_ctl} params, expected {want} "
            f"({NL} layers x 4 x {HID} + 2 x {HID})")
    assert sizes[1][1] == sizes[3][1], "block size must not change the parameter count"
    print(f"  [2] AttnRes adds {want} params ({want / n_ctl * 100:.3f}%), identical for block 1 and 3")

    # (3) different granularity, different function
    with torch.no_grad():
        y1 = sizes[1][0](input_ids=ids).logits
        y3 = sizes[3][0](input_ids=ids).logits
    for tag, y in (("block=1", y1), ("block=3", y3)):
        dd = (y - y_ctl).abs().max().item()
        assert dd > 1e-3, f"{tag} logits match the control ({dd:.2e}) -- AttnRes is inert"
    d13 = (y1 - y3).abs().max().item()
    assert d13 > 1e-3, f"block=1 and block=3 are the same model ({d13:.2e})"
    print(f"  [3] vs control: block1 {(y1 - y_ctl).abs().max():.3f}, "
          f"block3 {(y3 - y_ctl).abs().max():.3f}; block1 vs block3 {d13:.3f}")

    # (4) sites=1 halves the AttnRes parameters and is a different model from sites=2
    m1, _ = _build("3", pattern, sites=1)
    m1.load_state_dict(src_m.state_dict(), strict=False)
    n1 = sum(p.numel() for p in m1.parameters())
    want1 = NL * 2 * HID + 2 * HID
    assert n1 - n_ctl == want1, (
        f"sites=1 added {n1 - n_ctl} params, expected {want1} ({NL} layers x 2 x {HID} + 2 x {HID})")
    with torch.no_grad():
        ys1 = m1(input_ids=ids).logits
    assert (ys1 - y3).abs().max().item() > 1e-3, "sites=1 and sites=2 are the same model"
    assert (ys1 - y_ctl).abs().max().item() > 1e-3, "sites=1 matches the control -- inert"
    print(f"  [4] sites=1 adds {want1} params (half of {want}); differs from sites=2 by "
          f"{(ys1 - y3).abs().max():.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
