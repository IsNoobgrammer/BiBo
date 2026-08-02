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

from .configs import swa_block_pattern, SHARED
from .models import build_arm

HID = SHARED["hidden_size"]
NL = SHARED["num_hidden_layers"]


def _build(attn_res, pattern):
    torch.manual_seed(0)
    model, cfg = build_arm("bibo_min", device="cpu", dtype=torch.float32,
                           num_experts=6, special_pairs=0, use_xsa=True,
                           hybrid_layer_pattern=pattern, sliding_window=128,
                           swa_sink=False, attn_res=attn_res)
    return model.eval(), cfg


def main():
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
    print("PASS")


if __name__ == "__main__":
    main()
