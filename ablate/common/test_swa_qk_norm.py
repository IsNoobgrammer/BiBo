"""--swa_qk_norm false: QK-norm really is gone from the WINDOWED layers and still present on the
global ones, and the change is load-bearing end to end.

The structural check alone is not enough. A flag that reaches the config but never reaches the
module produces a model that trains fine, logs `qk_norm=False`, and is byte-identical to the
control -- an inert arm that costs a full run to discover. That has happened here before (per-head
xsa alpha shipped dead while its parity gate passed), so this asserts three separate things:

  1. windowed layers hold nn.Identity, global layers hold BiBoRMSNorm  (the flag reached the module)
  2. the parameter count drops by EXACTLY 2 * head_dim per windowed layer  (nothing else moved)
  3. removing those norms CHANGES the logits  (the norms were doing work, so the arm is a real arm)

CPU-only, no Triton, no GPU:
    python -m ablate.common.test_swa_qk_norm
"""
from . import _paths  # noqa: F401

import torch
import torch.nn as nn

from .configs import make_bibo_min_config, swa_block_pattern, SHARED
from src.modeling.norm import BiBoRMSNorm
from src.modeling_bibo import BiBoForCausalLM


def _build(qk_norm, pattern):
    torch.manual_seed(0)
    cfg = make_bibo_min_config(num_experts=6, special_pairs=0, use_xsa=True,
                               hybrid_layer_pattern=pattern, sliding_window=128,
                               swa_sink=False, swa_qk_norm=qk_norm)
    return BiBoForCausalLM(cfg).eval(), cfg


def main():
    pattern = swa_block_pattern(SHARED["num_hidden_layers"])
    n_swa = sum(pattern)
    print(f"pattern {pattern} -> {n_swa} windowed layers")

    on, cfg = _build(True, pattern)
    off, _ = _build(False, pattern)

    # (1) the flag reached the module, on the right layers only
    for i, is_swa in enumerate(pattern):
        a = on.model.layers[i].self_attn
        b = off.model.layers[i].self_attn
        if getattr(a, "q_proj", None) is None:      # mlp_only layer, no attention at all
            continue
        assert isinstance(a.q_norm, BiBoRMSNorm) and isinstance(a.k_norm, BiBoRMSNorm), (
            f"layer {i}: control lost QK-norm")
        want = nn.Identity if is_swa else BiBoRMSNorm
        assert isinstance(b.q_norm, want) and isinstance(b.k_norm, want), (
            f"layer {i} (swa={bool(is_swa)}): expected {want.__name__}, got "
            f"{type(b.q_norm).__name__}/{type(b.k_norm).__name__} -- flag did not reach the module")
    print(f"  [1] windowed layers -> Identity, global layers -> BiBoRMSNorm")

    # (2) exactly 2 * head_dim per windowed layer, nothing else
    p_on = sum(p.numel() for p in on.parameters())
    p_off = sum(p.numel() for p in off.parameters())
    want = n_swa * 2 * cfg.head_dim
    assert p_on - p_off == want, (
        f"param delta {p_on - p_off} != expected {want} "
        f"({n_swa} windowed x 2 norms x head_dim {cfg.head_dim}) -- something else changed too")
    print(f"  [2] params {p_on} -> {p_off}, delta {p_on - p_off} == {n_swa} x 2 x {cfg.head_dim}")

    # (3) the removed norms were doing work. Same weights for every shared parameter, so any
    #     logit difference is attributable to the norms alone and not to a different init.
    #     `off` has no q_norm/k_norm on the windowed layers, so those keys arrive UNEXPECTED --
    #     that is the whole point of the arm. Anything else differing would mean the two models
    #     are not otherwise identical and the logit diff below could not be attributed.
    r = off.load_state_dict(on.state_dict(), strict=False)
    assert not r.missing_keys, f"`off` wants keys `on` does not have: {r.missing_keys}"
    stray = [k for k in r.unexpected_keys if "q_norm" not in k and "k_norm" not in k]
    assert not stray, f"non-QK-norm keys differ between the two models: {stray}"
    assert len(r.unexpected_keys) == n_swa * 2, (
        f"expected {n_swa * 2} dropped norm tensors, got {len(r.unexpected_keys)}")

    ids = torch.randint(0, cfg.vocab_size, (2, 256))
    with torch.no_grad():
        y_on = on(input_ids=ids).logits
        y_off = off(input_ids=ids).logits
    d = (y_on - y_off).abs().max().item()
    assert d > 1e-3, (
        f"logits are identical (max abs diff {d:.2e}) -- QK-norm on the windowed layers is "
        f"INERT, so this arm would measure nothing")
    print(f"  [3] max abs logit diff {d:.4f} -- the arm is live")
    print("PASS")


if __name__ == "__main__":
    main()
