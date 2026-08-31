"""The colour encoding of the expert-load map, and that routers are keyed by real layer index."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from ablate.common.per_layer import _hsv_to_rgb, load_rgb


def test_hsv_corners():
    rgb = _hsv_to_rgb(np.array([0.0, 1 / 3, 2 / 3]), np.ones(3), np.ones(3))
    assert list(rgb[0]) == [255, 0, 0] and list(rgb[1]) == [0, 255, 0] and list(rgb[2]) == [0, 0, 255]


def _px(img, row, col, cell=10):
    return img[row * cell + 1, col * cell + 1][:3]


def test_load_map_encoding():
    E = 8
    uniform = [10] * E
    hot = [1] * (E - 1) + [400]          # last expert >>4x uniform -> saturated red
    dead = [0] + [10] * (E - 1)
    img, _ = load_rgb({0: uniform, 1: hot, 2: dead})
    r, g, b = _px(img, 0, 0)
    assert r > 240 and g > 240 and b > 240, "uniform must be white"
    r, g, b = _px(img, 1, E - 1)
    assert r > 200 and g < 40 and b < 40, "overloaded must be red"
    assert tuple(_px(img, 2, 0)) == (0, 0, 0), "dead expert must be black"
    r, g, b = _px(img, 1, 0)
    assert b > r and b > g, "starved must be blue"


def test_train_diagnostics_go_to_interp():
    from ablate.common.train import _interp
    out = _interp({"train/router/layer_3/max_load": 1, "train/attn_res_s/L0": 2,
                   "grad/norm/layers.mlp.gate_proj": 3, "grad/norm_min_over_tensors": 4})
    assert out == {"interp/router/layer_3/max_load": 1, "interp/attn_res_s/L0": 2,
                   "grad/norm/layers.mlp.gate_proj": 3, "grad/norm_min_over_tensors": 4}


@pytest.mark.parametrize("attn_res", ["off", "3"])
def test_moe_override_changes_the_built_layer(attn_res):
    """BOTH model paths. attn_res != "off" builds exp/modeling_bibo.py, which has its own layer
    constructor -- it ignored the override at first, so the run reported matched params while
    training an unmodified L0. Every arm in this round uses attn_res, so "off" alone proves
    nothing."""
    from ablate.common.models import build_arm, count_params
    import torch

    kw = dict(device="cpu", dtype=torch.float32, mlp_only_layers=[], num_experts=64, top_k=6,
              attn_res=attn_res, attn_res_sites=1)
    base, _ = build_arm("bibo_min", **kw)
    over, _ = build_arm("bibo_min", moe_overrides={
        0: {"num_routed_experts": 32, "num_experts_per_tok": 3, "moe_intermediate_size": 1536}}, **kw)

    l0, l1 = over.model.layers[0].mlp, over.model.layers[1].mlp
    assert l0.gate.num_routed_experts == 32 and l0.gate.top_k == 3
    assert tuple(l0.experts.gate_up_proj.shape[:2]) == (32, 3072), "E and WIDTH must both change"
    assert tuple(l1.experts.gate_up_proj.shape[:2]) == (64, 1536), "layer 1 weights untouched"
    assert l1.gate.num_routed_experts == 64 and l1.gate.top_k == 6, "layer 1 must be untouched"

    # active params matched to <1%: E/top_k is pinned by holding both totals fixed
    ta, _, aa = count_params(base)
    tb, _, ab = count_params(over)
    assert abs(ab - aa) / aa < 0.01, ("active not matched", aa, ab)
    assert abs(tb - ta) / ta < 0.01, ("total not matched", ta, tb)
