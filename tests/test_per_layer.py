"""The colour encoding of the expert-load map, and that routers are keyed by real layer index."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

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
