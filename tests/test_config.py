"""BiBoConfig: auto-derivation, validation guards, removed knobs, serialization."""
import tempfile

import pytest
from conftest import make_config

from src.configuration_bibo import BiBoConfig


def test_auto_derived_hyperparameters():
    c = make_config(num_hidden_layers=6, intermediate_size=96, num_experts_per_tok=3,
                    mlp_only_layers=None, rope_theta=None, moe_intermediate_size=None)
    assert c.rope_theta == 1e7
    assert c.moe_intermediate_size == 96 // 3, "moe_intermediate = intermediate // top_k (FLOP parity)"
    assert c.mlp_only_layers == [0, 5], "first + last layer dense"
    assert c.head_dim == 64 // 4
    assert c.rope_dim == 4 and c.rope_dim % 2 == 0, "round(0.334*16)=5 -> forced even -> 4"
    assert c.rope_scaling["type"] == "dynamic" and c.rope_scaling["factor"] == 1.0


def test_mlp_only_layers_dedupes_single_layer_model():
    c = make_config(num_hidden_layers=1, mlp_only_layers=None)
    assert c.mlp_only_layers == [0], "N==1 must give [0], not [0, 0]"


def test_bias_update_factor_is_a_fixed_small_step():
    """It must NOT scale with num_routed_experts. sign() never returns 0, so the bias dithers +-u
    forever and u is the balancer's steady-state routing-noise floor; it has to stay well under the
    top-k boundary gap, which SHRINKS as experts are added (0.041 at n=8 -> 0.0064 at n=128)."""
    assert make_config().bias_update_factor == 0.001
    for mult in (1, 2, 10, 42):     # 5 .. 128 routed experts
        c = make_config(polyglu_expert_multiplier=mult, special_expert_pairs=1,
                        num_experts_per_tok=1)
        assert c.bias_update_factor == 0.001, \
            f"u must not depend on n (got {c.bias_update_factor} at n={c.num_routed_experts})"


def test_explicit_bias_update_factor_wins():
    assert make_config(bias_update_factor=3e-2).bias_update_factor == 3e-2
    assert make_config(bias_update_factor=None).bias_update_factor == 0.001, "None -> default"
    assert make_config(bias_update_factor=0.0).bias_update_factor == 0.0, "0 disables balancing"


def test_negative_bias_update_factor_is_rejected():
    with pytest.raises(ValueError):
        make_config(bias_update_factor=-1.0)


@pytest.mark.parametrize("overrides,reason", [
    (dict(gate_type="sigmiod"), "typo'd gate_type must not fall through to softmax"),
    (dict(gate_type="situ", norm_topk_prob=False), "signed gate needs softmax normalization"),
    (dict(router_activation="gelu"), "unknown router_activation"),
    (dict(num_experts_per_tok=99), "top_k > num_routed_experts"),
    (dict(num_key_value_heads=0), "kv_heads must be > 0"),
    (dict(bias_update_threshold=0), "threshold must be > 0"),
    (dict(hidden_size=63), "hidden_size % num_attention_heads != 0"),
    (dict(layer_norm_type="layernorm"), "only rms is supported"),
    (dict(shared_expert_type="lstm"), "unknown shared_expert_type"),
    (dict(use_ssmax=True, add_full_attention_sink_bias=True), "G1 (global sink + SSMax) is guarded"),
    (dict(hybrid_layer_pattern=[1, 0, 0, 0], sliding_window=0), "SWA needs a positive window"),
])
def test_validation_guards(overrides, reason):
    with pytest.raises(ValueError):
        make_config(**overrides)


@pytest.mark.parametrize("knob,value", [("router_noise", 0.5),
                                        ("zero_expert", True), ("identity_expert", False)])
def test_removed_knobs_are_dropped_not_stored(knob, value):
    """PretrainedConfig setattr()s unknown kwargs, so a stale knob would reappear as an attribute
    AND be re-serialized into config.json as if the feature still existed."""
    c = make_config(**{knob: value})
    assert not hasattr(c, knob), f"{knob} leaked onto the config"
    assert knob not in c.to_dict(), f"{knob} would be written back to config.json"


def test_config_round_trip_preserves_derived_fields():
    c = make_config(gate_type="situ", hybrid_layer_pattern=[0, 1, 1, 0], use_ssmax=False)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        c2 = BiBoConfig.from_pretrained(d)
    for k in ("head_dim", "rope_dim", "num_routed_experts", "gate_type", "norm_topk_prob",
              "moe_intermediate_size", "bias_update_factor", "layer_types", "sliding_window"):
        assert getattr(c, k, None) == getattr(c2, k, None), f"{k} did not survive save/load"


def test_derived_dims_ignore_stale_serialized_values():
    """head_dim/rope_dim are computed AFTER super().__init__, so a hand-edited config.json cannot
    override the current derivation."""
    c = make_config(head_dim=999, rope_dim=999)
    assert c.head_dim == 16 and c.rope_dim == 4


def test_sliding_window_serializes_as_none_without_swa_layers():
    assert make_config().sliding_window is None, "HF machinery keys off sliding_window"
    c = make_config(hybrid_layer_pattern=[0, 1, 1, 0])
    assert c.sliding_window == 128
    assert c.layer_types == ["full_attention", "sliding_attention",
                            "sliding_attention", "full_attention"]


def test_num_routed_experts_derivation():
    for mult, pairs in ((2, 1), (3, 0), (1, 2), (10, 1)):
        c = make_config(polyglu_expert_multiplier=mult, special_expert_pairs=pairs,
                        num_experts_per_tok=1)
        assert c.num_routed_experts == mult * 3 + pairs * 2
