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


def test_bias_update_factor_is_a_fixed_step_not_a_function_of_n():
    """u must NOT scale with num_routed_experts. It is a gain on the deviation (which is already in
    share units), never a function of the expert count — the old auto-Hill grew u with n, which is
    backwards: the k|k+1 selection-boundary gap SHRINKS as experts are added."""
    assert make_config().bias_update_factor == 0.4
    for n in (4, 8, 22, 128):
        c = make_config(num_routed_experts=n, special_expert_pairs=1, num_experts_per_tok=1)
        assert c.bias_update_factor == 0.4, \
            f"u must not depend on n (got {c.bias_update_factor} at n={c.num_routed_experts})"


def test_explicit_bias_update_factor_wins():
    assert make_config(bias_update_factor=3e-2).bias_update_factor == 3e-2
    assert make_config(bias_update_factor=None).bias_update_factor == 0.4
    assert make_config(bias_update_factor=0.0).bias_update_factor == 0.0, "0 disables balancing"


@pytest.mark.parametrize("overrides,reason", [
    (dict(norm_topk_prob="divsum"), "typo'd norm_topk_prob must not silently fall through"),
    (dict(norm_topk_prob="none"), "'none' is not off — pass False"),
    (dict(num_experts_per_tok=99), "top_k > num_routed_experts"),
    (dict(num_experts_per_tok=0), "top_k must be >= 1"),
    (dict(num_routed_experts=2, special_expert_pairs=1), "specials would leave 0 GLU experts"),
    (dict(hidden_size=63), "hidden_size % num_attention_heads != 0"),
    (dict(partial_rotary_factor=0.05), "rope_dim < 2"),
    (dict(hybrid_layer_pattern=[1, 0, 0, 0], sliding_window=0), "SWA needs a positive window"),
    (dict(hybrid_layer_pattern=[1, 0]), "pattern length != num_hidden_layers"),
])
def test_validation_guards(overrides, reason):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_global_attention_sink_is_no_longer_gated():
    """The G1 guard existed only because a global sink had to be scaled by SSMax's C=s*log(n).
    SSMax was removed Aug 2 2026, so a sink on global layers is now just a sink."""
    c = make_config(add_full_attention_sink_bias=True)
    assert c.add_full_attention_sink_bias is True


def test_legacy_bool_norm_topk_prob_maps_to_sum():
    """Pre-debloat configs serialize `norm_topk_prob: true`; True meant "normalize", now "sum"."""
    assert make_config(norm_topk_prob=True).norm_topk_prob == "sum"
    assert make_config().norm_topk_prob == "sum", "sum is the default"
    assert make_config(norm_topk_prob=False).norm_topk_prob is False, "False = raw scores, still legal"


def test_config_round_trip_preserves_derived_fields():
    c = make_config(norm_topk_prob="softmax", hybrid_layer_pattern=[0, 1, 1, 0])
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        c2 = BiBoConfig.from_pretrained(d)
    for k in ("head_dim", "rope_dim", "num_routed_experts", "norm_topk_prob",
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


def test_glu_expert_count_is_derived_from_num_routed_experts():
    """num_routed_experts is THE knob (polyglu_expert_multiplier x POLYGLU_GROUP was deleted Aug 1
    2026). The GLU block is whatever the ±Identity specials leave behind."""
    for n, pairs in ((8, 1), (6, 0), (8, 2), (128, 1)):
        c = make_config(num_routed_experts=n, special_expert_pairs=pairs, num_experts_per_tok=1)
        assert c.num_routed_experts == n
        assert c.num_glu_experts == n - pairs * 2
    c = make_config(num_routed_experts=8, special_expert_pairs=1, neg_identity_expert=False)
    assert c.num_glu_experts == 7, "a disabled sign is a zero-width block, not a reserved slot"
