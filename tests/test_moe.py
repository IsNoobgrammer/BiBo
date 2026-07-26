"""MoE: expert layout, PolyGLU activations, router gating, load balancing."""
import pytest
import torch
from conftest import DEVICE, make_config, make_model

from src.modeling.ffn.moe import _POLYGLU_ACTIVATIONS, BiBoMoELayer
from src.modeling.ffn.router import BiBoMoERouter

GATES = ["sigmoid", "situ", "softmax"]


# ── expert layout ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mult,pairs", [(2, 1), (3, 0), (1, 2)])
def test_expert_layout_and_fused_weight_shapes(mult, pairs):
    c = make_config(polyglu_expert_multiplier=mult, special_expert_pairs=pairs,
                    num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    assert e.num_polyglu_experts == mult * 3
    assert e.identity_start == mult * 3, "GLU block comes first"
    assert e.zero_end == c.num_routed_experts, "Zero block closes the layout"
    assert e.gate_up_proj.shape == (mult * 3, 2 * c.moe_intermediate_size, c.hidden_size)
    assert e.down_proj.shape == (mult * 3, c.hidden_size, c.moe_intermediate_size)


def test_polyglu_activation_cycle_is_e_mod_3():
    assert _POLYGLU_ACTIVATIONS == ("silu", "relu2", "normsilu")
    n = BiBoMoELayer(make_config(polyglu_expert_multiplier=3)).to(DEVICE).experts.num_polyglu_experts
    assert [_POLYGLU_ACTIVATIONS[e % 3] for e in range(n)] == ["silu", "relu2", "normsilu"] * 3


def test_identity_and_zero_expert_semantics():
    c = make_config(polyglu_expert_multiplier=1, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    w = torch.ones(4, 1, device=DEVICE)
    idx = lambda i: torch.full((4, 1), i, device=DEVICE, dtype=torch.long)
    assert torch.allclose(e(x, idx(e.identity_start), w), x, atol=1e-6), "Identity is not passthrough"
    assert e(x, idx(e.zero_start), w).abs().max() == 0, "Zero emitted nonzero output"


def test_identity_weight_is_applied():
    c = make_config(polyglu_expert_multiplier=1, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    half = torch.full((4, 1), 0.5, device=DEVICE)
    out = e(x, torch.full((4, 1), e.identity_start, device=DEVICE, dtype=torch.long), half)
    assert torch.allclose(out, x * 0.5, atol=1e-6)


@pytest.mark.parametrize("identity,zero", [(False, True), (True, False), (False, False)])
def test_special_expert_toggles_make_zero_width_blocks(identity, zero):
    c = make_config(identity_expert=identity, zero_expert=zero, special_expert_pairs=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    assert (e.identity_end > e.identity_start) == identity
    assert (e.zero_end > e.zero_start) == zero


# ── router ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("gate", GATES)
def test_router_weights_are_fp32_positive_and_sum_to_one(gate):
    r = BiBoMoERouter(make_config(gate_type=gate)).to(DEVICE)
    idx, w = r(torch.randn(2, 5, 64, device=DEVICE) * 3)
    assert w.dtype == torch.float32, "the router->combine path stays fp32 end to end"
    assert (w > 0).all(), "softmax normalization must yield positive weights"
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
    assert idx.min() >= 0 and idx.max() < r.num_routed_experts


def test_normalization_is_over_topk_only_not_all_experts():
    """With 32 experts and k=2 the softmax must see 2 gathered scores, not 32."""
    c = make_config(polyglu_expert_multiplier=10, special_expert_pairs=1, num_experts_per_tok=2)
    assert c.num_routed_experts == 32
    _, w = BiBoMoERouter(c).to(DEVICE)(torch.randn(1, 4, 64, device=DEVICE) * 3)
    assert w.shape[-1] == 2
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6), \
        "top-k weights must carry the full unit mass into the residual"


def test_norm_topk_prob_false_leaves_raw_scores():
    r = BiBoMoERouter(make_config(norm_topk_prob=False)).to(DEVICE)
    _, w = r(torch.randn(2, 5, 64, device=DEVICE) * 3)
    assert not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-3)


def test_situ_gate_is_signed_before_normalization():
    """sigmoid(x)*tanh(x) dips negative, which is exactly why div-by-sum was replaced."""
    x = torch.tensor([-5.0, -0.7799, 0.0, 2.0], device=DEVICE)
    situ = torch.sigmoid(x) * torch.tanh(x)
    assert situ[1] < 0, "SiTU must go negative"
    assert situ[0] > situ[1], "SiTU is NON-monotonic in the logit: f(-5) > f(-0.78)"


def test_bias_affects_selection_only():
    r = BiBoMoERouter(make_config()).to(DEVICE)
    x = torch.randn(1, 6, 64, device=DEVICE)
    assert r.bias.requires_grad is False, "bias is heuristic, never optimizer-managed"
    with torch.no_grad():
        r.bias[0] = 5.0
    idx, w = r(x)
    assert (idx == 0).any(), "a large bias must change selection"
    assert w.max() <= 1.0 + 1e-6, "bias must never leak into the combine weights"


@pytest.mark.parametrize("gate", GATES)
def test_router_gradients_reach_the_projection(gate):
    r = BiBoMoERouter(make_config(gate_type=gate)).to(DEVICE)
    x = (torch.randn(2, 4, 64, device=DEVICE) * 3).requires_grad_(True)
    _, w = r(x)
    (w * torch.randn_like(w)).sum().backward()   # a plain sum() is constant under softmax
    for g in (r.gate_proj.weight.grad, x.grad):
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0


def test_router_projection_has_experts_as_the_row_dimension():
    c = make_config()
    r = BiBoMoERouter(c).to(DEVICE)
    assert r.gate_proj.weight.shape == (c.num_routed_experts, c.hidden_size)
    assert not hasattr(r, "gate_conv"), "the conv router was removed"


# ── load balancing ───────────────────────────────────────────────────────────
def test_bias_update_moves_toward_balance():
    c = make_config(load_balance_strategy="bias", bias_update_factor=0.1)
    layer = BiBoMoELayer(c).to(DEVICE)
    tpe = torch.zeros(c.num_routed_experts, device=DEVICE)
    tpe[0] = 100.0
    before = layer.gate.bias.clone()
    layer.update_bias(tpe)
    delta = layer.gate.bias - before
    assert delta[0] < 0, "the overloaded expert must be discouraged"
    assert (delta[1:] > 0).all(), "starved experts must be encouraged"


def test_balance_exclude_specials_freezes_special_biases():
    c = make_config(load_balance_strategy="bias", balance_exclude_specials=True,
                    bias_update_factor=0.1)
    layer = BiBoMoELayer(c).to(DEVICE)
    npg = layer.experts.num_polyglu_experts
    layer.update_bias(torch.arange(float(c.num_routed_experts), device=DEVICE) * 10)
    assert (layer.gate.bias[npg:] == 0).all(), "Identity/Zero biases must stay at 0"
    assert (layer.gate.bias[:npg] != 0).any(), "the GLU block should still be balanced"


def test_bias_update_fires_on_the_expected_step_interval():
    """Triggering on forward STEPS (not device tokens) keeps every DDP rank in lockstep."""
    tokens_per_fwd = 2 * 8
    c = make_config(load_balance_strategy="bias", bias_update_threshold=tokens_per_fwd * 3)
    layer = BiBoMoELayer(c).to(DEVICE).train()
    idx = torch.zeros(2, 8, 2, dtype=torch.long, device=DEVICE)
    fired = [layer._balance_step(idx, tokens_per_fwd) is not None for _ in range(7)]
    assert layer._update_every == 3
    assert fired == [False, False, True, False, False, True, False]


def test_no_balancing_when_strategy_is_none():
    c = make_config(load_balance_strategy="none")
    layer = BiBoMoELayer(c).to(DEVICE).train()
    x = torch.randn(2, 8, 64, device=DEVICE)
    layer(x)
    assert layer.gate.bias.abs().max() == 0, "strategy='none' must never touch the bias"


def test_router_stays_fp32_in_a_half_precision_model():
    layer = BiBoMoELayer(make_config()).to(DEVICE)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    layer = layer.to(dt)
    x = torch.randn(2, 5, 64, device=DEVICE, dtype=dt)
    _, w = layer.gate(x)
    assert w.dtype == torch.float32
    assert layer(x).dtype == dt, "the layer output must return to the model dtype"


@pytest.mark.parametrize("gate", GATES)
def test_moe_layer_forward_backward(gate):
    layer = BiBoMoELayer(make_config(gate_type=gate)).to(DEVICE)
    x = torch.randn(2, 6, 64, device=DEVICE, requires_grad=True)
    layer(x).sum().backward()
    assert torch.isfinite(layer.experts.gate_up_proj.grad).all()
    assert layer.experts.gate_up_proj.grad.abs().max() > 0


def test_shared_expert_is_off_by_default_and_adds_directly():
    assert make_config().use_shared_expert is False, "off by default to param-match Qwen"
    layer = BiBoMoELayer(make_config(use_shared_expert=True)).to(DEVICE)
    assert len(layer.shared_experts_list) == 1
    assert not hasattr(layer, "moe_shared_scaling"), "the scaling scalar was removed"
