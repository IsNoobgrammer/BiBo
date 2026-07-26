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
    assert e.pos_start == mult * 3, "GLU block comes first"
    assert e.neg_end == c.num_routed_experts, "-Identity block closes the layout"
    assert e.gate_up_proj.shape == (mult * 3, 2 * c.moe_intermediate_size, c.hidden_size)
    assert e.down_proj.shape == (mult * 3, c.hidden_size, c.moe_intermediate_size)


def test_polyglu_activation_cycle_is_e_mod_3():
    assert _POLYGLU_ACTIVATIONS == ("silu", "relu2", "normsilu")
    n = BiBoMoELayer(make_config(polyglu_expert_multiplier=3)).to(DEVICE).experts.num_polyglu_experts
    assert [_POLYGLU_ACTIVATIONS[e % 3] for e in range(n)] == ["silu", "relu2", "normsilu"] * 3


def test_signed_identity_expert_semantics():
    """+Identity emits +w*x, -Identity emits -w*x. No expert may emit 0 (that was the Zero expert,
    removed Jul 26 2026 — its output carries no gradient signal to the router)."""
    c = make_config(polyglu_expert_multiplier=1, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    w = torch.ones(4, 1, device=DEVICE)
    idx = lambda i: torch.full((4, 1), i, device=DEVICE, dtype=torch.long)
    assert torch.allclose(e(x, idx(e.pos_start), w), x, atol=1e-6), "+Identity is not passthrough"
    assert torch.allclose(e(x, idx(e.neg_start), w), -x, atol=1e-6), "-Identity is not negated passthrough"


def test_signed_identity_weight_is_applied():
    c = make_config(polyglu_expert_multiplier=1, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    half = torch.full((4, 1), 0.5, device=DEVICE)
    for start, sign in ((e.pos_start, 1.0), (e.neg_start, -1.0)):
        out = e(x, torch.full((4, 1), start, device=DEVICE, dtype=torch.long), half)
        assert torch.allclose(out, sign * x * 0.5, atol=1e-6)


def test_equal_weight_signed_pair_cancels_to_zero():
    """The ± pair spans the old Zero expert's behavior as the w+ == w- special case — but reachable
    by routing, with live gradients on both branches."""
    c = make_config(polyglu_expert_multiplier=1, special_expert_pairs=1, num_experts_per_tok=2)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    idx = torch.tensor([[e.pos_start, e.neg_start]] * 4, device=DEVICE)
    out = e(x, idx, torch.full((4, 2), 0.5, device=DEVICE))
    assert out.abs().max() < 1e-6, "equal-weight +x and -x must cancel"


def test_both_signed_specials_carry_a_combine_weight_gradient():
    """The reason Zero was removed: its output is 0, so d(loss)/d(its weight) = <grad_out, 0> is
    identically 0 and the router can never learn that picking it was right. Both ±Identity branches
    must carry real gradient."""
    c = make_config(polyglu_expert_multiplier=1, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(8, c.hidden_size, device=DEVICE)
    for start in (e.pos_start, e.neg_start):
        w = torch.full((8, 1), 0.5, device=DEVICE, requires_grad=True)
        idx = torch.full((8, 1), start, device=DEVICE, dtype=torch.long)
        # <out, random>, NOT out.sum(): a symmetric reduction can cancel to a vacuous zero gradient
        (e(x, idx, w) * torch.randn(8, c.hidden_size, device=DEVICE)).sum().backward()
        assert w.grad.abs().max() > 0, f"expert {start} has no gradient path to its router weight"


@pytest.mark.parametrize("pos,neg", [(False, True), (True, False), (False, False)])
def test_special_expert_toggles_make_zero_width_blocks(pos, neg):
    c = make_config(pos_identity_expert=pos, neg_identity_expert=neg, special_expert_pairs=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    assert (e.pos_end > e.pos_start) == pos
    assert (e.neg_end > e.neg_start) == neg


# ── conv router ──────────────────────────────────────────────────────────────
def test_conv_router_weight_is_2d_with_experts_as_rows():
    """REGRESSION GUARD for the Muon orthogonalization axis. NS batches over the LEADING dim and
    iterates on the smaller Gram, so an (E,H,K) nn.Conv1d weight gets its KERNEL TAPS decorrelated
    and its EXPERTS left correlated (it cannot even de-collapse identical experts). Storing the
    weight as 2D (E, H*K) makes the Gram (E,E) -> experts decorrelated, same as the MLP router.
    If this test fails, routing quality silently regresses. See ablate/common/optim.py."""
    K = 5
    c = make_config(router_type="conv", kernel_size=K)
    r = BiBoMoERouter(c).to(DEVICE)
    assert r.gate_conv.ndim == 2, "conv router weight must be 2D or Muon whitens taps, not experts"
    assert r.gate_conv.shape == (c.num_routed_experts, c.hidden_size * K)
    assert not hasattr(r, "gate_proj"), "conv router must not also build the MLP projection"


def test_conv_router_is_causal():
    """Position t must depend only on t-K+1..t. Perturbing a LATER token cannot move earlier logits."""
    c = make_config(router_type="conv", kernel_size=3)
    r = BiBoMoERouter(c).to(DEVICE)
    x = torch.randn(1, 12, c.hidden_size, device=DEVICE)
    x2 = x.clone()
    x2[0, 8] += 10.0
    a, b = r.router_logits(x), r.router_logits(x2)
    assert torch.allclose(a[:8], b[:8], atol=1e-5), "conv router leaked future information"
    assert not torch.allclose(a[8], b[8], atol=1e-3), "token 8 should have changed"


def test_conv_router_init_is_fan_in_aware():
    """A conv logit sums H*K terms vs the MLP's H. Sharing initializer_range would start the conv
    router sqrt(K)x sharper (measured 1.64x at K=3) -- an uncontrolled confound in any conv-vs-mlp
    ablation. Dividing the std by sqrt(K) matches the two logit scales."""
    K = 4
    h = torch.randn(2, 64, 64, device=DEVICE)
    lg_mlp = BiBoMoERouter(make_config(router_type="mlp")).to(DEVICE).router_logits(h)
    lg_conv = BiBoMoERouter(make_config(router_type="conv", kernel_size=K)).to(DEVICE).router_logits(h)
    ratio = (lg_conv.std() / lg_mlp.std()).item()
    assert 0.75 < ratio < 1.35, f"conv/mlp logit-scale ratio {ratio:.3f} (fan-in init broken?)"


@pytest.mark.parametrize("gate", GATES)
def test_conv_router_routes_and_carries_gradient(gate):
    c = make_config(router_type="conv", kernel_size=3, gate_type=gate)
    r = BiBoMoERouter(c).to(DEVICE)
    idx, w = r(torch.randn(2, 6, c.hidden_size, device=DEVICE))
    assert idx.shape == (2, 6, c.num_experts_per_tok) and w.dtype == torch.float32
    assert idx.min() >= 0 and idx.max() < c.num_routed_experts
    # <w, random>, not w.sum(): normalized weights sum to a constant -> vacuous zero gradient
    (w * torch.randn_like(w)).sum().backward()
    assert r.gate_conv.grad is not None and r.gate_conv.grad.abs().max() > 0


def test_router_type_is_validated():
    with pytest.raises(ValueError):
        make_config(router_type="attention")


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
    assert (layer.gate.bias[npg:] == 0).all(), "±Identity biases must stay at 0"
    assert (layer.gate.bias[:npg] != 0).any(), "the GLU block should still be balanced"


def test_glu_token_budget_shifts_the_whole_glu_block_when_off_budget():
    """LongCat's absolute target (K_e/K per block) vs DeepSeek's mean-relative one. When the GLU block
    is collectively over budget EVERY GLU bias must drop -- a uniform shift that cannot reorder the
    block but does push traffic across to the specials. The mean-relative rule cannot express this:
    its deviations always sum to ~0 inside the block."""
    c = make_config(load_balance_strategy="bias", bias_update_factor=0.1, glu_token_budget=0.75,
                    polyglu_expert_multiplier=2, special_expert_pairs=1)
    npg = BiBoMoELayer(c).experts.num_polyglu_experts
    n_special = c.num_routed_experts - npg
    for glu_share, want in ((0.95, -1.0), (0.40, 1.0)):     # over budget -> down, under -> up
        layer = BiBoMoELayer(c).to(DEVICE)
        tpe = torch.cat([torch.full((npg,), glu_share / npg, device=DEVICE),
                         torch.full((n_special,), (1 - glu_share) / n_special, device=DEVICE)]) * 1000
        layer.update_bias(tpe)
        b = layer.gate.bias
        assert torch.allclose(b[:npg], torch.full_like(b[:npg], want * 0.1)), \
            f"GLU share {glu_share} vs budget 0.75: expected a uniform {want:+.0f} shift, got {b[:npg]}"
        assert (b[npg:] == 0).all(), "specials are the residual sink -- Delta b must be 0 (LongCat)"


def test_glu_token_budget_still_equalizes_within_the_glu_block():
    """The budget sets the block's TOTAL share; experts inside it must still be balanced against
    each other, or the knob would trade balance for budget."""
    c = make_config(load_balance_strategy="bias", bias_update_factor=0.1, glu_token_budget=0.75,
                    polyglu_expert_multiplier=2, special_expert_pairs=1)
    layer = BiBoMoELayer(c).to(DEVICE)
    npg = layer.experts.num_polyglu_experts
    tpe = torch.full((c.num_routed_experts,), 10.0, device=DEVICE)
    tpe[0] = 1000.0                                          # one hogging GLU expert
    layer.update_bias(tpe)
    b = layer.gate.bias
    assert b[0] < 0 and (b[1:npg] > 0).all(), "the hog must be discouraged, its peers encouraged"


def test_glu_token_budget_is_validated():
    with pytest.raises(ValueError):
        make_config(glu_token_budget=1.5)
    with pytest.raises(ValueError):
        make_config(glu_token_budget=0.0)
    with pytest.raises(ValueError):                          # no specials -> nowhere for traffic to go
        make_config(glu_token_budget=0.75, special_expert_pairs=0)
    make_config(glu_token_budget=1.0, special_expert_pairs=0)   # 1.0 == "all GLU", always legal


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


def test_prop_mode_deviations_sum_to_zero_so_no_common_mode_drift():
    """The reason LongCat drops sign(). Raw deviations sum to EXACTLY 0 over the balanced block, so
    the block's MEAN bias cannot move -- only the spread within it. sign()ed deviations do not: a
    right-skewed load puts most experts below the mean, most get +1, and the whole block floats up.
    Measured on a real run (xsp, u=0.01): GLU biases at +1.28 while frozen specials sat at 0."""
    skewed = torch.tensor([500.0, 300.0] + [10.0] * 16 + [50.0] * 4, device=DEVICE)  # 22 experts
    c = make_config(load_balance_strategy="bias", bias_update_factor=0.1,
                    polyglu_expert_multiplier=6, special_expert_pairs=2, bias_update_mode="prop")
    layer = BiBoMoELayer(c).to(DEVICE)
    layer.update_bias(skewed)
    assert abs(float(layer.gate.bias.mean())) < 1e-6, "prop mode must not shift the block mean"
    assert float(layer.gate.bias.std()) > 0, "...but it must still spread the biases"

    sign_layer = BiBoMoELayer(make_config(
        load_balance_strategy="bias", bias_update_factor=0.1, polyglu_expert_multiplier=6,
        special_expert_pairs=2, bias_update_mode="sign")).to(DEVICE)
    sign_layer.update_bias(skewed)
    assert float(sign_layer.gate.bias.mean()) > 0.05, \
        "sign mode SHOULD drift the mean up on a right-skewed load (that is the defect)"


def test_prop_mode_step_shrinks_as_load_approaches_balance():
    """Proportional control has a fixed point; bang-bang does not. This is why u=0.01 in sign mode
    held its load target yet lost 0.094 loss: the bias never stops dithering by +-u."""
    c = make_config(load_balance_strategy="bias", bias_update_factor=1.0, bias_update_mode="prop")
    E = c.num_routed_experts
    steps = []
    for skew in (10.0, 1.0, 0.0):                       # progressively closer to perfectly balanced
        layer = BiBoMoELayer(c).to(DEVICE)
        tpe = torch.full((E,), 100.0, device=DEVICE)
        tpe[0] += skew * 10
        layer.update_bias(tpe)
        steps.append(float(layer.gate.bias.abs().max()))
    assert steps[0] > steps[1] > steps[2], f"prop steps must shrink with the deviation, got {steps}"
    assert steps[2] < 1e-6, "a perfectly balanced load must produce NO update in prop mode"

    # torch.sign(0) IS 0, so an exactly-balanced load produces no sign-mode step either. That is not
    # the real regime: a stochastic load is never exactly at its mean, and sign() maps ANY
    # infinitesimal deviation to a FULL +-u step. That is the dither, and prop mode does not have it.
    sc = make_config(load_balance_strategy="bias", bias_update_factor=0.1, bias_update_mode="sign")
    tpe = torch.full((sc.num_routed_experts,), 100.0, device=DEVICE)
    tpe[0] += 1e-3                                        # essentially, but not exactly, balanced
    sign_layer = BiBoMoELayer(sc).to(DEVICE)
    sign_layer.update_bias(tpe)
    assert float(sign_layer.gate.bias.abs().max()) == pytest.approx(0.1), \
        "sign mode takes a FULL u step on an infinitesimal deviation -- the dither floor"
    prop_layer = BiBoMoELayer(make_config(load_balance_strategy="bias", bias_update_factor=0.1,
                                          bias_update_mode="prop")).to(DEVICE)
    prop_layer.update_bias(tpe)
    assert float(prop_layer.gate.bias.abs().max()) < 1e-6, \
        "prop mode's step must vanish with the deviation"


def test_bias_update_mode_is_validated():
    with pytest.raises(ValueError):
        make_config(bias_update_mode="pid")
