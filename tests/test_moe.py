"""MoE: expert layout, radial NormSiLU, router gating, load balancing."""
import pytest
import torch
import torch.nn.functional as F
from conftest import DEVICE, make_config, make_model

from src.modeling.ffn.moe import BiBoMoELayer
from src.modeling.ffn.router import BiBoMoERouter

NORMS = ["sum", "softmax"]


# ── expert layout ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("routed,pairs", [(8, 1), (6, 0), (8, 2)])
def test_expert_layout_and_fused_weight_shapes(routed, pairs):
    c = make_config(num_routed_experts=routed, special_expert_pairs=pairs, num_experts_per_tok=1)
    n_glu = routed - pairs * 2
    e = BiBoMoELayer(c).to(DEVICE).experts
    assert e.num_glu_experts == n_glu
    assert e.pos_start == n_glu, "GLU block comes first"
    assert e.neg_end == c.num_routed_experts, "-Identity block closes the layout"
    assert e.gate_up_proj.shape == (n_glu, 2 * c.moe_intermediate_size, c.hidden_size)
    assert e.down_proj.shape == (n_glu, c.hidden_size, c.moe_intermediate_size)
    assert e.radial_theta.shape == (n_glu,), "one exponent per GLU expert, none for the specials"


# ── radial NormSiLU: THE expert activation ───────────────────────────────────
def _one_expert(e, x, i):
    """Route every row of x to expert i alone, weight 1."""
    n = x.shape[0]
    return e(x, torch.full((n, 1), i, device=DEVICE, dtype=torch.long),
             torch.ones(n, 1, device=DEVICE))


def test_radial_normsilu_matches_a_fp64_reference():
    """radial(g) = r^p * SiLU(g/r), r = rms(g), p = sigmoid(theta_e) in (0,1). The silu/relu2/normsilu
    menu was deleted Aug 1 2026 — there is one activation and this is its definition."""
    c = make_config(num_routed_experts=4, special_expert_pairs=0, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    with torch.no_grad():                                   # p = .12 / .50 / .82 / .95
        e.radial_theta.copy_(torch.tensor([-2.0, 0.0, 1.5, 3.0], device=DEVICE))
    x = torch.randn(5, c.hidden_size, device=DEVICE)
    for i in range(e.num_glu_experts):
        gate, up = F.linear(x, e.gate_up_proj[i]).chunk(2, dim=-1)
        g = gate.double()
        r = (g.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        p = torch.sigmoid(e.radial_theta[i].double())
        ref = F.linear(F.silu(g / r) * r.pow(p) * up.double(), e.down_proj[i].double())
        rel = ((_one_expert(e, x, i).double() - ref).abs().max() / ref.abs().max()).item()
        assert rel < 1e-6, f"expert {i} (p={float(p):.2f}) deviates from the reference: {rel:.2e}"


def test_radial_theta_is_a_real_learnable_parameter():
    """REGRESSION: theta used to be injected by ablate as `situ_alpha`, so the eager path RAISED
    without the harness — and any patch-free forward would silently sit at radial's p->0 floor."""
    c = make_config(special_expert_pairs=0, num_experts_per_tok=2)
    layer = BiBoMoELayer(c).to(DEVICE)
    th = layer.experts.radial_theta
    assert th.requires_grad, "theta must train like any weight"
    assert float(th.abs().max()) == 0.0, "init theta=0 -> p=sigmoid(0)=0.5, mid-range"
    x = torch.randn(2, 6, c.hidden_size, device=DEVICE)
    (layer(x) * torch.randn(2, 6, c.hidden_size, device=DEVICE)).sum().backward()
    assert th.grad is not None and th.grad.abs().max() > 0, "theta gets no gradient — radial is inert"


def test_radial_at_p_zero_is_exactly_normsilu():
    """p -> 0 is the floor of the family: r^0 = 1, i.e. plain SiLU(g/rms(g)). Guards that the gain is
    applied as r^sigmoid(theta) and not, say, r^theta."""
    c = make_config(num_routed_experts=4, special_expert_pairs=0, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    with torch.no_grad():
        e.radial_theta.fill_(-40.0)                          # sigmoid(-40) = 0
    x = torch.randn(5, c.hidden_size, device=DEVICE)
    gate, up = F.linear(x, e.gate_up_proj[0]).chunk(2, dim=-1)
    g = gate.float()
    normsilu = F.silu(g * torch.rsqrt(g.square().mean(-1, keepdim=True) + 1e-6))
    ref = F.linear(normsilu * up, e.down_proj[0])
    assert (_one_expert(e, x, 0) - ref).abs().max() < 1e-5


def test_signed_identity_expert_semantics():
    """+Identity emits +w*x, -Identity emits -w*x. No expert may emit 0 (that was the Zero expert,
    removed Jul 26 2026 — its output carries no gradient signal to the router)."""
    c = make_config(num_routed_experts=4, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    w = torch.ones(4, 1, device=DEVICE)
    idx = lambda i: torch.full((4, 1), i, device=DEVICE, dtype=torch.long)
    assert torch.allclose(e(x, idx(e.pos_start), w), x, atol=1e-6), "+Identity is not passthrough"
    assert torch.allclose(e(x, idx(e.neg_start), w), -x, atol=1e-6), "-Identity is not negated passthrough"


def test_signed_identity_weight_is_applied():
    c = make_config(num_routed_experts=4, special_expert_pairs=1, num_experts_per_tok=1)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    half = torch.full((4, 1), 0.5, device=DEVICE)
    for start, sign in ((e.pos_start, 1.0), (e.neg_start, -1.0)):
        out = e(x, torch.full((4, 1), start, device=DEVICE, dtype=torch.long), half)
        assert torch.allclose(out, sign * x * 0.5, atol=1e-6)


def test_equal_weight_signed_pair_cancels_to_zero():
    """The ± pair spans the old Zero expert's behavior as the w+ == w- special case — but reachable
    by routing, with live gradients on both branches."""
    c = make_config(num_routed_experts=4, special_expert_pairs=1, num_experts_per_tok=2)
    e = BiBoMoELayer(c).to(DEVICE).experts
    x = torch.randn(4, c.hidden_size, device=DEVICE)
    idx = torch.tensor([[e.pos_start, e.neg_start]] * 4, device=DEVICE)
    out = e(x, idx, torch.full((4, 2), 0.5, device=DEVICE))
    assert out.abs().max() < 1e-6, "equal-weight +x and -x must cancel"


def test_both_signed_specials_carry_a_combine_weight_gradient():
    """The reason Zero was removed: its output is 0, so d(loss)/d(its weight) = <grad_out, 0> is
    identically 0 and the router can never learn that picking it was right. Both ±Identity branches
    must carry real gradient."""
    c = make_config(num_routed_experts=4, special_expert_pairs=1, num_experts_per_tok=1)
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


# ── router ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("norm", NORMS)
def test_router_weights_are_fp32_positive_and_sum_to_one(norm):
    r = BiBoMoERouter(make_config(norm_topk_prob=norm)).to(DEVICE)
    idx, w = r(torch.randn(2, 5, 64, device=DEVICE) * 3)
    assert w.dtype == torch.float32, "the router->combine path stays fp32 end to end"
    assert (w > 0).all(), "softmax normalization must yield positive weights"
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)
    assert idx.min() >= 0 and idx.max() < r.num_routed_experts


def test_normalization_is_over_topk_only_not_all_experts():
    """With 22 experts and k=2 the normalization must see 2 gathered scores, not 22."""
    c = make_config(num_routed_experts=22, special_expert_pairs=1, num_experts_per_tok=2)
    assert c.num_glu_experts == 20
    _, w = BiBoMoERouter(c).to(DEVICE)(torch.randn(1, 4, 64, device=DEVICE) * 3)
    assert w.shape[-1] == 2
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6), \
        "top-k weights must carry the full unit mass into the residual"


def test_norm_topk_prob_false_leaves_raw_scores():
    r = BiBoMoERouter(make_config(norm_topk_prob=False)).to(DEVICE)
    _, w = r(torch.randn(2, 5, 64, device=DEVICE) * 3)
    assert not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-3)


def test_gate_is_sigmoid_only():
    c = make_config()
    r = BiBoMoERouter(c).to(DEVICE)
    x = torch.randn(2, 5, c.hidden_size, device=DEVICE) * 3
    r.norm_topk_prob = False
    _, w = r(x)
    scores = torch.sigmoid(r.router_logits(x))
    assert torch.allclose(w.sort(-1, descending=True).values.flatten().sort().values,
                          scores.topk(c.num_experts_per_tok, -1).values.flatten().sort().values,
                          atol=1e-6), "unnormalized weights must be raw sigmoid scores"
    assert (w > 0).all() and (w < 1).all(), "sigmoid range"


def test_bias_affects_selection_only():
    r = BiBoMoERouter(make_config()).to(DEVICE)
    x = torch.randn(1, 6, 64, device=DEVICE)
    assert r.bias.requires_grad is False, "bias is heuristic, never optimizer-managed"
    with torch.no_grad():
        r.bias[0] = 5.0
    idx, w = r(x)
    assert (idx == 0).any(), "a large bias must change selection"
    assert w.max() <= 1.0 + 1e-6, "bias must never leak into the combine weights"


@pytest.mark.parametrize("norm", NORMS)
def test_router_gradients_reach_the_projection(norm):
    r = BiBoMoERouter(make_config(norm_topk_prob=norm)).to(DEVICE)
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
    c = make_config(bias_update_factor=0.4)
    layer = BiBoMoELayer(c).to(DEVICE)
    tpe = torch.full((c.num_routed_experts,), 10.0, device=DEVICE)
    tpe[0] = 100.0
    before = layer.gate.bias.clone()
    layer.update_bias(tpe)
    delta = layer.gate.bias - before
    assert delta[0] < 0, "the overloaded expert must be discouraged"
    assert (delta[1:] > 0).all(), "starved experts must be encouraged"


def test_every_expert_including_the_specials_is_balanced():
    """One rule, whole vector: the specials are balanced too (freezing them was
    balance_exclude_specials / glu_token_budget, both removed Aug 1 2026)."""
    c = make_config(num_routed_experts=6, special_expert_pairs=1, bias_update_factor=0.4)
    layer = BiBoMoELayer(c).to(DEVICE)
    npg = layer.experts.num_glu_experts
    tpe = torch.full((c.num_routed_experts,), 10.0, device=DEVICE)
    tpe[npg:] = 500.0                                        # the specials are the hogs here
    layer.update_bias(tpe)
    assert (layer.gate.bias[npg:] < 0).all(), "special biases must move, not stay frozen at 0"
    assert (layer.gate.bias[:npg] > 0).all()


def test_bias_update_factor_zero_disables_balancing():
    c = make_config(bias_update_factor=0.0)
    layer = BiBoMoELayer(c).to(DEVICE).train()
    layer(torch.randn(2, 8, c.hidden_size, device=DEVICE))
    tpe = torch.full((c.num_routed_experts,), 10.0, device=DEVICE)
    tpe[0] = 1000.0
    layer.update_bias(tpe)
    assert layer.gate.bias.abs().max() == 0, "u=0 must never touch the bias"


def test_bias_update_fires_on_the_expected_step_interval():
    """Triggering on forward STEPS (not device tokens) keeps every DDP rank in lockstep."""
    tokens_per_fwd = 2 * 8
    c = make_config(bias_update_threshold=tokens_per_fwd * 3)
    layer = BiBoMoELayer(c).to(DEVICE).train()
    idx = torch.zeros(2, 8, 2, dtype=torch.long, device=DEVICE)
    fired = [layer._balance_step(idx, tokens_per_fwd) is not None for _ in range(7)]
    assert layer._update_every == 3
    assert fired == [False, False, True, False, False, True, False]


def test_router_stays_fp32_in_a_half_precision_model():
    layer = BiBoMoELayer(make_config()).to(DEVICE)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    layer = layer.to(dt)
    x = torch.randn(2, 5, 64, device=DEVICE, dtype=dt)
    _, w = layer.gate(x)
    assert w.dtype == torch.float32
    assert layer(x).dtype == dt, "the layer output must return to the model dtype"


@pytest.mark.parametrize("norm", NORMS + [False])
def test_moe_layer_forward_backward(norm):
    layer = BiBoMoELayer(make_config(norm_topk_prob=norm)).to(DEVICE)
    x = torch.randn(2, 6, 64, device=DEVICE, requires_grad=True)
    layer(x).sum().backward()
    assert torch.isfinite(layer.experts.gate_up_proj.grad).all()
    assert layer.experts.gate_up_proj.grad.abs().max() > 0


def test_shared_expert_is_off_by_default_and_adds_directly():
    assert make_config().use_shared_expert is False, "off by default to param-match Qwen"
    layer = BiBoMoELayer(make_config(use_shared_expert=True)).to(DEVICE)
    assert len(layer.shared_experts_list) == 1
    assert not hasattr(layer, "moe_shared_scaling"), "the scaling scalar was removed"


def test_bias_deviations_sum_to_zero_so_there_is_no_common_mode_drift():
    """Why sign() was dropped: raw deviations sum to EXACTLY 0, so the MEAN bias cannot move, only
    the spread. sign()ed deviations do not — a right-skewed load puts most experts below the mean, so
    most get +1 and the whole block floats up (measured +1.28, ~84% of the accumulated bias)."""
    skewed = torch.tensor([500.0, 300.0] + [10.0] * 8 + [50.0] * 4, device=DEVICE)   # 14 experts
    c = make_config(bias_update_factor=0.4, num_routed_experts=14, special_expert_pairs=2)
    assert c.num_routed_experts == 14
    layer = BiBoMoELayer(c).to(DEVICE)
    layer.update_bias(skewed)
    assert abs(float(layer.gate.bias.mean())) < 1e-6, "the update must not shift the mean bias"
    assert float(layer.gate.bias.std()) > 0, "...but it must still spread the biases"


def test_bias_step_shrinks_as_load_approaches_balance():
    """Proportional control has a fixed point; bang-bang did not, which is why u=0.01 under sign held
    its load target yet lost 0.094 loss to the permanent +-u dither."""
    c = make_config(bias_update_factor=1.0)
    steps = []
    for skew in (10.0, 1.0, 0.0):                       # progressively closer to perfectly balanced
        layer = BiBoMoELayer(c).to(DEVICE)
        tpe = torch.full((c.num_routed_experts,), 100.0, device=DEVICE)
        tpe[0] += skew * 10
        layer.update_bias(tpe)
        steps.append(float(layer.gate.bias.abs().max()))
    assert steps[0] > steps[1] > steps[2], f"steps must shrink with the deviation, got {steps}"
    assert steps[2] < 1e-6, "a perfectly balanced load must produce NO update"

    layer = BiBoMoELayer(make_config(bias_update_factor=0.4)).to(DEVICE)
    tpe = torch.full((layer.num_routed_experts,), 100.0, device=DEVICE)
    tpe[0] += 1e-3                                        # essentially, but not exactly, balanced
    layer.update_bias(tpe)
    assert float(layer.gate.bias.abs().max()) < 1e-6, \
        "an infinitesimal deviation must give an infinitesimal step (sign gave a full u -- the dither)"
