"""Attention: XSA, SWA banding, GQA, padding masks."""
import pytest
import torch
from conftest import DEVICE, make_model, tokens

from src.modeling.attn.xsa import apply_xsa

HYBRID = [0, 1, 1, 0]


def test_query_scaling_is_gone():
    """SSMax was removed Aug 2 2026 (refuted). No module, no per-head scale parameter, no knob."""
    with pytest.raises(ImportError):
        from src.modeling.attn.ssmax import apply_ssmax_query_scaling  # noqa: F401
    m = make_model(hybrid_layer_pattern=HYBRID)
    for layer in m.model.layers:
        a = layer.self_attn
        assert not hasattr(a, "ssmax_scale") and not hasattr(a, "use_ssmax")
    assert not hasattr(m.config, "use_ssmax")


# ── XSA ──────────────────────────────────────────────────────────────────────
def test_xsa_output_is_orthogonal_to_its_own_value_vector():
    q_len, H, KV, D = 5, 4, 2, 16
    o = torch.randn(1, H, q_len, D, device=DEVICE)
    v = torch.randn(1, KV, q_len, D, device=DEVICE)
    z = apply_xsa(o, v, enable_gqa=True)
    v_grouped = v.repeat_interleave(H // KV, dim=1)[..., -q_len:, :]
    assert (z * v_grouped).sum(-1).abs().max().item() < 1e-4, "rejection is not exact"
    assert not torch.allclose(z, o), "XSA was a no-op"


def test_xsa_handles_decode_where_q_len_differs_from_kv_len():
    m = make_model(use_xsa=True)
    out = m(tokens(1, 6), use_cache=True)
    step = m(tokens(1, 1), past_key_values=out.past_key_values, use_cache=True)
    assert torch.isfinite(step.logits).all()


# ── SWA banding ──────────────────────────────────────────────────────
def test_swa_band_mask_is_exact():
    W, S = 4, 12
    m = make_model(hybrid_layer_pattern=[1] * 4, sliding_window=W)
    attn = m(tokens(1, S), output_attentions=True).attentions[1][0, 0]
    for q in range(S):
        lo = max(0, q - W + 1)
        outside = attn[q, :lo].abs().sum() + attn[q, q + 1:].abs().sum()
        assert outside.item() < 1e-6, f"query {q} attends outside [{lo}, {q}]"
        assert attn[q, lo:q + 1].sum().item() > 0.99, f"query {q} band mass != 1"


def test_attention_sinks_are_gone():
    """Removed Aug 2 2026. The 524M board refuted them: with XSA on, the sink arm's train loss was
    superimposed on the no-sink arm to 5e-4 in every window while costing 2.5% throughput, 0.74 GB,
    and 2.1x the router-gap volatility -- XSA and the sink drain the same bucket. No parameter, no
    config key, no argument on the flavor modules."""
    import inspect
    from src.modeling.attn.swa import swa_attention
    from src.modeling.attn.full_attention import full_attention
    from src.modeling.attn.utils import eager_attention_forward

    m = make_model(hybrid_layer_pattern=HYBRID, sliding_window=4)
    for i, layer in enumerate(m.model.layers):
        assert not hasattr(layer.self_attn, "attention_sink_bias"), f"layer {i} still has a sink"
    assert not any("sink" in n for n, _ in m.named_parameters())
    for fn in (swa_attention, full_attention, eager_attention_forward):
        assert not any("sink" in p for p in inspect.signature(fn).parameters), \
            f"{fn.__name__} still takes a sink argument"
    # No pytest.raises here: PretrainedConfig swallows unknown kwargs onto the instance instead of
    # rejecting them, so a resurrected key would sit there inert. Absence is the assertion.
    assert not hasattr(m.config, "add_swa_attention_sink_bias")
    assert not hasattr(m.config, "add_full_attention_sink_bias")


# ── GQA ──────────────────────────────────────────────────────────────────────
def test_gqa_projection_geometry():
    a = make_model().model.layers[1].self_attn
    assert a.num_key_value_groups == 2
    assert a.q_proj.out_features == 4 * a.head_dim
    assert a.k_proj.out_features == 2 * a.head_dim == a.v_proj.out_features


# ── padding ──────────────────────────────────────────────────────────────────
def test_left_padded_forward_matches_unpadded_reference():
    torch.manual_seed(0)
    m = make_model().eval()
    real = tokens(1, 6, seed=1)
    ref = m(real).logits
    pad = 3
    padded = torch.cat([torch.zeros(1, pad, dtype=torch.long, device=DEVICE), real], 1)
    mask = torch.cat([torch.zeros(1, pad, device=DEVICE),
                      torch.ones(1, 6, device=DEVICE)], 1).long()
    got = m(padded, attention_mask=mask).logits[:, pad:]
    assert (ref - got).abs().max().item() < 2e-4


def test_all_ones_mask_short_circuits_to_the_fast_path():
    m = make_model().eval()
    x = tokens(2, 6, seed=2)
    ones = torch.ones(2, 6, dtype=torch.long, device=DEVICE)
    assert torch.equal(m(x).logits, m(x, attention_mask=ones).logits)


def test_four_dimensional_mask_is_rejected():
    m = make_model()
    with pytest.raises(ValueError):
        m(tokens(1, 4), attention_mask=torch.ones(1, 1, 4, 4, device=DEVICE))


def test_padded_hybrid_model_is_finite():
    m = make_model(hybrid_layer_pattern=HYBRID, sliding_window=4, use_xsa=True)
    x = tokens(2, 12)
    mask = torch.ones_like(x)
    mask[0, :4] = 0
    out = m(x, attention_mask=mask, labels=x)
    out.loss.backward()
    assert torch.isfinite(out.loss)
