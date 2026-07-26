"""Attention: SSMax, XSA, SWA banding, attention sinks, GQA, padding masks."""
import math

import pytest
import torch
from conftest import DEVICE, make_model, tokens

from src.modeling.attn.ssmax import apply_ssmax_query_scaling
from src.modeling.attn.xsa import apply_xsa

HYBRID = [0, 1, 1, 0]


# ── SSMax ────────────────────────────────────────────────────────────────────
def test_ssmax_init_is_one_over_log_half_max_pos():
    """Starts attention ~neutral instead of ~6x sharper than standard softmax."""
    m = make_model(use_ssmax=True)
    want = 1.0 / math.log(max(m.config.max_position_embeddings / 2, 2.0))
    for layer in m.model.layers:
        a = layer.self_attn
        if a.use_ssmax:
            assert a.ssmax_scale.shape == (1, m.config.num_attention_heads, 1, 1)
            assert abs(a.ssmax_scale.flatten()[0].item() - want) < 1e-6


def test_ssmax_is_forced_off_and_unallocated_on_swa_layers():
    """A window caps n, so a per-query log(n) term is a redundant constant there."""
    m = make_model(use_ssmax=True, hybrid_layer_pattern=HYBRID,
                   add_swa_attention_sink_bias=True, add_full_attention_sink_bias=False)
    for i, layer in enumerate(m.model.layers):
        a = layer.self_attn
        if a.is_swa:
            assert not a.use_ssmax, f"layer {i} is SWA but SSMax is on"
            assert not hasattr(a, "ssmax_scale"), f"layer {i} allocated an unused ssmax_scale"
        else:
            assert a.use_ssmax, f"global layer {i} lost SSMax"


def test_ssmax_counts_real_keys_under_padding():
    """Grid positions over-count by the pad width; n must come from mask.cumsum."""
    q = torch.randn(2, 4, 5, 16, device=DEVICE)
    scale = torch.full((1, 4, 1, 1), 0.14, device=DEVICE)
    mask = torch.tensor([[1] * 8, [0] * 3 + [1] * 5], device=DEVICE)
    masked = apply_ssmax_query_scaling(q, 8, scale, mask.cumsum(-1)[:, -5:])
    plain = apply_ssmax_query_scaling(q, 8, scale, None)
    assert torch.allclose(masked[0], plain[0], atol=1e-5), "unpadded row must be unaffected"
    assert not torch.allclose(masked[1], plain[1]), "padded row must differ"


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


# ── SWA banding + sinks ──────────────────────────────────────────────────────
def test_swa_band_mask_is_exact():
    W, S = 4, 12
    m = make_model(hybrid_layer_pattern=[1] * 4, sliding_window=W, use_ssmax=False,
                   add_swa_attention_sink_bias=False)
    attn = m(tokens(1, S), output_attentions=True).attentions[1][0, 0]
    for q in range(S):
        lo = max(0, q - W + 1)
        outside = attn[q, :lo].abs().sum() + attn[q, q + 1:].abs().sum()
        assert outside.item() < 1e-6, f"query {q} attends outside [{lo}, {q}]"
        assert attn[q, lo:q + 1].sum().item() > 0.99, f"query {q} band mass != 1"


def test_sink_placement_follows_the_config():
    m = make_model(use_ssmax=True, hybrid_layer_pattern=HYBRID,
                   add_swa_attention_sink_bias=True, add_full_attention_sink_bias=False)
    for i, layer in enumerate(m.model.layers):
        a = layer.self_attn
        if a.is_swa:
            assert a.attention_sink_bias is not None, f"SWA layer {i} has no sink"
            assert a.attention_sink_bias.shape == (m.config.num_attention_heads,)
        else:
            assert a.attention_sink_bias is None, f"global layer {i} got an unrequested sink"


def test_all_global_model_builds_zero_sink_params():
    m = make_model()
    assert all(l.self_attn.attention_sink_bias is None for l in m.model.layers)


def test_attention_sink_receives_gradient():
    m = make_model(hybrid_layer_pattern=[1] * 4, sliding_window=4, use_ssmax=False,
                   add_swa_attention_sink_bias=True)
    x = tokens(2, 8)
    m(x, labels=x).loss.backward()
    sinks = [l.self_attn.attention_sink_bias for l in m.model.layers
             if l.self_attn.attention_sink_bias is not None]
    assert sinks
    for s in sinks:
        assert s.grad is not None and torch.isfinite(s.grad).all() and s.grad.abs().max() > 0


# ── GQA ──────────────────────────────────────────────────────────────────────
def test_gqa_projection_geometry():
    a = make_model().model.layers[1].self_attn
    assert a.num_key_value_groups == 2
    assert a.q_proj.out_features == 4 * a.head_dim
    assert a.k_proj.out_features == 2 * a.head_dim == a.v_proj.out_features


# ── padding ──────────────────────────────────────────────────────────────────
def test_left_padded_forward_matches_unpadded_reference():
    torch.manual_seed(0)
    m = make_model(use_ssmax=True).eval()
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
    m = make_model(hybrid_layer_pattern=HYBRID, sliding_window=4, use_ssmax=True, use_xsa=True)
    x = tokens(2, 12)
    mask = torch.ones_like(x)
    mask[0, :4] = 0
    out = m(x, attention_mask=mask, labels=x)
    out.loss.backward()
    assert torch.isfinite(out.loss)
