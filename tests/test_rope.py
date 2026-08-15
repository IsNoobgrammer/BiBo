"""RoPE, under the FIXED positional-encoding architecture (Aug 14 2026).

Full-attention layers are NoPE; sliding-window layers get full RoPE over the whole head_dim.
There is no partial-rotary fraction, no per-layer-type width, no second base and no NTK scaling,
so the tests that used to cover those knobs are gone with them -- what is left has to assert the
architecture itself, because that is now the only thing that can regress.
"""
import torch
from conftest import DEVICE, make_config, make_model

from src.modeling.embed import BiBoRotaryEmbedding, apply_rotary_pos_emb


def _rotary(dim=8, base=1e7):
    return BiBoRotaryEmbedding(dim, base=base).to(DEVICE)


def test_cos_sin_span_the_whole_head_dim():
    """No partial slice any more: the rotary is built at head_dim, not a fraction of it."""
    m = make_model()
    c = m.config
    x = torch.randn(1, 6, c.head_dim, device=DEVICE)
    pos = torch.arange(6, device=DEVICE).unsqueeze(0)
    cos, sin = m.model.rotary_emb(x, pos)
    assert cos.shape[-1] == c.head_dim and sin.shape[-1] == c.head_dim


def test_rope_matches_the_reference_qwen3_formula():
    dim, base, L = 8, 10000.0, 16
    r = _rotary(dim=dim, base=base)
    x = torch.randn(1, L, dim, device=DEVICE)
    pos = torch.arange(L, device=DEVICE).unsqueeze(0)
    cos, sin = r(x, pos)
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(DEVICE) / dim))
    freqs = (inv[None, :, None].float() @ pos[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    assert torch.allclose(cos, emb.cos(), atol=1e-6)
    assert torch.allclose(sin, emb.sin(), atol=1e-6)


def test_frequencies_never_rescale_with_length():
    """Dynamic NTK is DELETED. It keyed off max_position_embeddings and silently changed the base
    at long context, so ctx1024 and ctx4095 were measured on two different models."""
    r = _rotary()
    before = r.inv_freq.clone()
    x = torch.randn(1, 8, 8, device=DEVICE)
    r(x, torch.arange(99999, 100007, device=DEVICE).unsqueeze(0))
    assert torch.equal(r.inv_freq, before)


def test_rotary_is_stateless_across_lengths():
    """A grow/reset history would make the result depend on prior batch lengths."""
    r = _rotary()
    x = torch.randn(1, 16, 8, device=DEVICE)
    short = torch.arange(16, device=DEVICE).unsqueeze(0)
    first = r(x, short)
    r(x, torch.arange(4096, device=DEVICE).unsqueeze(0))    # go long
    again = r(x, short)                                     # come back short
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def test_apply_rotary_actually_rotates():
    dim, L = 8, 6
    r = _rotary(dim=dim)
    q = torch.randn(1, 4, L, dim, device=DEVICE)
    k = torch.randn(1, 2, L, dim, device=DEVICE)
    pos = torch.arange(L, device=DEVICE).unsqueeze(0)
    cos, sin = r(torch.randn(1, L, dim, device=DEVICE), pos)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert not torch.allclose(q_rot, q) and not torch.allclose(k_rot, k)
    assert torch.allclose(q_rot[:, :, 0], q[:, :, 0], atol=1e-6), "position 0 is the identity"


# ── the architecture, not the kernel ─────────────────────────────────────────
def test_windowed_layers_rotate_and_full_attention_layers_do_not():
    """The whole point of the round that fixed this: NoPE on global layers, full RoPE on windowed
    ones. Asserted at the ATTENTION module, since that is where the branch lives -- a config that
    says the right thing while attention ignores it is the failure mode worth catching."""
    pattern = [0, 1, 1, 0]
    m = make_model(hybrid_layer_pattern=pattern, sliding_window=8)
    for idx, windowed in enumerate(pattern):
        attn = m.model.layers[idx].self_attn
        assert attn.is_swa == bool(windowed)

    cfg = m.config
    x = torch.randn(2, 6, cfg.hidden_size, device=DEVICE)
    pos = torch.arange(6, device=DEVICE).unsqueeze(0).expand(2, -1)
    pe = m.model.rotary_emb(x, pos)
    shifted = m.model.rotary_emb(x, pos + 3)     # same tokens, different absolute positions

    for idx, windowed in enumerate(pattern):
        attn = m.model.layers[idx].self_attn
        a, _ = attn(x, position_embeddings=pe, attention_mask=None)
        b, _ = attn(x, position_embeddings=shifted, attention_mask=None)
        same = torch.equal(a, b)
        assert same != bool(windowed), (
            f"layer {idx} (windowed={bool(windowed)}) "
            f"{'ignored' if windowed else 'used'} the position offset")


def test_config_exposes_no_rope_knobs():
    """These were deleted on purpose (BiBo 55143cb). If one comes back as an accepted kwarg it is
    silently inert -- a sweep could set it and the model would not change."""
    c = make_config()
    for dead in ("partial_rotary_factor", "swa_partial_rotary_factor", "rope_dim",
                 "swa_rope_dim", "swa_rope_theta", "rope_scaling"):
        assert not hasattr(c, dead), f"{dead} is back on the config"
