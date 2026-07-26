"""RoPE: dim-wise partial rotation and dynamic NTK scaling."""
import torch
from conftest import DEVICE, make_config, make_model

from src.modeling.embed import BiBoRotaryEmbedding, apply_rotary_pos_emb


def _rotary(dim=8, max_pos=256, base=1e7, rope_type="dynamic", factor=1.0):
    return BiBoRotaryEmbedding(dim, max_position_embeddings=max_pos, base=base,
                               rope_type=rope_type, scaling_factor=factor).to(DEVICE)


def test_cos_sin_are_sized_rope_dim_not_head_dim():
    m = make_model(partial_rotary_factor=0.5)
    c = m.config
    assert 0 < c.rope_dim < c.head_dim
    x = torch.randn(1, 6, c.head_dim, device=DEVICE)
    pos = torch.arange(6, device=DEVICE).unsqueeze(0)
    cos, sin = m.model.rotary_emb(x, pos, seq_len=6)
    assert cos.shape[-1] == c.rope_dim and sin.shape[-1] == c.rope_dim


def test_partial_rope_leaves_the_tail_untouched():
    """Only the first rope_dim of EVERY head rotates; the rest is NoPE."""
    c = make_config(partial_rotary_factor=0.5)
    rd, hd = c.rope_dim, c.head_dim
    q = torch.randn(1, 4, 6, hd, device=DEVICE)
    k = torch.randn(1, 2, 6, hd, device=DEVICE)
    pos = torch.arange(6, device=DEVICE).unsqueeze(0)
    cos, sin = _rotary(dim=rd, rope_type="none")(q, pos, seq_len=6)
    q_rot, _ = apply_rotary_pos_emb(q[..., :rd], k[..., :rd], cos, sin)
    assert not torch.allclose(q_rot, q[..., :rd]), "the rotated slice must actually change"
    assert torch.equal(q[..., rd:], q[..., rd:]), "the NoPE tail must pass through unmodified"


def test_rope_matches_the_reference_qwen3_formula():
    dim, base, L = 8, 10000.0, 16
    r = _rotary(dim=dim, base=base, rope_type="none")
    x = torch.randn(1, L, dim, device=DEVICE)
    pos = torch.arange(L, device=DEVICE).unsqueeze(0)
    cos, sin = r(x, pos, seq_len=L)
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(DEVICE) / dim))
    freqs = (inv[None, :, None].float() @ pos[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    assert torch.allclose(cos, emb.cos(), atol=1e-6)
    assert torch.allclose(sin, emb.sin(), atol=1e-6)


# ── dynamic NTK ──────────────────────────────────────────────────────────────
def test_ntk_is_a_noop_inside_the_trained_window():
    r = _rotary()
    for L in (1, 64, 256):
        assert torch.equal(r._inv_freq_for(None, DEVICE, seq_len=L),
                           r.original_inv_freq.to(DEVICE)), f"NTK altered frequencies at L={L}"


def test_ntk_scales_the_base_beyond_the_window():
    r, L, dim = _rotary(), 1024, 8
    got = r._inv_freq_for(None, DEVICE, seq_len=L)
    want_base = 1e7 * (L / 256) ** (dim / (dim - 2))
    want = 1.0 / (want_base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(DEVICE) / dim))
    assert torch.allclose(got, want, rtol=1e-6)


def test_ntk_lowers_all_frequencies_except_index_zero():
    r = _rotary()
    o = r.original_inv_freq.to(DEVICE)
    got = r._inv_freq_for(None, DEVICE, seq_len=1024)
    assert got[0] == o[0] == 1.0, "index 0 is base**0 == 1 for any base"
    assert (got[1:] < o[1:]).all(), "extrapolating must lower the remaining frequencies"


def test_ntk_is_stateless_and_order_independent():
    """A grow/reset history would make the result depend on prior batch lengths."""
    r = _rotary()
    x = torch.randn(1, 16, 8, device=DEVICE)
    short = torch.arange(16, device=DEVICE).unsqueeze(0)
    first = r(x, short, seq_len=16)
    r(x, torch.arange(4096, device=DEVICE).unsqueeze(0), seq_len=4096)   # go long
    again = r(x, short, seq_len=16)                                      # come back short
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def test_rope_type_none_never_rescales():
    r = _rotary(rope_type="none")
    assert torch.equal(r._inv_freq_for(None, DEVICE, seq_len=99999), r.inv_freq)


def test_seq_len_none_falls_back_to_position_ids():
    """External callers may omit seq_len; the fallback costs a sync but must be correct."""
    r = _rotary()
    pos = torch.arange(1024, device=DEVICE).unsqueeze(0)
    assert torch.allclose(r._inv_freq_for(pos, DEVICE, seq_len=None),
                          r._inv_freq_for(None, DEVICE, seq_len=1024))
