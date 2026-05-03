import math

import pytest
import torch

from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoAttention, apply_rotary_pos_emb, repeat_kv


def make_config(attention_type="softmax", **overrides):
    kwargs = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        attention_type=attention_type,
        sliding_window=3,
        use_ssmax=False,
        moe_shared_scaling=2.0,
    )
    kwargs.update(overrides)
    return BiBoConfig(**kwargs)


def make_causal_mask(batch_size, seq_len, dtype=torch.float32):
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype)
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask[:, :, causal] = torch.finfo(dtype).min
    return mask


def projection_states(attn, hidden_states):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return query, key, value


def forward_attention(attn, hidden_states):
    cos, sin = attn.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
    attention_mask = make_causal_mask(hidden_states.shape[0], hidden_states.shape[1], hidden_states.dtype)
    output, _, _ = attn(hidden_states, (cos, sin), attention_mask)
    return output


@pytest.mark.parametrize("attention_type", ["softmax", "sliding_window", "linear", "gdn", "kda"])
def test_attention_variants_return_finite_outputs_and_gradients(attention_type):
    torch.manual_seed(0)
    config = make_config(attention_type)
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=(attention_type == "sliding_window"))
    hidden_states = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    cos, sin = attn.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
    attention_mask = make_causal_mask(2, 5, hidden_states.dtype)

    output, attn_weights, cache = attn(hidden_states, (cos, sin), attention_mask)

    assert output.shape == hidden_states.shape
    assert attn_weights is None
    assert cache is None
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim"),
    [(1, 1, 4, 1, 8), (2, 5, 4, 2, 8), (3, 7, 6, 3, 4)],
)
def test_projection_rope_and_gqa_shape_contract(batch_size, seq_len, num_heads, num_kv_heads, head_dim):
    config = make_config(
        "softmax",
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )
    attn = BiBoAttention(config, layer_idx=0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    query, key, value = projection_states(attn, hidden_states)

    assert query.shape == (batch_size, num_heads, seq_len, head_dim)
    assert key.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert value.shape == (batch_size, num_kv_heads, seq_len, head_dim)

    cos, sin = attn.rotary_emb(hidden_states, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    assert query.shape == (batch_size, num_heads, seq_len, head_dim)
    assert key.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert repeat_kv(key, attn.num_key_value_groups).shape == (batch_size, num_heads, seq_len, head_dim)
    assert repeat_kv(value, attn.num_key_value_groups).shape == (batch_size, num_heads, seq_len, head_dim)


@pytest.mark.parametrize("attention_type", ["linear", "gdn", "kda"])
@pytest.mark.parametrize(
    ("batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim"),
    [(1, 3, 4, 1, 8), (2, 6, 4, 2, 8), (2, 4, 8, 4, 4)],
)
def test_recurrent_attention_shape_contract(attention_type, batch_size, seq_len, num_heads, num_kv_heads, head_dim):
    config = make_config(
        attention_type,
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
    )
    attn = BiBoAttention(config, layer_idx=0)
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    query, key, value = projection_states(attn, hidden_states)
    key = repeat_kv(key, attn.num_key_value_groups)
    value = repeat_kv(value, attn.num_key_value_groups)

    output = attn.eager_recurrent_attention(hidden_states, query, key, value)

    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("attention_type", ["gdn", "kda"])
def test_delta_gate_and_beta_projection_shape_contract(attention_type):
    config = make_config(attention_type, hidden_size=48, num_attention_heads=6, num_key_value_heads=2)
    attn = BiBoAttention(config, layer_idx=0)
    hidden_states = torch.randn(2, 5, config.hidden_size)

    beta = attn.delta_beta_proj(hidden_states)
    gate = attn.delta_gate_proj(hidden_states)

    assert beta.shape == (2, 5, config.num_attention_heads)
    if attention_type == "gdn":
        assert gate.shape == (2, 5, config.num_attention_heads)
    else:
        assert gate.shape == (2, 5, config.hidden_size)


@pytest.mark.parametrize("attention_type", ["softmax", "sliding_window", "linear", "gdn", "kda"])
def test_attention_is_causal_for_earlier_outputs(attention_type):
    torch.manual_seed(4)
    config = make_config(attention_type, sliding_window=2)
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=(attention_type == "sliding_window"))
    hidden_states = torch.randn(2, 6, config.hidden_size)
    modified_hidden_states = hidden_states.clone()
    modified_hidden_states[:, -1, :] = torch.randn_like(modified_hidden_states[:, -1, :]) * 10.0

    output = forward_attention(attn, hidden_states)
    modified_output = forward_attention(attn, modified_hidden_states)

    torch.testing.assert_close(output[:, :-1, :], modified_output[:, :-1, :], atol=1e-5, rtol=1e-5)


def test_sliding_window_supports_decode_shape_with_longer_kv_cache():
    torch.manual_seed(5)
    config = make_config("sliding_window", sliding_window=3)
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=True)
    query = torch.randn(2, config.num_attention_heads, 2, attn.head_dim)
    key = torch.randn(2, config.num_attention_heads, 5, attn.head_dim)
    value = torch.randn(2, config.num_attention_heads, 5, attn.head_dim)
    mask = torch.zeros(2, 1, 2, 5)

    output = attn.eager_sliding_window_attention(query, key, value, mask, sliding_window=3)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_source_token_mask_shape_and_values_for_recurrent_attention():
    config = make_config("linear")
    attn = BiBoAttention(config, layer_idx=0)
    mask = torch.zeros(2, 1, 4, 4)
    mask[:, :, :, 2] = torch.finfo(mask.dtype).min

    source_mask = attn._source_token_mask(mask, batch_size=2, seq_len=4, dtype=torch.float32)

    assert source_mask.shape == (2, 1, 4, 1)
    expected = torch.tensor([1.0, 1.0, 0.0, 1.0]).view(1, 1, 4, 1).expand(2, 1, 4, 1)
    torch.testing.assert_close(source_mask, expected)


def test_sliding_window_matches_manual_local_softmax_attention():
    torch.manual_seed(1)
    config = make_config("sliding_window", sliding_window=2)
    attn = BiBoAttention(config, layer_idx=0, use_sliding_window=True)

    query = torch.randn(1, config.num_attention_heads, 4, attn.head_dim)
    key = torch.randn(1, config.num_attention_heads, 4, attn.head_dim)
    value = torch.randn(1, config.num_attention_heads, 4, attn.head_dim)
    attention_mask = make_causal_mask(1, 4, query.dtype)

    actual = attn.eager_sliding_window_attention(query, key, value, attention_mask, sliding_window=2)

    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(attn.head_dim)
    allowed = torch.zeros(4, 4, dtype=torch.bool)
    for query_idx in range(4):
        start = max(0, query_idx - 1)
        allowed[query_idx, start : query_idx + 1] = True
    scores = scores.masked_fill(~allowed.view(1, 1, 4, 4), torch.finfo(scores.dtype).min)
    expected = torch.matmul(torch.softmax(scores, dim=-1), value)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_linear_attention_matches_manual_cumulative_kv_state():
    torch.manual_seed(2)
    config = make_config("linear", linear_attention_feature_map="elu")
    attn = BiBoAttention(config, layer_idx=0)
    query = torch.randn(1, config.num_attention_heads, 4, attn.head_dim)
    key = torch.randn(1, config.num_attention_heads, 4, attn.head_dim)
    value = torch.randn(1, config.num_attention_heads, 4, attn.head_dim)
    hidden_states = torch.randn(1, 4, config.hidden_size)

    actual = attn.eager_recurrent_attention(hidden_states, query, key, value)

    q = torch.nn.functional.elu(query) + 1.0 + config.linear_attention_eps
    k = torch.nn.functional.elu(key) + 1.0 + config.linear_attention_eps
    kv_state = torch.cumsum(torch.einsum("bhtd,bhte->bhtde", k, value), dim=2)
    k_state = torch.cumsum(k, dim=2)
    numerator = torch.einsum("bhtd,bhtde->bhte", q, kv_state)
    denominator = torch.einsum("bhtd,bhtd->bht", q, k_state).unsqueeze(-1)
    expected = numerator / denominator.clamp_min(config.linear_attention_eps)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("attention_type", ["gdn", "kda"])
def test_delta_attention_uses_learned_gates(attention_type):
    torch.manual_seed(3)
    config = make_config(attention_type)
    attn = BiBoAttention(config, layer_idx=0)
    hidden_states = torch.randn(2, 4, config.hidden_size)
    cos, sin = attn.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])

    output_a, _, _ = attn(hidden_states, (cos, sin), make_causal_mask(2, 4))
    with torch.no_grad():
        attn.delta_gate_proj.bias.fill_(-10.0)
    output_b, _, _ = attn(hidden_states, (cos, sin), make_causal_mask(2, 4))

    assert not torch.allclose(output_a, output_b)


def test_config_rejects_unknown_attention_type():
    with pytest.raises(ValueError, match="attention_type"):
        make_config("not-real")
