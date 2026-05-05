"""Test modular BiBo implementation"""
import sys
sys.path.insert(0, '.')
import torch
import pytest
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoModel, BiBoForCausalLM
from src.modeling.attn import BiBoAttention
from src.modeling.ffn import BiBoMLP, BiBoMoELayer
from src.modeling.layers import BiBoCausalResidualConv, BiBoMultiStreamResidual, BiBoResidualGate

@pytest.fixture
def cfg():
    """Test config fixture"""
    return BiBoConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_routed_experts=8,
        num_experts_per_tok=2,
    )

def test_attention(cfg):
    """Test attention module"""
    attn = BiBoAttention(cfg, layer_idx=0)
    x = torch.randn(2, 8, cfg.hidden_size)
    pos_emb = (torch.randn(8, cfg.hidden_size // cfg.num_attention_heads),
               torch.randn(8, cfg.hidden_size // cfg.num_attention_heads))
    out, _, _ = attn(x, pos_emb, None)
    assert out.shape == x.shape
    print(f"✓ Attention: {x.shape} → {out.shape}")

def test_mlp(cfg):
    """Test MLP"""
    mlp = BiBoMLP(cfg, is_expert=False)
    x = torch.randn(2, 8, cfg.hidden_size)
    out = mlp(x)
    assert out.shape == x.shape
    print(f"✓ MLP: {x.shape} → {out.shape}")

def test_moe(cfg):
    """Test MoE layer"""
    moe = BiBoMoELayer(cfg)
    x = torch.randn(2, 8, cfg.hidden_size)
    out = moe(x)
    assert out.shape == x.shape
    print(f"✓ MoE: {x.shape} → {out.shape}")

def test_model(cfg):
    """Test full model"""
    model = BiBoModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids)
    assert out.last_hidden_state.shape == (2, 8, cfg.hidden_size)
    print(f"✓ Model: input {input_ids.shape} → hidden {out.last_hidden_state.shape}")

def test_causal_lm(cfg):
    """Test causal LM"""
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids)
    print(f"✓ CausalLM: input {input_ids.shape} → logits {out.logits.shape}")
    
    # Test with labels
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids, labels=labels)
    print(f"✓ CausalLM loss: {out.loss.item():.4f}")

def test_forward_backward(cfg):
    """Test forward + backward"""
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    
    out = model(input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    
    # Check grads exist
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"✓ Backward: {has_grads}/{total_params} params have grads")

def test_residual_gate_starts_near_identity():
    """Residual write gate should preserve baseline flow at init."""
    cfg = BiBoConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_routed_experts=8,
        residual_gate_type="token",
        residual_gate_init=0.95,
    )
    gate = BiBoResidualGate(cfg, "attn")
    x = torch.randn(2, 8, cfg.hidden_size)
    branch = torch.ones_like(x)

    out = gate(x, branch)
    assert out.shape == branch.shape
    assert torch.allclose(out.mean(), torch.tensor(0.95), atol=1e-5)
    assert gate.stats()["attn/mean"] == pytest.approx(0.95, abs=1e-5)

def test_residual_gates_work_in_model_and_receive_gradients():
    """Token gates should expose per-branch flow stats and train."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_gate_type="token",
        residual_gate_init=0.95,
    )
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    out.loss.backward()
    stats = model.model.residual_gate_stats()

    assert "layer_0/attn/mean" in stats
    assert "layer_0/mlp/mean" in stats
    assert stats["layer_0/attn/mean"] == pytest.approx(0.95, abs=1e-5)
    assert stats["layer_0/mlp/mean"] == pytest.approx(0.95, abs=1e-5)
    assert model.model.layers[0].attn_residual_gate.bias.grad is not None
    assert model.model.layers[0].mlp_residual_gate.bias.grad is not None

def test_causal_residual_conv_starts_near_current_state():
    """Depth-causal residual conv should prefer the current layer at init."""
    cfg = BiBoConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_routed_experts=8,
        residual_mixer_type="causal_conv",
        residual_conv_kernel_size=3,
        residual_conv_init=0.9,
    )
    mixer = BiBoCausalResidualConv(cfg, layer_idx=0)
    current = torch.ones(2, 8, cfg.hidden_size)
    previous = torch.zeros_like(current)

    mixed = mixer(current, (previous, previous))
    assert mixed.shape == current.shape
    assert torch.allclose(mixed.mean(), torch.tensor(0.9), atol=1e-5)
    stats = mixer.stats()
    assert stats["layer_0/residual_conv/current_weight"] == pytest.approx(0.9, abs=1e-5)
    assert stats["layer_0/residual_conv/previous_mass"] == pytest.approx(0.1, abs=1e-5)
    assert stats["layer_0/residual_conv/num_states"] == 3

def test_causal_residual_conv_works_in_model_and_receives_gradients():
    """Model should train causal depth-conv residual mixer parameters."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_mixer_type="causal_conv",
        residual_conv_kernel_size=3,
        residual_conv_init=0.9,
    )
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    out.loss.backward()
    stats = model.model.residual_mixer_stats()

    assert "layer_0/residual_conv/current_weight" in stats
    assert "layer_1/residual_conv/current_weight" in stats
    assert stats["layer_0/residual_conv/num_states"] == 1
    assert stats["layer_1/residual_conv/num_states"] == 2
    assert stats["layer_2/residual_conv/num_states"] == 3
    assert model.model.layers[0].residual_mixer.kernel_logits.grad is not None
    assert model.model.layers[1].residual_mixer.kernel_logits.grad is not None

def test_dynamic_causal_residual_conv_works_in_model_and_receives_gradients():
    """Dynamic depth conv should learn token-conditioned residual-depth reads."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_mixer_type="dynamic_causal_conv",
        residual_conv_kernel_size=3,
        residual_conv_init=0.9,
    )
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    out.loss.backward()
    stats = model.model.residual_mixer_stats()

    assert stats["layer_0/residual_conv/num_states"] == 1
    assert stats["layer_1/residual_conv/num_states"] == 2
    assert stats["layer_2/residual_conv/num_states"] == 3
    assert model.model.layers[0].residual_mixer.dynamic_weight.grad is not None
    assert model.model.layers[0].residual_mixer.dynamic_bias.grad is not None

def test_residual_history_can_include_input_when_requested():
    """Depth residual history should exclude embeddings by default but keep the option."""
    base_kwargs = dict(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_mixer_type="causal_conv",
        residual_conv_kernel_size=3,
    )
    input_ids = torch.randint(0, 100, (2, 8))

    default_model = BiBoForCausalLM(BiBoConfig(**base_kwargs))
    default_model(input_ids)
    default_stats = default_model.model.residual_mixer_stats()

    include_input_model = BiBoForCausalLM(
        BiBoConfig(**base_kwargs, residual_history_include_input=True)
    )
    include_input_model(input_ids)
    include_input_stats = include_input_model.model.residual_mixer_stats()

    assert default_stats["layer_0/residual_conv/num_states"] == 1
    assert default_stats["layer_1/residual_conv/num_states"] == 2
    assert include_input_stats["layer_0/residual_conv/num_states"] == 2
    assert include_input_stats["layer_1/residual_conv/num_states"] == 3

def test_multistream_residual_prefers_main_stream_at_init():
    """mHC-style residual streams should read/write mostly from stream 0 at init."""
    cfg = BiBoConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_routed_experts=8,
        residual_num_streams=3,
        residual_stream_gate_type="token",
        residual_stream_read_init=0.9,
        residual_stream_write_init=0.8,
    )
    mixer = BiBoMultiStreamResidual(cfg, layer_idx=0)
    streams = torch.stack([
        torch.ones(2, 8, cfg.hidden_size),
        torch.zeros(2, 8, cfg.hidden_size),
        torch.zeros(2, 8, cfg.hidden_size),
    ], dim=2)
    update = torch.ones(2, 8, cfg.hidden_size)

    read = mixer.read(streams)
    streams = mixer.write(streams, update)
    stats = mixer.stats()

    assert read.shape == (2, 8, cfg.hidden_size)
    assert torch.allclose(read.mean(), torch.tensor(0.9), atol=1e-5)
    assert stats["layer_0/streams/read_main"] == pytest.approx(0.9, abs=1e-5)
    assert stats["layer_0/streams/write_main"] == pytest.approx(0.8, abs=1e-5)

def test_multistream_residual_works_in_model_and_receives_gradients():
    """Full model should train stream read/write gate parameters."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_num_streams=2,
        residual_stream_gate_type="token",
        residual_stream_read_init=0.99,
        residual_stream_write_init=0.99,
    )
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    out.loss.backward()
    stats = model.model.residual_stream_stats()

    assert "layer_0/streams/read_main" in stats
    assert "layer_1/streams/write_main" in stats
    assert model.model.residual_stream_mixers[0].read_bias.grad is not None
    assert model.model.residual_stream_mixers[0].write_bias.grad is not None
    assert model.model.residual_stream_mixers[0].read_weight.grad is not None
    assert model.model.residual_stream_mixers[0].write_weight.grad is not None

def test_multistream_model_reads_once_per_layer_plus_final_read():
    """Stream reads should not repeat after every write; only final output is re-read."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_num_streams=2,
        residual_stream_gate_type="scalar",
    )
    model = BiBoForCausalLM(cfg)

    for mixer in model.model.residual_stream_mixers:
        mixer._test_read_calls = 0
        original_read = mixer.read

        def counted_read(streams, *, _mixer=mixer, _original_read=original_read):
            _mixer._test_read_calls += 1
            return _original_read(streams)

        mixer.read = counted_read

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    model(input_ids)

    assert [mixer._test_read_calls for mixer in model.model.residual_stream_mixers] == [1, 2]

def test_delay_line_residual_shifts_states_and_reads_depth_axis():
    """Delay-line streams should encode current, one-layer-old, two-layer-old states."""
    cfg = BiBoConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_routed_experts=8,
        residual_num_streams=3,
        residual_stream_mode="delay_line",
        residual_stream_gate_type="scalar",
        residual_stream_read_init=0.8,
    )
    mixer = BiBoMultiStreamResidual(cfg, layer_idx=0)
    streams = torch.stack([
        torch.ones(2, 8, cfg.hidden_size),
        torch.full((2, 8, cfg.hidden_size), 2.0),
        torch.full((2, 8, cfg.hidden_size), 3.0),
    ], dim=2)
    new_state = torch.full((2, 8, cfg.hidden_size), 10.0)

    read = mixer.read(streams)
    shifted = mixer.write(streams, new_state)
    stats = mixer.stats()

    assert read.shape == (2, 8, cfg.hidden_size)
    assert read.mean().item() == pytest.approx(1.3, abs=1e-5)
    assert torch.allclose(shifted[:, :, 0, :], new_state)
    assert torch.allclose(shifted[:, :, 1, :], streams[:, :, 0, :])
    assert torch.allclose(shifted[:, :, 2, :], streams[:, :, 1, :])
    assert mixer.write_bias is None
    assert stats["layer_0/streams/read_main"] == pytest.approx(0.8, abs=1e-5)
    assert stats["layer_0/streams/write_main"] == pytest.approx(1.0, abs=1e-5)

def test_delay_line_residual_works_in_model_and_receives_gradients():
    """Delay-line mode should train only the causal depth read gates."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_num_streams=3,
        residual_stream_mode="delay_line",
        residual_stream_gate_type="token",
        residual_stream_read_init=0.9,
    )
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    out.loss.backward()
    stats = model.model.residual_stream_stats()

    assert torch.isfinite(out.loss)
    assert "layer_0/streams/read_main" in stats
    assert stats["layer_0/streams/write_main"] == pytest.approx(1.0, abs=1e-5)
    assert model.model.residual_stream_mixers[0].read_bias.grad is not None
    assert model.model.residual_stream_mixers[0].read_weight.grad is not None
    assert model.model.residual_stream_mixers[0].write_bias is None
    assert model.model.residual_stream_mixers[0].write_weight is None

def test_residual_experiments_can_be_combined():
    """Write gates, stream gates, and dynamic depth conv should compose."""
    cfg = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        use_sliding_window=False,
        residual_gate_type="token",
        residual_mixer_type="dynamic_causal_conv",
        residual_conv_kernel_size=3,
        residual_num_streams=2,
        residual_stream_gate_type="token",
    )
    model = BiBoForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    out.loss.backward()

    assert torch.isfinite(out.loss)
    assert model.model.residual_gate_stats()
    assert model.model.residual_mixer_stats()
    assert model.model.residual_stream_stats()

def count_params(model):
    """Count model params"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Params: {total:,} total, {trainable:,} trainable")

if __name__ == "__main__":
    print("Testing modular BiBo implementation...\n")
    
    cfg = test_config()
    print()
    
    test_attention(cfg)
    test_mlp(cfg)
    test_moe(cfg)
    print()
    
    test_model(cfg)
    model = BiBoModel(cfg)
    count_params(model)
    print()
    
    test_causal_lm(cfg)
    print()
    
    test_forward_backward(cfg)
    print()
    
    print("✅ All tests passed!")
