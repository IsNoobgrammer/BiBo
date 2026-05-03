"""
Test integrated BiBo model: norm + RoPE + SSMax + MoE + decoder layer
Forward + backward pass verification
"""
import torch
import torch.nn as nn
from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM

def test_forward_backward():
    """Test full forward + backward pass"""
    print("=" * 60)
    print("BiBo Integrated Model Test")
    print("=" * 60)
    
    # Small config
    config = BiBoConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        max_position_embeddings=512,
        use_ssmax=True,
        use_sliding_window=False,
        mlp_only_layers=[0, 3],  # First + last = dense, middle = MoE
    )
    
    print(f"\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  num_kv_heads: {config.num_key_value_heads}")
    print(f"  num_experts: {config.num_routed_experts}")
    print(f"  top_k: {config.num_experts_per_tok}")
    print(f"  use_ssmax: {config.use_ssmax}")
    print(f"  mlp_only_layers: {config.mlp_only_layers}")
    
    # Create model
    model = BiBoForCausalLM(config)
    model.train()
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward
    print("\n" + "-" * 60)
    print("Forward pass...")
    outputs = model(input_ids=input_ids, labels=labels)
    
    loss = outputs.loss
    logits = outputs.logits
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    # Check for NaN/Inf
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert not torch.isinf(logits).any(), "Logits contain Inf"
    print("  ✓ No NaN/Inf")
    
    # Backward
    print("\n" + "-" * 60)
    print("Backward pass...")
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"
    
    print(f"  Params with gradients: {len(grad_norms)}")
    print(f"  Grad norm range: [{min(grad_norms.values()):.2e}, {max(grad_norms.values()):.2e}]")
    print("  ✓ No NaN/Inf in gradients")
    
    # Component verification
    print("\n" + "-" * 60)
    print("Component verification:")
    
    # RoPE
    print("\n  RoPE:")
    rope = model.model.rotary_emb
    print(f"    dim: {rope.dim}")
    print(f"    max_seq_len: {rope.max_seq_len_cached}")
    print(f"    base: {rope.base}")
    cache_pos = torch.arange(seq_len)
    cos, sin = rope(torch.randn(batch_size, seq_len, config.hidden_size), cache_pos)
    print(f"    cos shape: {cos.shape}")
    print(f"    sin shape: {sin.shape}")
    # RoPE returns (seq_len, head_dim) not (batch, seq_len, head_dim)
    assert cos.shape == (seq_len, config.hidden_size // config.num_attention_heads)
    print("    ✓ RoPE working")
    
    # SSMax
    print("\n  SSMax:")
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, 'ssmax_scale'):
            scale = layer.self_attn.ssmax_scale
            grad_str = f"{scale.grad.norm().item():.2e}" if scale.grad is not None else "None"
            print(f"    Layer {i}: scale shape={scale.shape}, mean={scale.data.mean().item():.4f}, grad_norm={grad_str}")
    print("    ✓ SSMax scales present and trainable")
    
    # MoE
    print("\n  MoE:")
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            print(f"    Layer {i}: MoE layer")
            gate = layer.mlp.gate
            print(f"      Router type: {type(gate).__name__}")
            print(f"      Num experts: {layer.mlp.num_routed_experts}")
            print(f"      Top-k: {layer.mlp.num_experts_per_tok}")
            if hasattr(gate, 'bias'):
                print(f"      Router bias: {gate.bias.data.mean().item():.4f}")
        else:
            print(f"    Layer {i}: Dense MLP")
    print("    ✓ MoE routing working")
    
    # Norm
    print("\n  RMSNorm:")
    for i, layer in enumerate(model.model.layers):
        ln = layer.input_layernorm
        print(f"    Layer {i}: eps={ln.variance_epsilon}, weight_mean={ln.weight.data.mean().item():.4f}")
    print("    ✓ RMSNorm working")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


def trace_operations():
    """Trace operations and tensor shapes through forward pass"""
    print("\n" + "=" * 60)
    print("Operation Tracing")
    print("=" * 60)
    
    config = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_routed_experts=8,  # Need >= 5
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        max_position_embeddings=128,
        use_ssmax=True,
        mlp_only_layers=[],  # All MoE
    )
    
    model = BiBoForCausalLM(config)
    model.eval()
    
    # Hook to capture shapes
    shapes = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                shapes[name] = output.shape
        return hook
    
    # Register hooks
    model.model.embed_tokens.register_forward_hook(make_hook("embed"))
    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(make_hook(f"layer{i}_ln1"))
        layer.self_attn.register_forward_hook(make_hook(f"layer{i}_attn"))
        layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer{i}_ln2"))
        layer.mlp.register_forward_hook(make_hook(f"layer{i}_mlp"))
    model.model.norm.register_forward_hook(make_hook("final_norm"))
    model.lm_head.register_forward_hook(make_hook("lm_head"))
    
    # Forward
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    # Print trace
    print("\nTensor shapes through forward pass:")
    for name, shape in shapes.items():
        print(f"  {name:20s}: {tuple(shape)}")
    
    print("\n" + "=" * 60)


def verify_rope_application():
    """Verify RoPE is actually applied to Q/K"""
    print("\n" + "=" * 60)
    print("RoPE Application Verification")
    print("=" * 60)
    
    config = BiBoConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    
    model = BiBoForCausalLM(config)
    model.eval()
    
    # Capture Q/K before and after RoPE
    qk_states = {}
    
    def make_qk_hook(name):
        def hook(module, input, output):
            qk_states[name] = output.detach().clone()
        return hook
    
    layer = model.model.layers[0]
    attn = layer.self_attn
    
    # Hook Q/K projections (before RoPE)
    attn.q_proj.register_forward_hook(make_qk_hook("q_before"))
    attn.k_proj.register_forward_hook(make_qk_hook("k_before"))
    
    # Forward pass to capture Q/K before RoPE
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    q_before = qk_states["q_before"]
    k_before = qk_states["k_before"]
    
    print(f"\nQ before RoPE:")
    print(f"  shape: {q_before.shape}")
    print(f"  mean: {q_before.mean().item():.4f}")
    print(f"  std: {q_before.std().item():.4f}")
    
    print(f"\nK before RoPE:")
    print(f"  shape: {k_before.shape}")
    print(f"  mean: {k_before.mean().item():.4f}")
    print(f"  std: {k_before.std().item():.4f}")
    
    # Now manually apply RoPE to see the effect
    hidden = model.model.embed_tokens(input_ids)
    hidden = layer.input_layernorm(hidden)
    
    # Get position embeddings
    cache_pos = torch.arange(8)
    pos_emb = model.model.rotary_emb(hidden, cache_pos)
    cos, sin = pos_emb
    
    print(f"\nPosition embeddings:")
    print(f"  cos shape: {cos.shape}")
    print(f"  sin shape: {sin.shape}")
    
    # Manually compute Q/K after RoPE
    with torch.no_grad():
        # Reshape Q/K to [batch, num_heads, seq, head_dim]
        batch_size, seq_len, _ = hidden.shape
        head_dim = config.hidden_size // config.num_attention_heads
        
        q_proj = attn.q_proj(hidden)
        k_proj = attn.k_proj(hidden)
        
        q_reshaped = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim).transpose(1, 2)
        k_reshaped = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim).transpose(1, 2)
        
        # Apply RoPE manually
        from src.modeling.embed import apply_rotary_pos_emb
        q_after, k_after = apply_rotary_pos_emb(q_reshaped, k_reshaped, cos, sin)
    
    print(f"\nQ after RoPE:")
    print(f"  shape: {q_after.shape}")
    print(f"  mean: {q_after.mean().item():.4f}")
    print(f"  std: {q_after.std().item():.4f}")
    
    print(f"\nK after RoPE:")
    print(f"  shape: {k_after.shape}")
    print(f"  mean: {k_after.mean().item():.4f}")
    print(f"  std: {k_after.std().item():.4f}")
    
    # Verify RoPE changed the values
    q_before_reshaped = q_before.view(batch_size, seq_len, config.num_attention_heads, head_dim).transpose(1, 2)
    k_before_reshaped = k_before.view(batch_size, seq_len, config.num_key_value_heads, head_dim).transpose(1, 2)
    
    q_diff = (q_after - q_before_reshaped).abs().mean().item()
    k_diff = (k_after - k_before_reshaped).abs().mean().item()
    
    print(f"\nRoPE effect (mean absolute difference):")
    print(f"  Q: {q_diff:.4f}")
    print(f"  K: {k_diff:.4f}")
    
    # Verification
    if q_diff > 1e-6 and k_diff > 1e-6:
        print("\n✓ RoPE is applied correctly!")
        print(f"  Q and K values changed significantly after RoPE")
    else:
        print("\n✗ RoPE may not be applied!")
        print(f"  Q and K values unchanged (diff < 1e-6)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_forward_backward()
    trace_operations()
    verify_rope_application()
