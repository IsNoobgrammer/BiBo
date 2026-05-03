"""Test RoPE implementation"""
import sys
sys.path.insert(0, '.')
import torch
from src.modeling.embed import BiBoRotaryEmbedding, apply_rotary_pos_emb

def test_rope_basic():
    """Test basic RoPE functionality"""
    head_dim = 16
    max_pos = 128
    rope = BiBoRotaryEmbedding(head_dim, max_position_embeddings=max_pos)
    
    # Test forward
    x = torch.randn(2, 8, head_dim)
    cos, sin = rope(x, seq_len=8)
    
    print(f"✓ RoPE init: head_dim={head_dim}, max_pos={max_pos}")
    print(f"✓ cos shape: {cos.shape}, sin shape: {sin.shape}")
    assert cos.shape == (8, head_dim), f"Expected (8, {head_dim}), got {cos.shape}"
    assert sin.shape == (8, head_dim), f"Expected (8, {head_dim}), got {sin.shape}"

def test_rope_cache_expansion():
    """Test cache expansion for longer sequences"""
    head_dim = 16
    max_pos = 32
    rope = BiBoRotaryEmbedding(head_dim, max_position_embeddings=max_pos)
    
    # Request longer sequence
    x = torch.randn(2, 64, head_dim)
    cos, sin = rope(x, seq_len=64)
    
    print(f"✓ Cache expansion: requested 64 > max 32")
    print(f"✓ New cache size: {rope.max_seq_len_cached}")
    assert rope.max_seq_len_cached >= 64
    assert cos.shape == (64, head_dim)

def test_rope_with_tensor_seq_len():
    """Test with tensor seq_len (cache_position)"""
    head_dim = 16
    rope = BiBoRotaryEmbedding(head_dim, max_position_embeddings=128)
    
    # Simulate cache_position tensor [0, 1, 2, 3, 4, 5, 6, 7]
    cache_position = torch.arange(8)
    x = torch.randn(2, 8, head_dim)
    cos, sin = rope(x, seq_len=cache_position)
    
    print(f"✓ Tensor seq_len: cache_position max={cache_position.max().item()}")
    print(f"✓ cos shape: {cos.shape}")
    # Should return embeddings for positions 0..8 (9 total)
    assert cos.shape[0] == cache_position.max().item() + 1

def test_apply_rope():
    """Test apply_rotary_pos_emb"""
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    
    rope = BiBoRotaryEmbedding(head_dim, max_position_embeddings=128)
    cos, sin = rope(q, seq_len=seq_len)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"✓ apply_rotary_pos_emb:")
    print(f"  q: {q.shape} → {q_rot.shape}")
    print(f"  k: {k.shape} → {k_rot.shape}")
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    
    # Check rotation actually changed values
    assert not torch.allclose(q, q_rot), "RoPE should modify query"
    assert not torch.allclose(k, k_rot), "RoPE should modify key"

def test_rope_position_encoding():
    """Test that different positions get different encodings"""
    head_dim = 16
    rope = BiBoRotaryEmbedding(head_dim, max_position_embeddings=128)
    
    x = torch.randn(1, 1, head_dim)
    
    # Get embeddings for position 0 and position 10
    cos_0, sin_0 = rope(x, seq_len=1)
    cos_10, sin_10 = rope(x, seq_len=11)
    
    # Position 0 vs position 10 should be different
    pos_0 = cos_0[0]
    pos_10 = cos_10[10]
    
    print(f"✓ Position encoding:")
    print(f"  pos 0 norm: {pos_0.norm().item():.4f}")
    print(f"  pos 10 norm: {pos_10.norm().item():.4f}")
    print(f"  difference: {(pos_0 - pos_10).norm().item():.4f}")
    
    assert not torch.allclose(pos_0, pos_10), "Different positions should have different encodings"

def test_rope_in_attention():
    """Test RoPE in full attention context"""
    from src.modeling.attn import BiBoAttention
    from src.configuration_bibo import BiBoConfig
    
    cfg = BiBoConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    
    attn = BiBoAttention(cfg, layer_idx=0)
    x = torch.randn(2, 8, cfg.hidden_size)
    
    # Get position embeddings
    cos, sin = attn.rotary_emb(x, seq_len=8)
    
    print(f"✓ RoPE in attention:")
    print(f"  hidden_size: {cfg.hidden_size}")
    print(f"  head_dim: {attn.head_dim}")
    print(f"  cos/sin shape: {cos.shape}")
    
    # Forward pass
    out, _, _ = attn(x, (cos, sin), None)
    assert out.shape == x.shape
    print(f"✓ Attention with RoPE: {x.shape} → {out.shape}")

if __name__ == "__main__":
    print("Testing RoPE implementation...\n")
    
    test_rope_basic()
    print()
    
    test_rope_cache_expansion()
    print()
    
    test_rope_with_tensor_seq_len()
    print()
    
    test_apply_rope()
    print()
    
    test_rope_position_encoding()
    print()
    
    test_rope_in_attention()
    print()
    
    print("✅ All RoPE tests passed!")
