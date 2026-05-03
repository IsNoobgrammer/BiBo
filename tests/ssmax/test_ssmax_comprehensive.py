"""
Comprehensive SSMax tests - modular architecture compatible
Tests SSMax scaling, initialization, and attention behavior
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import pytest
from src.configuration_bibo import BiBoConfig
from src.modeling.attn import BiBoAttention
from src.modeling.attn.ssmax import apply_ssmax_query_scaling


class TestSSMaxInitialization:
    """Test SSMax scale parameter initialization"""
    
    def test_ssmax_scale_init_formula(self):
        """SSMax scale initialized to 1/log(max_pos_emb/2)"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        expected = 1.0 / math.log(2048 / 2)  # ≈ 0.145
        actual = attn.ssmax_scale.mean().item()  # Per-head scale, take mean
        
        assert abs(actual - expected) < 0.01, f"Expected {expected:.3f}, got {actual:.3f}"
        print(f"✓ SSMax scale init: {actual:.3f} (expected {expected:.3f})")
    
    def test_effective_scale_at_init(self):
        """Effective scale ≈ 1.0 at typical seq_len"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        # At seq_len=1024 (typical), effective scale should be ≈ 1.0
        seq_len = 1024
        scale = attn.ssmax_scale.mean().item()
        effective = scale * math.log(seq_len)
        
        assert 0.9 < effective < 1.1, f"Effective scale {effective:.3f} not ≈ 1.0"
        print(f"✓ Effective scale at seq_len={seq_len}: {effective:.3f}")
    
    def test_buggy_init_oversharpens(self):
        """Old init (scale=1.0) causes oversharpening"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=256,
            num_attention_heads=4,
            max_position_embeddings=2048,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        # Override to buggy init
        with torch.no_grad():
            attn.ssmax_scale.fill_(1.0)
        
        seq_len = 2048
        effective = attn.ssmax_scale.mean().item() * math.log(seq_len)
        
        # Buggy init: effective scale ≈ 7.6 (way too high)
        assert effective > 5.0, f"Buggy init should give high scale, got {effective:.3f}"
        print(f"✓ Buggy init effective scale: {effective:.3f} (oversharpens)")


class TestSSMaxScaling:
    """Test SSMax query scaling function"""
    
    def test_scaling_disabled_when_use_ssmax_false(self):
        """No scaling when use_ssmax=False"""
        cfg = BiBoConfig(
            use_ssmax=False,
            hidden_size=256,
            num_attention_heads=4,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        q = torch.randn(2, 4, 128, 64)
        q_scaled = attn._apply_ssmax_query_scaling(q, kv_len=128)
        
        assert torch.allclose(q, q_scaled), "Query should be unchanged when use_ssmax=False"
        print("✓ Scaling disabled when use_ssmax=False")
    
    def test_scaling_enabled_when_use_ssmax_true(self):
        """Scaling applied when use_ssmax=True"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=256,
            num_attention_heads=4,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        q = torch.randn(2, 4, 128, 64)
        q_scaled = attn._apply_ssmax_query_scaling(q, kv_len=128)
        
        assert not torch.allclose(q, q_scaled), "Query should be scaled when use_ssmax=True"
        
        # Check scaling magnitude
        scale_factor = (q_scaled / q).mean().item()
        expected_factor = attn.ssmax_scale.mean().item() * math.log(128)
        
        assert abs(scale_factor - expected_factor) < 0.1, f"Scale factor {scale_factor:.3f} ≠ {expected_factor:.3f}"
        print(f"✓ Scaling enabled: factor={scale_factor:.3f}")
    
    def test_scaling_adapts_to_length(self):
        """Scaling increases with sequence length"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=256,
            num_attention_heads=4,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        q = torch.randn(2, 4, 8, 64)
        
        scales = {}
        for seq_len in [128, 512, 2048]:
            q_scaled = attn._apply_ssmax_query_scaling(q, kv_len=seq_len)
            scales[seq_len] = (q_scaled / q).mean().item()
        
        # Longer sequences → higher scaling
        assert scales[512] > scales[128], "Scale should increase with seq_len"
        assert scales[2048] > scales[512], "Scale should increase with seq_len"
        
        print(f"✓ Adaptive scaling: 128→{scales[128]:.2f}, 512→{scales[512]:.2f}, 2048→{scales[2048]:.2f}")


class TestSSMaxAttentionBehavior:
    """Test SSMax in full attention context"""
    
    def test_forward_pass_with_ssmax(self):
        """SSMax attention forward pass works"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, cfg.hidden_size)
        cos, sin = attn.rotary_emb(x, seq_len=seq_len)
        
        out, _, _ = attn(x, (cos, sin), None)
        
        assert out.shape == x.shape, f"Output shape {out.shape} ≠ input {x.shape}"
        assert torch.isfinite(out).all(), "Output contains NaN/Inf"
        print(f"✓ Forward pass: {x.shape} → {out.shape}")
    
    def test_ssmax_vs_standard_output_similar_at_init(self):
        """SSMax ≈ standard at init (scale ≈ 1.0)"""
        cfg_std = BiBoConfig(
            use_ssmax=False,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        cfg_ssmax = BiBoConfig(
            use_ssmax=True,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        
        attn_std = BiBoAttention(cfg_std, layer_idx=0)
        attn_ssmax = BiBoAttention(cfg_ssmax, layer_idx=0)
        
        # Copy weights
        with torch.no_grad():
            attn_ssmax.q_proj.weight.copy_(attn_std.q_proj.weight)
            attn_ssmax.k_proj.weight.copy_(attn_std.k_proj.weight)
            attn_ssmax.v_proj.weight.copy_(attn_std.v_proj.weight)
            attn_ssmax.o_proj.weight.copy_(attn_std.o_proj.weight)
        
        torch.manual_seed(42)
        x = torch.randn(2, 16, cfg_std.hidden_size)
        
        cos, sin = attn_std.rotary_emb(x, seq_len=16)
        
        with torch.no_grad():
            out_std, _, _ = attn_std(x, (cos, sin), None)
            out_ssmax, _, _ = attn_ssmax(x, (cos, sin), None)
        
        diff = (out_std - out_ssmax).abs().mean().item()
        
        # At init, outputs should be similar (scale ≈ 1.0)
        assert diff < 0.1, f"Outputs differ by {diff:.4f} (should be < 0.1 at init)"
        print(f"✓ SSMax ≈ standard at init: diff={diff:.4f}")
    
    def test_ssmax_prevents_entropy_collapse_at_long_seq(self):
        """SSMax maintains entropy at long sequences"""
        cfg = BiBoConfig(
            use_ssmax=True,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        )
        attn = BiBoAttention(cfg, layer_idx=0)
        
        # Long sequence
        batch, seq_len = 1, 1024
        x = torch.randn(batch, seq_len, cfg.hidden_size)
        cos, sin = attn.rotary_emb(x, seq_len=seq_len)
        
        with torch.no_grad():
            out, _, _ = attn(x, (cos, sin), None)
        
        # Check output is not collapsed (all same value)
        out_std = out.std().item()
        assert out_std > 0.01, f"Output collapsed: std={out_std:.6f}"
        print(f"✓ No collapse at seq_len={seq_len}: output_std={out_std:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
