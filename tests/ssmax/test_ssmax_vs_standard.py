"""
SSMax vs Standard Attention: Scale Comparison Tests
Verify SSMax init prevents over-sharpening vs standard attention baseline.
"""
import math
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoAttention


class TestSSMaxVsStandard:
    """Compare SSMax scaling to standard attention baseline."""
    
    @pytest.fixture
    def configs(self):
        """Create configs for standard and SSMax attention."""
        base_cfg = dict(
            max_position_embeddings=2048,
            num_attention_heads=4,
            hidden_size=256,
            num_key_value_heads=2,
            attention_dropout=0.0,
        )
        
        cfg_standard = BiBoConfig(use_ssmax=False, **base_cfg)
        cfg_ssmax = BiBoConfig(use_ssmax=True, **base_cfg)
        
        return cfg_standard, cfg_ssmax
    
    def test_ssmax_init_value(self, configs):
        """SSMax scale = 1/log(typical_seq_len)."""
        _, cfg_ssmax = configs
        attn = BiBoAttention(cfg_ssmax, layer_idx=0)
        
        expected = 1.0 / math.log(cfg_ssmax.max_position_embeddings / 2)
        actual = attn.ssmax_scale.mean().item()
        
        assert abs(actual - expected) < 1e-6
    
    def test_effective_scale_at_init(self, configs):
        """SSMax effective scale ≈ standard at typical seq_len."""
        cfg_std, cfg_ssmax = configs
        attn_ssmax = BiBoAttention(cfg_ssmax, layer_idx=0)
        
        head_dim = cfg_std.hidden_size // cfg_std.num_attention_heads
        kv_len = 512  # typical early training
        
        # Standard scale
        std_scale = 1.0 / math.sqrt(head_dim)
        
        # SSMax effective scale
        ssmax_init = attn_ssmax.ssmax_scale.mean().item()
        ssmax_scale = ssmax_init * math.log(kv_len) / math.sqrt(head_dim)
        
        ratio = ssmax_scale / std_scale
        
        # Should be close to 1.0 (within 20%)
        assert 0.8 <= ratio <= 1.2, f"Ratio {ratio:.3f} not in [0.8, 1.2]"
    
    def test_buggy_init_oversharpens(self, configs):
        """Buggy init (scale=1.0) causes >4x over-sharpening."""
        cfg_std, _ = configs
        
        head_dim = cfg_std.hidden_size // cfg_std.num_attention_heads
        kv_len = 512
        
        std_scale = 1.0 / math.sqrt(head_dim)
        buggy_scale = 1.0 * math.log(kv_len) / math.sqrt(head_dim)
        
        ratio = buggy_scale / std_scale
        
        assert ratio > 4.0, f"Buggy ratio {ratio:.3f} should be >4.0"
    
    def test_attention_weights_comparison(self, configs):
        """Compare actual attention weight distributions."""
        cfg_std, cfg_ssmax = configs
        
        attn_std = BiBoAttention(cfg_std, layer_idx=0)
        attn_ssmax = BiBoAttention(cfg_ssmax, layer_idx=0)
        
        # Copy weights for fair comparison
        with torch.no_grad():
            attn_ssmax.q_proj.weight.copy_(attn_std.q_proj.weight)
            attn_ssmax.k_proj.weight.copy_(attn_std.k_proj.weight)
            attn_ssmax.v_proj.weight.copy_(attn_std.v_proj.weight)
        
        batch, seq_len, hidden = 2, 512, cfg_std.hidden_size
        head_dim = hidden // cfg_std.num_attention_heads
        
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, hidden)
        
        # Compute attention weights
        def get_attn_weights(attn_module):
            with torch.no_grad():
                q = attn_module.q_norm(
                    attn_module.q_proj(x).view(batch, seq_len, cfg_std.num_attention_heads, -1)
                ).transpose(1, 2)
                
                k = attn_module.k_norm(
                    attn_module.k_proj(x).view(batch, seq_len, cfg_std.num_key_value_heads, -1)
                ).transpose(1, 2)
                
                # Repeat k for GQA
                from modeling_bibo import repeat_kv
                k = repeat_kv(k, cfg_std.num_attention_heads // cfg_std.num_key_value_heads)
                
                if attn_module.use_ssmax:
                    q = attn_module._apply_ssmax_query_scaling(q, seq_len)
                
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
                
                # Causal mask
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
                
                return torch.softmax(scores, dim=-1)
        
        weights_std = get_attn_weights(attn_std)
        weights_ssmax = get_attn_weights(attn_ssmax)
        
        # Compute entropy
        def entropy(w):
            eps = 1e-10
            return -(w * torch.log(w + eps)).sum(dim=-1).mean().item()
        
        ent_std = entropy(weights_std)
        ent_ssmax = entropy(weights_ssmax)
        
        # Entropies should be similar (within 30%)
        rel_diff = abs(ent_std - ent_ssmax) / ent_std
        assert rel_diff < 0.3, f"Entropy diff {rel_diff*100:.1f}% > 30%"
        
        # Both should be healthy (>2.0)
        assert ent_std > 2.0, f"Standard entropy {ent_std:.2f} < 2.0"
        assert ent_ssmax > 2.0, f"SSMax entropy {ent_ssmax:.2f} < 2.0"
    
    def test_buggy_ssmax_collapses_entropy(self, configs):
        """Buggy SSMax (scale=1.0) causes entropy collapse."""
        _, cfg_ssmax = configs
        attn = BiBoAttention(cfg_ssmax, layer_idx=0)
        
        # Override to buggy init
        with torch.no_grad():
            attn.ssmax_scale.fill_(1.0)
        
        batch, seq_len, hidden = 2, 512, cfg_ssmax.hidden_size
        head_dim = hidden // cfg_ssmax.num_attention_heads
        
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, hidden)
        
        with torch.no_grad():
            q = attn.q_norm(attn.q_proj(x).view(batch, seq_len, cfg_ssmax.num_attention_heads, -1)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).view(batch, seq_len, cfg_ssmax.num_key_value_heads, -1)).transpose(1, 2)
            
            # Repeat k for GQA
            from modeling_bibo import repeat_kv
            k = repeat_kv(k, cfg_ssmax.num_attention_heads // cfg_ssmax.num_key_value_heads)
            
            q = attn._apply_ssmax_query_scaling(q, seq_len)
            
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            
            weights = torch.softmax(scores, dim=-1)
        
        # Entropy should be lower (more peaked)
        eps = 1e-10
        ent = -(weights * torch.log(weights + eps)).sum(dim=-1).mean().item()
        
        # Max weight should be high
        max_w = weights.max().item()
        
        # Buggy should cause lower entropy and higher max weight
        assert ent < 3.5, f"Buggy entropy {ent:.2f} should be <3.5"
        assert max_w > 0.3, f"Buggy max weight {max_w:.3f} should be >0.3"
    
    def test_scale_across_seq_lengths(self, configs):
        """SSMax adapts to different seq lengths."""
        _, cfg_ssmax = configs
        attn = BiBoAttention(cfg_ssmax, layer_idx=0)
        
        head_dim = cfg_ssmax.hidden_size // cfg_ssmax.num_attention_heads
        std_scale = 1.0 / math.sqrt(head_dim)
        ssmax_init = attn.ssmax_scale.mean().item()
        
        seq_lens = [128, 256, 512, 1024, 2048]
        
        for kv_len in seq_lens:
            ssmax_eff = ssmax_init * math.log(kv_len) / math.sqrt(head_dim)
            ratio = ssmax_eff / std_scale
            
            # At typical training lengths (512-2048), should be close
            if 512 <= kv_len <= 2048:
                assert 0.7 <= ratio <= 1.3, f"At {kv_len}, ratio {ratio:.3f} not in [0.7, 1.3]"


class TestSSMaxScaling:
    """Test SSMax query scaling function."""
    
    def test_scaling_disabled(self):
        """No scaling when use_ssmax=False."""
        cfg = BiBoConfig(use_ssmax=False, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        attn = BiBoAttention(cfg, layer_idx=0)
        
        q = torch.randn(2, 4, 512, 64)
        scaled = attn._apply_ssmax_query_scaling(q, 512)
        
        assert torch.allclose(scaled, q)
    
    def test_scaling_enabled(self):
        """Scaling applied when use_ssmax=True."""
        cfg = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        attn = BiBoAttention(cfg, layer_idx=0)
        
        q = torch.randn(2, 4, 512, 64)
        kv_len = 512
        
        scaled = attn._apply_ssmax_query_scaling(q, kv_len)
        
        expected_scale = attn.ssmax_scale * math.log(kv_len)
        expected = q * expected_scale
        
        assert torch.allclose(scaled, expected, rtol=1e-5)
    
    def test_scaling_adapts_to_length(self):
        """Scale factor increases with seq length."""
        cfg = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        attn = BiBoAttention(cfg, layer_idx=0)
        
        q = torch.randn(2, 4, 128, 64)
        
        scale_128 = (attn._apply_ssmax_query_scaling(q, 128) / q).mean().item()
        scale_512 = (attn._apply_ssmax_query_scaling(q, 512) / q).mean().item()
        scale_2048 = (attn._apply_ssmax_query_scaling(q, 2048) / q).mean().item()
        
        # Scale should increase with seq length
        assert scale_128 < scale_512 < scale_2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
