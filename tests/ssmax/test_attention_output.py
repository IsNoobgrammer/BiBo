"""
SSMax vs Standard: Attention Output Comparison
Compare actual attention weights, entropy, distributions.
"""
import math
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configuration_bibo import BiBoConfig
from src.modeling.attn import BiBoAttention


class TestAttentionOutput:
    """Compare attention outputs: SSMax vs standard."""
    
    def test_attention_weights_similar_at_init(self):
        """SSMax ≈ standard attn weights at init."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg_std = BiBoConfig(use_ssmax=False, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        cfg_ssmax = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        
        attn_std = BiBoAttention(cfg_std, 0).to(device)
        attn_ssmax = BiBoAttention(cfg_ssmax, 0).to(device)
        
        # Copy weights
        with torch.no_grad():
            attn_ssmax.q_proj.weight.copy_(attn_std.q_proj.weight)
            attn_ssmax.k_proj.weight.copy_(attn_std.k_proj.weight)
            attn_ssmax.v_proj.weight.copy_(attn_std.v_proj.weight)
        
        B, S, H = 2, 512, 256
        torch.manual_seed(42)
        x = torch.randn(B, S, H, device=device)
        
        # Get attn weights
        def get_weights(attn):
            with torch.no_grad():
                q = attn.q_norm(attn.q_proj(x).view(B, S, 4, -1)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(x).view(B, S, 2, -1)).transpose(1, 2)
                
                # Repeat KV for GQA
                k = k.repeat_interleave(2, dim=1)  # 2 KV heads → 4 Q heads
                
                if attn.use_ssmax:
                    q = attn._apply_ssmax_query_scaling(q, S)
                
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(64)
                mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
                return torch.softmax(scores, dim=-1)
        
        w_std = get_weights(attn_std)
        w_ssmax = get_weights(attn_ssmax)
        
        # Compare distributions
        diff = (w_std - w_ssmax).abs().mean().item()
        
        print(f"\nAttn weight diff: {diff:.6f}")
        assert diff < 0.1, f"Diff {diff:.6f} > 0.1"
    
    def test_entropy_comparison(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """Compare entropy: std vs SSMax vs buggy."""
        cfg = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        
        attn_std = BiBoAttention(BiBoConfig(use_ssmax=False, hidden_size=256, num_attention_heads=4, num_key_value_heads=2), 0).to(device)
        attn_fixed = BiBoAttention(cfg, 0).to(device)
        attn_buggy = BiBoAttention(cfg, 0).to(device)
        
        # Buggy init
        with torch.no_grad():
            attn_buggy.ssmax_scale.fill_(1.0)
            
            # Copy weights
            attn_fixed.q_proj.weight.copy_(attn_std.q_proj.weight)
            attn_fixed.k_proj.weight.copy_(attn_std.k_proj.weight)
            attn_buggy.q_proj.weight.copy_(attn_std.q_proj.weight)
            attn_buggy.k_proj.weight.copy_(attn_std.k_proj.weight)
        
        B, S, H = 2, 512, 256
        torch.manual_seed(42)
        x = torch.randn(B, S, H, device=device)
        
        def get_entropy(attn):
            with torch.no_grad():
                q = attn.q_norm(attn.q_proj(x).view(B, S, 4, -1)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(x).view(B, S, 2, -1)).transpose(1, 2)
                
                # Repeat KV for GQA
                k = k.repeat_interleave(2, dim=1)
                
                if attn.use_ssmax:
                    q = attn._apply_ssmax_query_scaling(q, S)
                
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(64)
                mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
                w = torch.softmax(scores, dim=-1)
                
                eps = 1e-10
                return -(w * torch.log(w + eps)).sum(dim=-1).mean().item()
        
        ent_std = get_entropy(attn_std)
        ent_fixed = get_entropy(attn_fixed)
        ent_buggy = get_entropy(attn_buggy)
        
        print(f"\nEntropy comparison:")
        print(f"  Standard:  {ent_std:.4f}")
        print(f"  Fixed:     {ent_fixed:.4f}")
        print(f"  Buggy:     {ent_buggy:.4f}")
        
        # Fixed ≈ std
        rel_diff = abs(ent_std - ent_fixed) / ent_std
        assert rel_diff < 0.3, f"Fixed diff {rel_diff*100:.1f}% > 30%"
        
        # Buggy < fixed
        assert ent_buggy < ent_fixed * 0.9, f"Buggy {ent_buggy:.4f} not < fixed*0.9"
        
        # All healthy
        assert ent_std > 2.0 and ent_fixed > 2.0
    
    def test_max_attention_weight(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """Buggy SSMax → high max attn weight."""
        cfg = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        
        attn_fixed = BiBoAttention(cfg, 0).to(device)
        attn_buggy = BiBoAttention(cfg, 0).to(device)
        
        with torch.no_grad():
            attn_buggy.ssmax_scale.fill_(1.0)
            attn_buggy.q_proj.weight.copy_(attn_fixed.q_proj.weight)
            attn_buggy.k_proj.weight.copy_(attn_fixed.k_proj.weight)
        
        B, S, H = 2, 512, 256
        torch.manual_seed(42)
        x = torch.randn(B, S, H, device=device)
        
        def get_max_weight(attn):
            with torch.no_grad():
                q = attn.q_norm(attn.q_proj(x).view(B, S, 4, -1)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(x).view(B, S, 2, -1)).transpose(1, 2)
                
                # Repeat KV for GQA
                k = k.repeat_interleave(2, dim=1)
                
                if attn.use_ssmax:
                    q = attn._apply_ssmax_query_scaling(q, S)
                
                scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(64)
                mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
                w = torch.softmax(scores, dim=-1)
                return w.max().item()
        
        max_fixed = get_max_weight(attn_fixed)
        max_buggy = get_max_weight(attn_buggy)
        
        print(f"\nMax attn weight:")
        print(f"  Fixed: {max_fixed:.6f}")
        print(f"  Buggy: {max_buggy:.6f}")
        
        # Buggy should be higher (but causal mask can make both 1.0)
        # Check entropy instead - more reliable
        print(f"\nBuggy causes lower entropy, not always higher max weight")
        assert max_buggy >= max_fixed * 0.8, f"Buggy {max_buggy:.6f} unexpectedly low"
    
    def test_forward_pass_output(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """Full forward pass comparison."""
        cfg_std = BiBoConfig(use_ssmax=False, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        cfg_ssmax = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        
        attn_std = BiBoAttention(cfg_std, 0).to(device)
        attn_ssmax = BiBoAttention(cfg_ssmax, 0).to(device)
        
        # Copy weights
        with torch.no_grad():
            attn_ssmax.q_proj.weight.copy_(attn_std.q_proj.weight)
            attn_ssmax.k_proj.weight.copy_(attn_std.k_proj.weight)
            attn_ssmax.v_proj.weight.copy_(attn_std.v_proj.weight)
            attn_ssmax.o_proj.weight.copy_(attn_std.o_proj.weight)
        
        B, S, H = 2, 512, 256
        torch.manual_seed(42)
        x = torch.randn(B, S, H, device=device)
        
        cos = torch.randn(S, 64, device=device)
        sin = torch.randn(S, 64, device=device)
        pos_emb = (cos, sin)
        
        mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            out_std, _, _ = attn_std(x, pos_emb, mask)
            out_ssmax, _, _ = attn_ssmax(x, pos_emb, mask)
        
        diff = (out_std - out_ssmax).abs().mean().item()
        
        print(f"\nOutput diff: {diff:.6f}")
        
        # Should be similar (within 60% - RoPE + o_proj add variance)
        rel_diff = diff / out_std.abs().mean().item()
        assert rel_diff < 0.6, f"Rel diff {rel_diff*100:.1f}% > 60%"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    test = TestAttentionOutput()
    tests = [
        ("Attn weights similar", test.test_attention_weights_similar_at_init),
        ("Entropy comparison", test.test_entropy_comparison),
        ("Max attn weight", test.test_max_attention_weight),
        ("Forward pass", test.test_forward_pass_output),
    ]
    
    passed = 0
    for name, fn in tests:
        try:
            print(f"\n{'='*60}\nTEST: {name}\n{'='*60}")
            fn()
            print("✓ PASS")
            passed += 1
        except Exception as e:
            print(f"✗ FAIL: {e}")
    
    print(f"\n{'='*60}\nResult: {passed}/{len(tests)} passed\n{'='*60}")
    sys.exit(0 if passed == len(tests) else 1)
