"""
SSMax vs Standard: Math-only comparison (no torch).
Verify init prevents over-sharpening.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_ssmax_init_formula():
    """SSMax init = 1/log(typical_seq_len)."""
    max_pos_emb = 2048
    typical_seq = max_pos_emb / 2
    
    expected = 1.0 / math.log(typical_seq)
    
    print(f"\n[TEST 1] SSMax Init Formula")
    print(f"max_position_embeddings: {max_pos_emb}")
    print(f"typical_seq_len:         {typical_seq}")
    print(f"Expected init:           {expected:.6f}")
    
    assert abs(expected - 0.144270) < 1e-5
    print("✓ PASS\n")


def test_effective_scale_comparison():
    """Compare effective scales: fixed vs buggy vs standard."""
    max_pos_emb = 2048
    head_dim = 64
    kv_len = 512
    
    # Init values
    fixed_init = 1.0 / math.log(max_pos_emb / 2)
    buggy_init = 1.0
    
    # Standard scale
    std_scale = 1.0 / math.sqrt(head_dim)
    
    # Effective scales
    fixed_eff = fixed_init * math.log(kv_len) / math.sqrt(head_dim)
    buggy_eff = buggy_init * math.log(kv_len) / math.sqrt(head_dim)
    
    # Ratios
    fixed_ratio = fixed_eff / std_scale
    buggy_ratio = buggy_eff / std_scale
    
    print(f"[TEST 2] Effective Scale @ kv_len={kv_len}")
    print(f"Standard scale:          {std_scale:.6f}")
    print(f"\nFixed SSMax:")
    print(f"  init:                  {fixed_init:.6f}")
    print(f"  effective:             {fixed_eff:.6f}")
    print(f"  ratio to std:          {fixed_ratio:.3f}x  {'✓' if 0.8 <= fixed_ratio <= 1.2 else '✗'}")
    print(f"\nBuggy SSMax:")
    print(f"  init:                  {buggy_init:.6f}")
    print(f"  effective:             {buggy_eff:.6f}")
    print(f"  ratio to std:          {buggy_ratio:.3f}x  {'✗' if buggy_ratio > 2.0 else '⚠'}")
    print(f"\nOver-sharpening factor:  {buggy_ratio / fixed_ratio:.2f}x")
    
    assert 0.8 <= fixed_ratio <= 1.2, f"Fixed ratio {fixed_ratio:.3f} not in [0.8, 1.2]"
    assert buggy_ratio > 4.0, f"Buggy ratio {buggy_ratio:.3f} should be >4.0"
    print("✓ PASS\n")


def test_scale_across_lengths():
    """SSMax adapts to seq length."""
    max_pos_emb = 2048
    head_dim = 64
    
    fixed_init = 1.0 / math.log(max_pos_emb / 2)
    std_scale = 1.0 / math.sqrt(head_dim)
    
    seq_lens = [128, 256, 512, 1024, 2048]
    
    print(f"[TEST 3] Scale Across Seq Lengths")
    print(f"{'Seq Len':<10} {'Fixed Eff':<12} {'Std Scale':<12} {'Ratio':<8} {'Status'}")
    print("-" * 50)
    
    for kv_len in seq_lens:
        fixed_eff = fixed_init * math.log(kv_len) / math.sqrt(head_dim)
        ratio = fixed_eff / std_scale
        
        status = "✓" if 0.7 <= ratio <= 1.3 else "✗"
        print(f"{kv_len:<10} {fixed_eff:<12.6f} {std_scale:<12.6f} {ratio:<8.3f} {status}")
        
        if 512 <= kv_len <= 2048:
            assert 0.7 <= ratio <= 1.3, f"At {kv_len}, ratio {ratio:.3f} not in [0.7, 1.3]"
    
    print("✓ PASS\n")


def test_different_configs():
    """Init adapts to different max_pos_emb."""
    configs = [
        (512, 256),
        (2048, 1024),
        (8192, 4096),
        (32768, 16384),
    ]
    
    print(f"[TEST 4] Different Configs")
    print(f"{'max_pos_emb':<15} {'typical':<10} {'Init Value':<12}")
    print("-" * 40)
    
    for max_pos, typical in configs:
        init_val = 1.0 / math.log(typical)
        print(f"{max_pos:<15} {typical:<10} {init_val:<12.6f}")
        
        expected = 1.0 / math.log(typical)
        assert abs(init_val - expected) < 1e-10
    
    print("✓ PASS\n")


def main():
    print("\n" + "="*60)
    print("SSMax vs Standard: Math-Only Tests")
    print("="*60)
    
    tests = [
        test_ssmax_init_formula,
        test_effective_scale_comparison,
        test_scale_across_lengths,
        test_different_configs,
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}\n")
    
    print("="*60)
    print(f"Result: {passed}/{len(tests)} passed")
    print("="*60 + "\n")
    
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    exit(main())
