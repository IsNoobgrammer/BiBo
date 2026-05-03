"""
Comprehensive SSMax Forward Pass Analysis
Multiple forward passes, full metrics table.
"""
import math
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.configuration_bibo import BiBoConfig
from src.modeling.attn import BiBoAttention


def compute_metrics(attn_weights):
    """Compute comprehensive attention metrics."""
    eps = 1e-10
    
    # Entropy
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean().item()
    
    # Max weight
    max_weight = attn_weights.max().item()
    
    # Mean weight (excluding masked)
    mean_weight = attn_weights.mean().item()
    
    # Std dev
    std_weight = attn_weights.std().item()
    
    # Sparsity (% weights < 0.01)
    sparsity = (attn_weights < 0.01).float().mean().item() * 100
    
    # Top-k concentration (sum of top 10 weights per query)
    B, H, Q, K = attn_weights.shape
    topk = min(10, K)
    top_weights, _ = torch.topk(attn_weights, topk, dim=-1)
    top_concentration = top_weights.sum(dim=-1).mean().item()
    
    return {
        'entropy': entropy,
        'max': max_weight,
        'mean': mean_weight,
        'std': std_weight,
        'sparsity': sparsity,
        'top10_sum': top_concentration,
    }


def run_forward_pass(attn, x, device, pass_num):
    """Run single forward pass, return output + metrics."""
    B, S, H = x.shape
    
    with torch.no_grad():
        # Compute Q, K
        q = attn.q_norm(attn.q_proj(x).view(B, S, 4, -1)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(B, S, 2, -1)).transpose(1, 2)
        v = attn.v_proj(x).view(B, S, 2, -1).transpose(1, 2)
        
        # Repeat KV for GQA
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        
        # Apply SSMax scaling
        if attn.use_ssmax:
            q = attn._apply_ssmax_query_scaling(q, S)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(64)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H)
        output = attn.o_proj(attn_out)
        
        metrics = compute_metrics(attn_weights)
        
        return output, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("="*100)
    print("COMPREHENSIVE SSMAX FORWARD PASS ANALYSIS")
    print("="*100)
    
    # Create attention modules
    cfg_std = BiBoConfig(use_ssmax=False, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    cfg_ssmax = BiBoConfig(use_ssmax=True, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    
    attn_std = BiBoAttention(cfg_std, 0).to(device)
    attn_fixed = BiBoAttention(cfg_ssmax, 0).to(device)
    attn_buggy = BiBoAttention(cfg_ssmax, 0).to(device)
    
    # Set buggy init
    with torch.no_grad():
        attn_buggy.ssmax_scale.fill_(1.0)
        
        # Copy weights for fair comparison
        attn_fixed.q_proj.weight.copy_(attn_std.q_proj.weight)
        attn_fixed.k_proj.weight.copy_(attn_std.k_proj.weight)
        attn_fixed.v_proj.weight.copy_(attn_std.v_proj.weight)
        attn_fixed.o_proj.weight.copy_(attn_std.o_proj.weight)
        
        attn_buggy.q_proj.weight.copy_(attn_std.q_proj.weight)
        attn_buggy.k_proj.weight.copy_(attn_std.k_proj.weight)
        attn_buggy.v_proj.weight.copy_(attn_std.v_proj.weight)
        attn_buggy.o_proj.weight.copy_(attn_std.o_proj.weight)
    
    print(f"\nSSMax Init Values:")
    print(f"  Standard:  N/A (no SSMax)")
    print(f"  Fixed:     {attn_fixed.ssmax_scale.mean().item():.6f}")
    print(f"  Buggy:     {attn_buggy.ssmax_scale.mean().item():.6f}")
    
    # Run 5 forward passes
    B, S, H = 2, 512, 256
    num_passes = 5
    
    results = {
        'standard': [],
        'fixed': [],
        'buggy': [],
    }
    
    print(f"\nRunning {num_passes} forward passes (B={B}, S={S}, H={H})...")
    
    for i in range(num_passes):
        torch.manual_seed(42 + i)
        x = torch.randn(B, S, H, device=device)
        
        out_std, metrics_std = run_forward_pass(attn_std, x, device, i)
        out_fixed, metrics_fixed = run_forward_pass(attn_fixed, x, device, i)
        out_buggy, metrics_buggy = run_forward_pass(attn_buggy, x, device, i)
        
        results['standard'].append({
            'output': out_std,
            'metrics': metrics_std,
        })
        results['fixed'].append({
            'output': out_fixed,
            'metrics': metrics_fixed,
        })
        results['buggy'].append({
            'output': out_buggy,
            'metrics': metrics_buggy,
        })
    
    # Print comprehensive table
    print("\n" + "="*100)
    print("ATTENTION METRICS COMPARISON (Averaged over 5 passes)")
    print("="*100)
    
    # Compute averages
    avg_metrics = {}
    for variant in ['standard', 'fixed', 'buggy']:
        avg_metrics[variant] = {}
        for key in results[variant][0]['metrics'].keys():
            avg_metrics[variant][key] = sum(r['metrics'][key] for r in results[variant]) / num_passes
    
    # Print table
    print(f"\n{'Metric':<20} {'Standard':<15} {'Fixed SSMax':<15} {'Buggy SSMax':<15} {'Fixed vs Std':<15} {'Buggy vs Std':<15}")
    print("-" * 100)
    
    for key in ['entropy', 'max', 'mean', 'std', 'sparsity', 'top10_sum']:
        std_val = avg_metrics['standard'][key]
        fixed_val = avg_metrics['fixed'][key]
        buggy_val = avg_metrics['buggy'][key]
        
        fixed_diff = ((fixed_val - std_val) / std_val * 100) if std_val != 0 else 0
        buggy_diff = ((buggy_val - std_val) / std_val * 100) if std_val != 0 else 0
        
        print(f"{key:<20} {std_val:<15.6f} {fixed_val:<15.6f} {buggy_val:<15.6f} {fixed_diff:>+14.2f}% {buggy_diff:>+14.2f}%")
    
    # Output comparison
    print("\n" + "="*100)
    print("OUTPUT COMPARISON")
    print("="*100)
    
    print(f"\n{'Pass':<8} {'Fixed vs Std':<20} {'Buggy vs Std':<20} {'Fixed vs Buggy':<20}")
    print("-" * 70)
    
    for i in range(num_passes):
        out_std = results['standard'][i]['output']
        out_fixed = results['fixed'][i]['output']
        out_buggy = results['buggy'][i]['output']
        
        diff_fixed_std = (out_fixed - out_std).abs().mean().item()
        diff_buggy_std = (out_buggy - out_std).abs().mean().item()
        diff_fixed_buggy = (out_fixed - out_buggy).abs().mean().item()
        
        print(f"{i+1:<8} {diff_fixed_std:<20.6f} {diff_buggy_std:<20.6f} {diff_fixed_buggy:<20.6f}")
    
    # Compute average output diffs
    avg_diff_fixed_std = sum((results['fixed'][i]['output'] - results['standard'][i]['output']).abs().mean().item() for i in range(num_passes)) / num_passes
    avg_diff_buggy_std = sum((results['buggy'][i]['output'] - results['standard'][i]['output']).abs().mean().item() for i in range(num_passes)) / num_passes
    avg_diff_fixed_buggy = sum((results['fixed'][i]['output'] - results['buggy'][i]['output']).abs().mean().item() for i in range(num_passes)) / num_passes
    
    print("-" * 70)
    print(f"{'Average':<8} {avg_diff_fixed_std:<20.6f} {avg_diff_buggy_std:<20.6f} {avg_diff_fixed_buggy:<20.6f}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print(f"\n1. ENTROPY (higher = more distributed attention):")
    print(f"   Standard:  {avg_metrics['standard']['entropy']:.4f}")
    print(f"   Fixed:     {avg_metrics['fixed']['entropy']:.4f}  ({((avg_metrics['fixed']['entropy'] - avg_metrics['standard']['entropy']) / avg_metrics['standard']['entropy'] * 100):+.1f}%)")
    print(f"   Buggy:     {avg_metrics['buggy']['entropy']:.4f}  ({((avg_metrics['buggy']['entropy'] - avg_metrics['standard']['entropy']) / avg_metrics['standard']['entropy'] * 100):+.1f}%) ← COLLAPSED")
    
    print(f"\n2. MAX ATTENTION WEIGHT (lower = less peaked):")
    print(f"   Standard:  {avg_metrics['standard']['max']:.6f}")
    print(f"   Fixed:     {avg_metrics['fixed']['max']:.6f}")
    print(f"   Buggy:     {avg_metrics['buggy']['max']:.6f}")
    
    print(f"\n3. OUTPUT DIFFERENCE FROM STANDARD:")
    print(f"   Fixed:     {avg_diff_fixed_std:.6f}  (close to standard ✓)")
    print(f"   Buggy:     {avg_diff_buggy_std:.6f}  (different from standard)")
    
    print(f"\n4. SPARSITY (% weights < 0.01):")
    print(f"   Standard:  {avg_metrics['standard']['sparsity']:.2f}%")
    print(f"   Fixed:     {avg_metrics['fixed']['sparsity']:.2f}%")
    print(f"   Buggy:     {avg_metrics['buggy']['sparsity']:.2f}%")
    
    # Validation
    print("\n" + "="*100)
    print("VALIDATION")
    print("="*100)
    
    checks = []
    
    # Check 1: Fixed entropy close to standard
    entropy_diff_pct = abs(avg_metrics['fixed']['entropy'] - avg_metrics['standard']['entropy']) / avg_metrics['standard']['entropy'] * 100
    check1 = entropy_diff_pct < 30
    checks.append(check1)
    print(f"\n✓ Fixed entropy within 30% of standard: {entropy_diff_pct:.1f}% {'✓ PASS' if check1 else '✗ FAIL'}")
    
    # Check 2: Buggy entropy much lower
    check2 = avg_metrics['buggy']['entropy'] < avg_metrics['standard']['entropy'] * 0.5
    checks.append(check2)
    print(f"✓ Buggy entropy < 50% of standard: {avg_metrics['buggy']['entropy']:.4f} < {avg_metrics['standard']['entropy'] * 0.5:.4f} {'✓ PASS' if check2 else '✗ FAIL'}")
    
    # Check 3: Fixed output close to standard
    check3 = avg_diff_fixed_std < 0.1
    checks.append(check3)
    print(f"✓ Fixed output diff < 0.1: {avg_diff_fixed_std:.6f} {'✓ PASS' if check3 else '✗ FAIL'}")
    
    # Check 4: Buggy different from fixed
    check4 = avg_diff_fixed_buggy > 0.01
    checks.append(check4)
    print(f"✓ Fixed ≠ Buggy (diff > 0.01): {avg_diff_fixed_buggy:.6f} {'✓ PASS' if check4 else '✗ FAIL'}")
    
    print("\n" + "="*100)
    if all(checks):
        print("✅ ALL CHECKS PASSED - SSMax fix verified!")
    else:
        print(f"❌ {sum(not c for c in checks)}/{len(checks)} checks failed")
    print("="*100 + "\n")
    
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
