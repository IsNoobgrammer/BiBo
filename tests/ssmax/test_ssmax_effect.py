"""
Test SSMax effect on attention scores.
Compare: standard softmax vs SSMax across seq lengths.
"""
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from pathlib import Path


def compute_attention_scores(query, key, mask, head_dim, use_ssmax=False, ssmax_scale=None, kv_len=None):
    """
    Compute attention scores (before softmax).
    
    Args:
        query: (batch, num_heads, q_len, head_dim)
        key: (batch, num_heads, kv_len, head_dim)
        mask: Causal mask
        head_dim: Head dimension
        use_ssmax: Apply SSMax scaling
        ssmax_scale: SSMax scale param
        kv_len: KV sequence length (for SSMax)
    
    Returns:
        attn_weights: (batch, num_heads, q_len, kv_len)
    """
    if use_ssmax and ssmax_scale is not None:
        # Apply SSMax query scaling
        log_n = torch.log(torch.clamp(torch.tensor(kv_len, device=query.device, dtype=ssmax_scale.dtype), min=2.0))
        query = query * ssmax_scale * log_n
    
    # Attention scores
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
    
    if mask is not None:
        attn_weights = attn_weights + mask
    
    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    return attn_weights


def make_causal_mask(batch_size, seq_len, device, dtype):
    """Causal mask"""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)


def test_ssmax_effect_on_scores():
    """
    Test: How SSMax changes attention distribution
    """
    batch_size = 1
    num_heads = 8
    head_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Test multiple seq lengths
    seq_lens = [16, 32, 64, 128, 256]
    
    # SSMax scale param (typical init: ~0.15)
    ssmax_scale = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.15, device=device, dtype=dtype))
    
    results = {
        'seq_lens': seq_lens,
        'standard_entropy': [],
        'ssmax_entropy': [],
        'standard_max_prob': [],
        'ssmax_max_prob': [],
        'scale_factor': [],
    }
    
    print("="*80)
    print("SSMax Effect on Attention Scores")
    print("="*80)
    print(f"\nSSMax scale param: {ssmax_scale.mean().item():.3f}")
    print(f"Device: {device}\n")
    
    for seq_len in seq_lens:
        torch.manual_seed(42)
        
        # Random Q, K
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        mask = make_causal_mask(batch_size, seq_len, device, dtype)
        
        # Standard softmax
        attn_standard = compute_attention_scores(query, key, mask, head_dim, use_ssmax=False)
        
        # SSMax
        attn_ssmax = compute_attention_scores(query, key, mask, head_dim, use_ssmax=True, 
                                              ssmax_scale=ssmax_scale, kv_len=seq_len)
        
        # Compute entropy (measure of distribution spread)
        # Higher entropy = more uniform, lower = more peaked
        def entropy(probs):
            # Avoid log(0)
            probs = torch.clamp(probs, min=1e-10)
            return -(probs * torch.log(probs)).sum(dim=-1).mean().item()
        
        standard_ent = entropy(attn_standard)
        ssmax_ent = entropy(attn_ssmax)
        
        # Max probability (how peaked is distribution)
        standard_max = attn_standard.max(dim=-1)[0].mean().item()
        ssmax_max = attn_ssmax.max(dim=-1)[0].mean().item()
        
        # Scale factor applied
        log_n = math.log(max(seq_len, 2))
        scale_factor = ssmax_scale.mean().item() * log_n
        
        results['standard_entropy'].append(standard_ent)
        results['ssmax_entropy'].append(ssmax_ent)
        results['standard_max_prob'].append(standard_max)
        results['ssmax_max_prob'].append(ssmax_max)
        results['scale_factor'].append(scale_factor)
        
        print(f"Seq len: {seq_len:3d}")
        print(f"  Scale factor: {scale_factor:.3f}")
        print(f"  Entropy:      standard={standard_ent:.3f}, ssmax={ssmax_ent:.3f}, "
              f"diff={ssmax_ent - standard_ent:+.3f}")
        print(f"  Max prob:     standard={standard_max:.3f}, ssmax={ssmax_max:.3f}, "
              f"diff={ssmax_max - standard_max:+.3f}")
        print()
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Entropy comparison
    ax = axes[0, 0]
    ax.plot(seq_lens, results['standard_entropy'], 'o-', label='Standard', linewidth=2)
    ax.plot(seq_lens, results['ssmax_entropy'], 's-', label='SSMax', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Distribution Entropy\n(Higher = More Uniform)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Max probability
    ax = axes[0, 1]
    ax.plot(seq_lens, results['standard_max_prob'], 'o-', label='Standard', linewidth=2)
    ax.plot(seq_lens, results['ssmax_max_prob'], 's-', label='SSMax', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Max Attention Probability')
    ax.set_title('Peak Attention Probability\n(Higher = More Peaked)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Entropy difference
    ax = axes[1, 0]
    entropy_diff = [s - st for s, st in zip(results['ssmax_entropy'], results['standard_entropy'])]
    ax.plot(seq_lens, entropy_diff, 'o-', color='green', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Entropy Difference (SSMax - Standard)')
    ax.set_title('SSMax Effect on Entropy\n(Positive = SSMax More Uniform)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Scale factor vs seq len
    ax = axes[1, 1]
    ax.plot(seq_lens, results['scale_factor'], 'o-', color='red', linewidth=2)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('SSMax Scale Factor (scale * log(n))')
    ax.set_title(f'SSMax Query Scaling\n(scale={ssmax_scale.mean().item():.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_dir = Path('logs/ssmax_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'ssmax_attention_effect.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved → {save_path}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"SSMax scale param: {ssmax_scale.mean().item():.3f}")
    print(f"\nEffect on attention distribution:")
    print(f"  • Entropy change: {entropy_diff[0]:+.3f} (seq=16) → {entropy_diff[-1]:+.3f} (seq=256)")
    print(f"  • Max prob change: {results['ssmax_max_prob'][0] - results['standard_max_prob'][0]:+.3f} (seq=16) → "
          f"{results['ssmax_max_prob'][-1] - results['standard_max_prob'][-1]:+.3f} (seq=256)")
    print(f"\nInterpretation:")
    if entropy_diff[-1] > 0:
        print(f"  ✓ SSMax makes attention MORE UNIFORM at long sequences")
        print(f"    (prevents attention collapse/fading)")
    else:
        print(f"  ✗ SSMax makes attention MORE PEAKED at long sequences")
    print("="*80)


def test_ssmax_attention_heatmap():
    """
    Visualize attention patterns: standard vs SSMax
    """
    batch_size = 1
    num_heads = 1  # Single head for clarity
    seq_len = 64
    head_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # SSMax scale
    ssmax_scale = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.15, device=device, dtype=dtype))
    
    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    mask = make_causal_mask(batch_size, seq_len, device, dtype)
    
    # Compute attention
    attn_standard = compute_attention_scores(query, key, mask, head_dim, use_ssmax=False)
    attn_ssmax = compute_attention_scores(query, key, mask, head_dim, use_ssmax=True, 
                                          ssmax_scale=ssmax_scale, kv_len=seq_len)
    
    # Extract single head
    attn_standard = attn_standard[0, 0].detach().cpu().numpy()
    attn_ssmax = attn_ssmax[0, 0].detach().cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Standard
    im1 = axes[0].imshow(attn_standard, cmap='viridis', aspect='auto', vmin=0, vmax=0.1)
    axes[0].set_title('Standard Softmax Attention')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # SSMax
    im2 = axes[1].imshow(attn_ssmax, cmap='viridis', aspect='auto', vmin=0, vmax=0.1)
    axes[1].set_title(f'SSMax Attention (scale={ssmax_scale.mean().item():.3f})')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = attn_ssmax - attn_standard
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.02, vmax=0.02)
    axes[2].set_title('Difference (SSMax - Standard)')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Save
    save_dir = Path('logs/ssmax_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'ssmax_attention_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Heatmap saved → {save_path}")
    
    plt.close()


if __name__ == '__main__':
    test_ssmax_effect_on_scores()
    test_ssmax_attention_heatmap()
