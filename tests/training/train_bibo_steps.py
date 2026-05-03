"""
Train BiBo model for 50 steps
Track loss + expert selection per layer
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM


def train_50_steps():
    """Train for 50 steps, track loss + expert selection"""
    print("=" * 70)
    print("BiBo 50-Step Training")
    print("=" * 70)
    
    # Bigger config
    config = BiBoConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_routed_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=384,
        max_position_embeddings=512,
        use_ssmax=True,
        use_sliding_window=False,
        mlp_only_layers=[0, 5],  # First + last dense
    )
    
    print(f"\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_heads: {config.num_attention_heads}")
    print(f"  num_experts: {config.num_routed_experts}")
    print(f"  top_k: {config.num_experts_per_tok}")
    print(f"  mlp_only_layers: {config.mlp_only_layers}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = BiBoForCausalLM(config).to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Data
    batch_size = 4
    seq_len = 128
    
    print(f"\nTraining:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  steps: 50")
    print(f"  lr: 1e-4")
    
    # Track
    losses = []
    expert_selections = defaultdict(lambda: defaultdict(int))  # {layer_idx: {expert_idx: count}}
    
    # Hook to capture expert selections
    def make_expert_hook(layer_idx):
        def hook(module, input, output):
            if hasattr(module, 'gate'):
                # Capture routing decisions
                hidden = input[0]
                with torch.no_grad():
                    selected, _ = module.gate(hidden)
                    # selected: [batch, seq, top_k]
                    for expert_idx in selected.flatten().cpu().tolist():
                        expert_selections[layer_idx][expert_idx] += 1
        return hook
    
    # Register hooks on MoE layers
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            layer.mlp.register_forward_hook(make_expert_hook(i))
    
    print("\n" + "-" * 70)
    print("Training...")
    
    for step in range(50):
        # Random batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:3d}: loss={loss.item():.4f}")
    
    print("-" * 70)
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f} ({(losses[0]-losses[-1])/losses[0]*100:.1f}%)")
    
    # Plot loss
    print("\n" + "=" * 70)
    print("Plotting results...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Loss curve
    ax = axes[0]
    ax.plot(range(1, 51), losses, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss (50 Steps)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    
    # Expert selection heatmap
    ax = axes[1]
    
    # Get MoE layers
    moe_layers = [i for i in range(config.num_hidden_layers) if i not in config.mlp_only_layers]
    
    if moe_layers:
        # Build matrix: [num_moe_layers, num_experts]
        matrix = np.zeros((len(moe_layers), config.num_routed_experts))
        
        for i, layer_idx in enumerate(moe_layers):
            for expert_idx in range(config.num_routed_experts):
                matrix[i, expert_idx] = expert_selections[layer_idx].get(expert_idx, 0)
        
        # Normalize by row (per layer)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_norm = np.where(row_sums > 0, matrix / row_sums, 0)
        
        im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=matrix_norm.max())
        
        ax.set_xlabel('Expert Index', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_title('Expert Selection Distribution (Normalized per Layer)', fontsize=14, fontweight='bold')
        
        # Ticks
        ax.set_xticks(range(config.num_routed_experts))
        ax.set_xticklabels(range(config.num_routed_experts))
        ax.set_yticks(range(len(moe_layers)))
        ax.set_yticklabels([f"Layer {idx}" for idx in moe_layers])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Selection Frequency', fontsize=11)
        
        # Annotate cells
        for i in range(len(moe_layers)):
            for j in range(config.num_routed_experts):
                val = matrix_norm[i, j]
                if val > 0.01:  # Only show if > 1%
                    text_color = 'white' if val > matrix_norm.max() * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('training_50steps.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: training_50steps.png")
    
    # Print expert stats
    print("\n" + "=" * 70)
    print("Expert Selection Statistics:")
    print("=" * 70)
    
    for layer_idx in moe_layers:
        print(f"\nLayer {layer_idx} (MoE):")
        total = sum(expert_selections[layer_idx].values())
        
        # Sort by selection count
        sorted_experts = sorted(
            expert_selections[layer_idx].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"  Total selections: {total}")
        print(f"  Top 5 experts:")
        for expert_idx, count in sorted_experts[:5]:
            pct = count / total * 100 if total > 0 else 0
            print(f"    Expert {expert_idx:2d}: {count:6d} ({pct:5.1f}%)")
        
        # Check balance
        if sorted_experts:
            max_count = sorted_experts[0][1]
            min_count = sorted_experts[-1][1]
            balance_ratio = min_count / max_count if max_count > 0 else 0
            print(f"  Balance ratio (min/max): {balance_ratio:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train_50_steps()
