"""
Train BiBo with mixed routers (conv + MLP) for 100 steps
Track loss + expert selection + router bias updates
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM


def train_mixed_routers():
    """Train with conv + MLP routers, track bias updates"""
    print("=" * 70)
    print("BiBo Mixed Router Training (100 Steps)")
    print("=" * 70)
    
    # Config with mixed routers
    config = BiBoConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_routed_experts=8,  # 1 identity + 1 zero + 1 noise + 1 relu + 4 MLP
        num_shared_experts=1,  # 1 causal conv
        num_experts_per_tok=4,
        moe_intermediate_size=384,
        max_position_embeddings=512,
        use_ssmax=True,
        use_sliding_window=False,
        mlp_only_layers=[0, 5],  # First + last dense
        bias_update_factor=1e-2,
        bias_update_threshold=2048,  # Update every 2048 tokens (4 batch × 128 seq × 4 steps)
        router_type="mlp",  # Will override per layer
        kernel_size=3,
    )
    
    print(f"\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_routed_experts: {config.num_routed_experts}")
    print(f"  num_shared_experts: {config.num_shared_experts}")
    print(f"  top_k: {config.num_experts_per_tok}")
    print(f"  bias_update_threshold: {config.bias_update_threshold} tokens")
    print(f"  mlp_only_layers: {config.mlp_only_layers}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model with default router_type
    model = BiBoForCausalLM(config).to(device)
    
    # Override router types: layer 1,3 = conv, layer 2,4 = MLP
    print("\nRouter configuration:")
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            if i in [1, 3]:
                # Change to conv router
                old_gate = layer.mlp.gate
                # Create new config with conv router
                conv_config = BiBoConfig(**{k: v for k, v in config.__dict__.items()})
                conv_config.router_type = "conv"
                from src.modeling.ffn.router import BiBoMoERouter
                layer.mlp.gate = BiBoMoERouter(conv_config).to(device)
                # Copy bias
                if hasattr(old_gate, 'bias'):
                    layer.mlp.gate.bias.data.copy_(old_gate.bias.data)
                print(f"  Layer {i}: Conv router (kernel_size={config.kernel_size})")
            else:
                print(f"  Layer {i}: MLP router")
        else:
            print(f"  Layer {i}: Dense MLP (no router)")
    
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total_params:,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Data
    batch_size = 4
    seq_len = 128
    
    print(f"\nTraining:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  steps: 100")
    print(f"  lr: 1e-4")
    
    # Track
    losses = []
    expert_selections = defaultdict(lambda: defaultdict(int))
    router_bias_history = defaultdict(list)  # {layer_idx: [bias_snapshots]}
    bias_update_steps = []  # Steps where bias was updated
    
    # Hook to capture expert selections
    def make_expert_hook(layer_idx):
        def hook(module, input, output):
            if hasattr(module, 'gate'):
                hidden = input[0]
                with torch.no_grad():
                    selected, _ = module.gate(hidden)
                    for expert_idx in selected.flatten().cpu().tolist():
                        expert_selections[layer_idx][expert_idx] += 1
        return hook
    
    # Register hooks
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            layer.mlp.register_forward_hook(make_expert_hook(i))
    
    print("\n" + "-" * 70)
    print("Training...")
    
    for step in range(100):
        # Random batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Capture bias before forward
        bias_before = {}
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'bias'):
                bias_before[i] = layer.mlp.gate.bias.data.clone()
        
        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # Check if bias updated
        bias_updated = False
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'bias'):
                bias_after = layer.mlp.gate.bias.data
                if not torch.allclose(bias_before[i], bias_after, atol=1e-8):
                    bias_updated = True
                    router_bias_history[i].append(bias_after.cpu().clone())
        
        if bias_updated:
            bias_update_steps.append(step + 1)
        
        # Log bias periodically
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}: loss={loss.item():.4f}")
            for i, layer in enumerate(model.model.layers):
                if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'bias'):
                    bias = layer.mlp.gate.bias.data
                    router_type = "Conv" if i in [1, 3] else "MLP"
                    print(f"    Layer {i} ({router_type}): bias mean={bias.mean().item():.4f}, std={bias.std().item():.4f}")
    
    print("-" * 70)
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f} ({(losses[0]-losses[-1])/losses[0]*100:.1f}%)")
    print(f"\nBias updates occurred at steps: {bias_update_steps}")
    
    # Plot
    print("\n" + "=" * 70)
    print("Plotting results...")
    
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Loss curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(1, 101), losses, linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss (100 Steps)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 100)
    
    # Expert selection heatmap
    ax2 = fig.add_subplot(gs[1, :])
    moe_layers = [i for i in range(config.num_hidden_layers) if i not in config.mlp_only_layers]
    
    if moe_layers:
        matrix = np.zeros((len(moe_layers), config.num_routed_experts))
        for i, layer_idx in enumerate(moe_layers):
            for expert_idx in range(config.num_routed_experts):
                matrix[i, expert_idx] = expert_selections[layer_idx].get(expert_idx, 0)
        
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_norm = np.where(row_sums > 0, matrix / row_sums, 0)
        
        im = ax2.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=matrix_norm.max())
        ax2.set_xlabel('Expert Index', fontsize=12)
        ax2.set_ylabel('Layer Index', fontsize=12)
        
        # Add router type to y-labels
        y_labels = []
        for idx in moe_layers:
            router_type = "Conv" if idx in [1, 3] else "MLP"
            y_labels.append(f"L{idx} ({router_type})")
        ax2.set_yticks(range(len(moe_layers)))
        ax2.set_yticklabels(y_labels)
        
        ax2.set_title('Expert Selection Distribution (Normalized per Layer)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(config.num_routed_experts))
        ax2.set_xticklabels(range(config.num_routed_experts))
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Selection Frequency', fontsize=11)
        
        for i in range(len(moe_layers)):
            for j in range(config.num_routed_experts):
                val = matrix_norm[i, j]
                if val > 0.01:
                    text_color = 'white' if val > matrix_norm.max() * 0.5 else 'black'
                    ax2.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=text_color, fontsize=7)
    
    # Router bias evolution (layer 1 - Conv)
    ax3 = fig.add_subplot(gs[2, 0])
    if 1 in router_bias_history and len(router_bias_history[1]) > 0:
        bias_array = torch.stack(router_bias_history[1]).numpy()  # [num_updates, num_experts]
        for expert_idx in range(min(5, bias_array.shape[1])):  # Plot first 5 experts
            ax3.plot(bias_array[:, expert_idx], label=f'Expert {expert_idx}', linewidth=1.5)
        ax3.set_xlabel('Bias Update Count', fontsize=11)
        ax3.set_ylabel('Bias Value', fontsize=11)
        ax3.set_title('Layer 1 Router Bias Evolution (Conv)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
    
    # Router bias evolution (layer 2 - MLP)
    ax4 = fig.add_subplot(gs[2, 1])
    if 2 in router_bias_history and len(router_bias_history[2]) > 0:
        bias_array = torch.stack(router_bias_history[2]).numpy()
        for expert_idx in range(min(5, bias_array.shape[1])):
            ax4.plot(bias_array[:, expert_idx], label=f'Expert {expert_idx}', linewidth=1.5)
        ax4.set_xlabel('Bias Update Count', fontsize=11)
        ax4.set_ylabel('Bias Value', fontsize=11)
        ax4.set_title('Layer 2 Router Bias Evolution (MLP)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3)
    
    plt.savefig('training_mixed_routers.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: training_mixed_routers.png")
    
    # Print stats
    print("\n" + "=" * 70)
    print("Expert Selection Statistics:")
    print("=" * 70)
    
    for layer_idx in moe_layers:
        router_type = "Conv" if layer_idx in [1, 3] else "MLP"
        print(f"\nLayer {layer_idx} ({router_type} router):")
        total = sum(expert_selections[layer_idx].values())
        
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
        
        if sorted_experts:
            max_count = sorted_experts[0][1]
            min_count = sorted_experts[-1][1]
            balance_ratio = min_count / max_count if max_count > 0 else 0
            print(f"  Balance ratio (min/max): {balance_ratio:.3f}")
    
    # Router bias final state
    print("\n" + "=" * 70)
    print("Final Router Bias State:")
    print("=" * 70)
    
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, 'bias'):
            bias = layer.mlp.gate.bias.data.cpu()
            router_type = "Conv" if i in [1, 3] else "MLP"
            print(f"\nLayer {i} ({router_type} router):")
            print(f"  Bias shape: {bias.shape}")
            print(f"  Mean: {bias.mean().item():.4f}")
            print(f"  Std: {bias.std().item():.4f}")
            print(f"  Min: {bias.min().item():.4f}")
            print(f"  Max: {bias.max().item():.4f}")
            print(f"  Top 3 bias values: {bias.topk(3).values.tolist()}")
            print(f"  Bottom 3 bias values: {bias.topk(3, largest=False).values.tolist()}")
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train_mixed_routers()
