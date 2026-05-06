"""
Long context benchmark: BiBo+SSMax vs Qwen3MoE
Train until loss ~1.5, then test on varying seq lengths (64, 128, 256)
Compare loss + attention entropy across seq lengths
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM


class PatternDataset(Dataset):
    """Pattern-based language modeling"""
    def __init__(self, vocab_size=2000, num_samples=1000, seq_len=128, seed=42):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        torch.manual_seed(seed)
        self.sequences = []
        
        for _ in range(num_samples):
            seq = []
            
            # Arithmetic
            start = torch.randint(0, vocab_size // 4, (1,)).item()
            step = torch.randint(1, 10, (1,)).item()
            for i in range(seq_len // 4):
                seq.append((start + i * step) % vocab_size)
            
            # Repeating motifs
            motif_len = torch.randint(3, 8, (1,)).item()
            motif = torch.randint(0, vocab_size, (motif_len,)).tolist()
            for _ in range(seq_len // 4 // motif_len):
                seq.extend(motif)
            
            # Fibonacci-like
            a, b = torch.randint(0, 100, (2,)).tolist()
            for _ in range(seq_len // 4):
                seq.append(a % vocab_size)
                a, b = b, (a + b)
            
            # Local correlations
            for i in range(seq_len // 4):
                if i == 0:
                    seq.append(torch.randint(0, vocab_size, (1,)).item())
                else:
                    prev_sum = sum(seq[-3:]) if len(seq) >= 3 else seq[-1]
                    seq.append((prev_sum * 7 + 13) % vocab_size)
            
            if len(seq) < seq_len:
                seq.extend([0] * (seq_len - len(seq)))
            seq = seq[:seq_len]
            
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine LR schedule"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_attention_entropy(model, dataloader, device, max_batches=10):
    """
    Compute avg attention entropy across dataset.
    Higher entropy = more uniform attention.
    """
    model.eval()
    entropies = []
    max_probs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            inputs = inputs.to(device)
            
            # Forward w/ attention output
            outputs = model(input_ids=inputs, output_attentions=True)
            
            if outputs.attentions is None:
                continue
            
            # Compute entropy for each layer
            for layer_attn in outputs.attentions:
                # layer_attn: (batch, num_heads, seq_len, seq_len)
                if layer_attn is None:
                    continue
                    
                # Avoid log(0)
                attn_probs = torch.clamp(layer_attn, min=1e-10)
                entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean().item()
                entropies.append(entropy)
                
                # Max prob (peakedness)
                max_prob = attn_probs.max(dim=-1)[0].mean().item()
                max_probs.append(max_prob)
    
    return {
        'entropy': np.mean(entropies) if entropies else 0.0,
        'max_prob': np.mean(max_probs) if max_probs else 0.0,
    }


def train_until_target_loss(model, model_name, dataloader, device, target_loss=1.5, 
                            max_steps=2000, lr=1e-3, warmup_steps=50, use_wandb=False):
    """Train until loss reaches target"""
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} until loss ~{target_loss}")
    print(f"{'='*70}\n")
    
    step = 0
    losses = []
    
    while step < max_steps:
        for inputs, targets in dataloader:
            if step >= max_steps:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            losses.append(loss.item())
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'{model_name}/train_loss': loss.item(),
                    f'{model_name}/lr': current_lr,
                    f'{model_name}/step': step,
                })
            
            if (step + 1) % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                print(f"  Step {step+1:4d}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={current_lr:.2e}")
                
                # Check if target reached
                if avg_loss <= target_loss:
                    print(f"\n✓ Target loss {target_loss} reached at step {step+1}")
                    return step + 1, losses
            
            step += 1
    
    print(f"\n⚠ Max steps {max_steps} reached, final loss: {np.mean(losses[-50:]):.4f}")
    return step, losses


def evaluate_on_seq_lengths(model, model_name, vocab_size, device, seq_lens=[64, 128, 256], 
                            batch_size=8, num_samples=200):
    """Evaluate model on different seq lengths"""
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} on varying seq lengths")
    print(f"{'='*70}\n")
    
    for seq_len in seq_lens:
        print(f"Seq len: {seq_len}")
        
        # Create val dataset
        val_dataset = PatternDataset(vocab_size=vocab_size, num_samples=num_samples, 
                                     seq_len=seq_len, seed=123)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        losses = []
        perplexities = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(input_ids=inputs, labels=targets)
                loss = outputs.loss.item()
                losses.append(loss)
                perplexities.append(np.exp(loss))
        
        avg_loss = np.mean(losses)
        avg_ppl = np.mean(perplexities)
        
        # Compute attention stats
        attn_stats = compute_attention_entropy(model, val_loader, device, max_batches=10)
        
        results[seq_len] = {
            'loss': avg_loss,
            'perplexity': avg_ppl,
            'entropy': attn_stats['entropy'],
            'max_prob': attn_stats['max_prob'],
        }
        
        print(f"  Loss: {avg_loss:.4f}, PPL: {avg_ppl:.2f}")
        print(f"  Entropy: {attn_stats['entropy']:.3f}, Max Prob: {attn_stats['max_prob']:.3f}\n")
    
    return results


def plot_training_curves(bibo_losses, qwen_losses, bibo_steps, qwen_steps, save_dir):
    """Plot training loss curves comparison"""
    print(f"\nGenerating training curves comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    ax = axes[0]
    ax.plot(range(len(bibo_losses)), bibo_losses, label='BiBo+SSMax', linewidth=1.5, alpha=0.7, color='#2E86AB')
    ax.plot(range(len(qwen_losses)), qwen_losses, label='Qwen3MoE', linewidth=1.5, alpha=0.7, color='#A23B72')
    ax.axhline(y=1.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target Loss')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Steps to target comparison
    ax = axes[1]
    models = ['BiBo+SSMax', 'Qwen3MoE']
    steps = [bibo_steps, qwen_steps]
    colors = ['#2E86AB', '#A23B72']
    bars = ax.bar(models, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, step in zip(bars, steps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(step)} steps',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Steps to Reach Loss 1.5', fontsize=12)
    ax.set_title('Training Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add winner annotation
    winner = 'BiBo+SSMax' if bibo_steps < qwen_steps else 'Qwen3MoE'
    diff = abs(bibo_steps - qwen_steps)
    diff_pct = diff / max(bibo_steps, qwen_steps) * 100
    ax.text(0.5, 0.95, f'Winner: {winner}\n({diff_pct:.1f}% faster)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved → {save_path}")
    plt.close()


def plot_attention_heatmaps(bibo_model, qwen_model, vocab_size, device, save_dir, seq_len=128):
    """Plot attention heatmaps: BiBo+SSMax vs Qwen"""
    print(f"\nGenerating attention heatmaps (seq_len={seq_len})...")
    
    # Create sample
    torch.manual_seed(42)
    val_dataset = PatternDataset(vocab_size=vocab_size, num_samples=1, seq_len=seq_len, seed=999)
    sample_input = val_dataset[0][0].unsqueeze(0).to(device)  # (1, seq_len)
    
    # Get attention from both models
    bibo_model.eval()
    qwen_model.eval()
    
    with torch.no_grad():
        bibo_out = bibo_model(input_ids=sample_input, output_attentions=True)
        qwen_out = qwen_model(input_ids=sample_input, output_attentions=True)
    
    # Extract middle layer attention (single head)
    mid_layer = len(bibo_out.attentions) // 2
    bibo_attn = bibo_out.attentions[mid_layer][0, 0].cpu().numpy()  # (seq_len, seq_len)
    qwen_attn = qwen_out.attentions[mid_layer][0, 0].cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # BiBo+SSMax
    im1 = axes[0].imshow(bibo_attn, cmap='viridis', aspect='auto', vmin=0, vmax=0.1)
    axes[0].set_title(f'BiBo+SSMax Attention\n(Layer {mid_layer}, Head 0)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Qwen
    im2 = axes[1].imshow(qwen_attn, cmap='viridis', aspect='auto', vmin=0, vmax=0.1)
    axes[1].set_title(f'Qwen3MoE Attention\n(Layer {mid_layer}, Head 0)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = bibo_attn - qwen_attn
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.02, vmax=0.02)
    axes[2].set_title('Difference (BiBo - Qwen)\nRed=BiBo higher, Blue=Qwen higher', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    save_path = save_dir / f'attention_heatmap_seq{seq_len}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Heatmap saved → {save_path}")
    plt.close()
    
    # Compute stats
    bibo_entropy = -(bibo_attn * np.log(np.clip(bibo_attn, 1e-10, 1))).sum(axis=-1).mean()
    qwen_entropy = -(qwen_attn * np.log(np.clip(qwen_attn, 1e-10, 1))).sum(axis=-1).mean()
    bibo_max = bibo_attn.max(axis=-1).mean()
    qwen_max = qwen_attn.max(axis=-1).mean()
    
    print(f"  BiBo entropy: {bibo_entropy:.3f}, max_prob: {bibo_max:.3f}")
    print(f"  Qwen entropy: {qwen_entropy:.3f}, max_prob: {qwen_max:.3f}")
    print(f"  Entropy diff: {bibo_entropy - qwen_entropy:+.3f} (BiBo - Qwen)")


def plot_results(bibo_results, qwen_results, save_dir):
    """Plot comprehensive comparison"""
    seq_lens = sorted(bibo_results.keys())
    
    bibo_losses = [bibo_results[s]['loss'] for s in seq_lens]
    qwen_losses = [qwen_results[s]['loss'] for s in seq_lens]
    bibo_ppls = [bibo_results[s]['perplexity'] for s in seq_lens]
    qwen_ppls = [qwen_results[s]['perplexity'] for s in seq_lens]
    bibo_entropies = [bibo_results[s]['entropy'] for s in seq_lens]
    qwen_entropies = [qwen_results[s]['entropy'] for s in seq_lens]
    bibo_max_probs = [bibo_results[s]['max_prob'] for s in seq_lens]
    qwen_max_probs = [qwen_results[s]['max_prob'] for s in seq_lens]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(seq_lens, bibo_losses, 'o-', label='BiBo+SSMax', linewidth=2, markersize=8, color='#2E86AB')
    ax.plot(seq_lens, qwen_losses, 's-', label='Qwen3MoE', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Loss vs Sequence Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Perplexity
    ax = axes[0, 1]
    ax.plot(seq_lens, bibo_ppls, 'o-', label='BiBo+SSMax', linewidth=2, markersize=8, color='#2E86AB')
    ax.plot(seq_lens, qwen_ppls, 's-', label='Qwen3MoE', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Perplexity vs Sequence Length\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Entropy
    ax = axes[1, 0]
    ax.plot(seq_lens, bibo_entropies, 'o-', label='BiBo+SSMax', linewidth=2, markersize=8, color='#2E86AB')
    ax.plot(seq_lens, qwen_entropies, 's-', label='Qwen3MoE', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Attention Entropy', fontsize=12)
    ax.set_title('Attention Entropy vs Sequence Length\n(Higher = More Uniform)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Max Prob
    ax = axes[1, 1]
    ax.plot(seq_lens, bibo_max_probs, 'o-', label='BiBo+SSMax', linewidth=2, markersize=8, color='#2E86AB')
    ax.plot(seq_lens, qwen_max_probs, 's-', label='Qwen3MoE', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Max Attention Probability', fontsize=12)
    ax.set_title('Peak Attention Probability vs Sequence Length\n(Lower = Less Peaked)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = save_dir / 'long_context_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved → {save_path}")
    plt.close()


def main():
    print("="*80)
    print("Long Context Benchmark: BiBo+SSMax vs Qwen3MoE")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    save_dir = Path('logs/long_context')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Small model config (~30M params)
    vocab_size = 2000
    hidden_size = 384
    num_layers = 8
    num_heads = 6
    num_kv_heads = 2
    
    # BiBo: 8 experts (1 identity, 1 relu, 1 zero, 1 noise, 2 MLP, 2 Conv)
    # Special experts are lightweight, so use larger MoE intermediate
    bibo_intermediate = 1024
    bibo_moe_intermediate = 512  # Larger to compensate for special experts
    
    # Qwen: 6 heavier experts to match params
    qwen_intermediate = 1024
    qwen_moe_intermediate = 512  # Match BiBo
    
    # Training config
    train_seq_len = 128
    batch_size = 8
    target_loss = 1.5
    max_steps = 2000
    
    print(f"\nModel size: ~30M params")
    print(f"Training seq len: {train_seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Target loss: {target_loss}")
    
    # Initialize wandb
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project="bibo-long-context",
            name=f"bibo_vs_qwen_{int(time.time())}",
            config={
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "train_seq_len": train_seq_len,
                "batch_size": batch_size,
                "target_loss": target_loss,
                "max_steps": max_steps,
                "bibo_experts": 8,
                "qwen_experts": 6,
                "device": str(device),
            }
        )
        
        # Define metrics
        wandb.define_metric("BiBo/step")
        wandb.define_metric("Qwen3MoE/step")
        wandb.define_metric("BiBo/*", step_metric="BiBo/step")
        wandb.define_metric("Qwen3MoE/*", step_metric="Qwen3MoE/step")
        
        print(f"\n✓ wandb initialized: {wandb.run.url}")
    else:
        print("\n⚠ wandb not available, skipping logging")
    
    # Create training dataset
    print("\nCreating training dataset...")
    train_dataset = PatternDataset(vocab_size=vocab_size, num_samples=800, 
                                   seq_len=train_seq_len, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # BiBo config (with SSMax)
    print("\n" + "-"*80)
    print("Creating BiBo model (with SSMax)...")
    bibo_config = BiBoConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=bibo_intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        num_routed_experts=8,  # 1 identity, 1 relu, 1 zero, 1 noise, 2 MLP, 2 Conv
        num_experts_per_tok=2,
        moe_intermediate_size=bibo_moe_intermediate,
        router_type='mlp',
        mlp_only_layers=[0, num_layers-1],
        use_ssmax=True,  # SSMax enabled
        max_position_embeddings=512,
    )
    bibo_model = BiBoForCausalLM(bibo_config).to(device)
    bibo_params = sum(p.numel() for p in bibo_model.parameters())
    print(f"  Total params: {bibo_params:,}")
    print(f"  Experts: 8 (1 identity, 1 relu, 1 zero, 1 noise, 2 MLP, 2 Conv)")
    print(f"  SSMax: ENABLED")
    
    # Qwen config (no SSMax) - match params
    print("\n" + "-"*80)
    print("Creating Qwen3MoE baseline...")
    qwen_config = Qwen3MoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=qwen_intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        num_experts=6,  # Fewer but heavier
        num_experts_per_tok=2,
        moe_intermediate_size=qwen_moe_intermediate,
        mlp_only_layers=[0, num_layers-1],
        max_position_embeddings=512,
    )
    qwen_model = Qwen3MoeForCausalLM(qwen_config).to(device)
    qwen_params = sum(p.numel() for p in qwen_model.parameters())
    print(f"  Total params: {qwen_params:,}")
    print(f"  Experts: 6 (standard MLP)")
    print(f"  SSMax: DISABLED")
    
    param_diff = abs(bibo_params - qwen_params)
    param_diff_pct = param_diff / max(bibo_params, qwen_params) * 100
    print(f"\nParam difference: {param_diff:,} ({param_diff_pct:.2f}%)")
    
    if param_diff_pct > 10:
        print(f"⚠ Warning: Param mismatch > 10%")
        print(f"  This may affect fair comparison. Consider adjusting config.")
    else:
        print(f"✓ Param difference within acceptable range (<10%)")
    
    # Train BiBo
    print("\n" + "="*80)
    print("TRAINING BIBO")
    print("="*80)
    bibo_steps, bibo_train_losses = train_until_target_loss(
        bibo_model, 'BiBo+SSMax', train_loader, device, 
        target_loss=target_loss, max_steps=max_steps, use_wandb=use_wandb
    )
    
    # Save BiBo checkpoint
    bibo_ckpt_path = save_dir / 'bibo_checkpoint.pt'
    torch.save({
        'model_state_dict': bibo_model.state_dict(),
        'config': bibo_config,
        'steps': bibo_steps,
    }, bibo_ckpt_path)
    print(f"✓ BiBo checkpoint saved → {bibo_ckpt_path}")
    
    # Train Qwen
    print("\n" + "="*80)
    print("TRAINING QWEN3MOE")
    print("="*80)
    qwen_steps, qwen_train_losses = train_until_target_loss(
        qwen_model, 'Qwen3MoE', train_loader, device,
        target_loss=target_loss, max_steps=max_steps, use_wandb=use_wandb
    )
    
    # Save Qwen checkpoint
    qwen_ckpt_path = save_dir / 'qwen_checkpoint.pt'
    torch.save({
        'model_state_dict': qwen_model.state_dict(),
        'config': qwen_config,
        'steps': qwen_steps,
    }, qwen_ckpt_path)
    print(f"✓ Qwen checkpoint saved → {qwen_ckpt_path}")
    
    # Evaluate on varying seq lengths
    print("\n" + "="*80)
    print("EVALUATION ON VARYING SEQ LENGTHS")
    print("="*80)
    
    seq_lens = [64, 128, 256]
    
    bibo_results = evaluate_on_seq_lengths(
        bibo_model, 'BiBo+SSMax', vocab_size, device, seq_lens=seq_lens
    )
    
    qwen_results = evaluate_on_seq_lengths(
        qwen_model, 'Qwen3MoE', vocab_size, device, seq_lens=seq_lens
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Seq Len':<10} {'Metric':<15} {'BiBo+SSMax':<15} {'Qwen3MoE':<15} {'Winner':<10}")
    print("-"*65)
    
    for seq_len in seq_lens:
        bibo_loss = bibo_results[seq_len]['loss']
        qwen_loss = qwen_results[seq_len]['loss']
        bibo_ppl = bibo_results[seq_len]['perplexity']
        qwen_ppl = qwen_results[seq_len]['perplexity']
        bibo_ent = bibo_results[seq_len]['entropy']
        qwen_ent = qwen_results[seq_len]['entropy']
        bibo_max = bibo_results[seq_len]['max_prob']
        qwen_max = qwen_results[seq_len]['max_prob']
        
        print(f"{seq_len:<10} {'Loss':<15} {bibo_loss:<15.4f} {qwen_loss:<15.4f} "
              f"{'BiBo' if bibo_loss < qwen_loss else 'Qwen':<10}")
        print(f"{'':<10} {'Perplexity':<15} {bibo_ppl:<15.2f} {qwen_ppl:<15.2f} "
              f"{'BiBo' if bibo_ppl < qwen_ppl else 'Qwen':<10}")
        print(f"{'':<10} {'Entropy':<15} {bibo_ent:<15.3f} {qwen_ent:<15.3f} "
              f"{'BiBo' if bibo_ent > qwen_ent else 'Qwen':<10}")
        print(f"{'':<10} {'Max Prob':<15} {bibo_max:<15.3f} {qwen_max:<15.3f} "
              f"{'BiBo' if bibo_max < qwen_max else 'Qwen':<10}")
        print()
    
    print("="*65)
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    # Loss trend
    loss_diff_64 = bibo_results[64]['loss'] - qwen_results[64]['loss']
    loss_diff_256 = bibo_results[256]['loss'] - qwen_results[256]['loss']
    
    print("\n1. LOSS PERFORMANCE:")
    if loss_diff_256 < loss_diff_64:
        print("  ✓ BiBo+SSMax improves MORE at longer sequences")
        print(f"    Loss diff: {loss_diff_64:+.4f} (seq=64) → {loss_diff_256:+.4f} (seq=256)")
        print("    → SSMax prevents attention fading at long context")
    else:
        print("  ⚠ BiBo+SSMax does not show clear long-context advantage")
        print(f"    Loss diff: {loss_diff_64:+.4f} (seq=64) → {loss_diff_256:+.4f} (seq=256)")
    
    # Entropy
    ent_diff_256 = bibo_results[256]['entropy'] - qwen_results[256]['entropy']
    print("\n2. ATTENTION DISTRIBUTION:")
    if ent_diff_256 > 0:
        print(f"  ✓ BiBo+SSMax has higher entropy at seq=256 (+{ent_diff_256:.3f})")
        print("    → More uniform attention (all tokens accessible)")
    else:
        print(f"  ⚠ BiBo+SSMax has lower entropy at seq=256 ({ent_diff_256:+.3f})")
    
    # Max prob
    max_diff_256 = bibo_results[256]['max_prob'] - qwen_results[256]['max_prob']
    if max_diff_256 < 0:
        print(f"  ✓ BiBo+SSMax has lower max_prob at seq=256 ({max_diff_256:+.3f})")
        print("    → Less peaked attention (prevents collapse)")
    
    # FFN advantage
    print("\n3. FFN ARCHITECTURE:")
    print("  BiBo uses diverse experts:")
    print("    • 1 identity (skip connection)")
    print("    • 1 ReLU (non-linearity)")
    print("    • 1 zero (regularization)")
    print("    • 1 noise (exploration)")
    print("    • 2 MLP (standard)")
    print("    • 2 Conv (local patterns)")
    print("  → Richer expert diversity vs Qwen's uniform MLPs")
    
    print("\n4. OVERALL:")
    bibo_wins = sum(1 for s in seq_lens if bibo_results[s]['loss'] < qwen_results[s]['loss'])
    if bibo_wins >= 2:
        print(f"  ✓ BiBo+SSMax wins on {bibo_wins}/{len(seq_lens)} seq lengths")
        print("    → SSMax + diverse FFN = better long-context modeling")
    else:
        print(f"  ⚠ BiBo+SSMax wins on {bibo_wins}/{len(seq_lens)} seq lengths")
        print("    → May need more training or tuning")
    
    # Save results
    results = {
        'training': {
            'bibo_steps': bibo_steps,
            'qwen_steps': qwen_steps,
            'bibo_final_loss': bibo_train_losses[-1],
            'qwen_final_loss': qwen_train_losses[-1],
            'winner': 'BiBo+SSMax' if bibo_steps < qwen_steps else 'Qwen3MoE',
            'efficiency_gain_pct': abs(bibo_steps - qwen_steps) / max(bibo_steps, qwen_steps) * 100,
        },
        'bibo': {
            'params': bibo_params,
            'train_steps': bibo_steps,
            'results': bibo_results,
        },
        'qwen': {
            'params': qwen_params,
            'train_steps': qwen_steps,
            'results': qwen_results,
        },
        'comparison': {
            'param_diff': param_diff,
            'param_diff_pct': param_diff_pct,
            'seq_lens_tested': seq_lens,
        }
    }
    
    results_path = save_dir / 'long_context_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved → {results_path}")
    
    # Plot
    plot_results(bibo_results, qwen_results, save_dir)
    
    # Plot attention heatmaps
    for seq_len in [64, 128, 256]:
        plot_attention_heatmaps(bibo_model, qwen_model, vocab_size, device, save_dir, seq_len=seq_len)
    
    # Plot training curves comparison
    plot_training_curves(bibo_train_losses, qwen_train_losses, bibo_steps, qwen_steps, save_dir)
    
    # Log to wandb
    if use_wandb:
        print("\n" + "="*80)
        print("LOGGING TO WANDB")
        print("="*80)
        
        # Log training comparison
        wandb.log({
            'training/bibo_steps_to_target': bibo_steps,
            'training/qwen_steps_to_target': qwen_steps,
            'training/bibo_final_loss': bibo_train_losses[-1],
            'training/qwen_final_loss': qwen_train_losses[-1],
        })
        
        # Log validation results
        for seq_len in seq_lens:
            wandb.log({
                f'val/bibo_loss_seq{seq_len}': bibo_results[seq_len]['loss'],
                f'val/qwen_loss_seq{seq_len}': qwen_results[seq_len]['loss'],
                f'val/bibo_ppl_seq{seq_len}': bibo_results[seq_len]['perplexity'],
                f'val/qwen_ppl_seq{seq_len}': qwen_results[seq_len]['perplexity'],
                f'val/bibo_entropy_seq{seq_len}': bibo_results[seq_len]['entropy'],
                f'val/qwen_entropy_seq{seq_len}': qwen_results[seq_len]['entropy'],
                f'val/bibo_max_prob_seq{seq_len}': bibo_results[seq_len]['max_prob'],
                f'val/qwen_max_prob_seq{seq_len}': qwen_results[seq_len]['max_prob'],
            })
        
        # Create comparison tables
        train_table = wandb.Table(
            columns=["Model", "Steps to Loss 1.5", "Final Loss"],
            data=[
                ["BiBo+SSMax", bibo_steps, bibo_train_losses[-1]],
                ["Qwen3MoE", qwen_steps, qwen_train_losses[-1]],
            ]
        )
        wandb.log({"training_comparison": train_table})
        
        val_table = wandb.Table(
            columns=["Seq Len", "BiBo Loss", "Qwen Loss", "BiBo PPL", "Qwen PPL", "BiBo Entropy", "Qwen Entropy"],
            data=[
                [
                    seq_len,
                    bibo_results[seq_len]['loss'],
                    qwen_results[seq_len]['loss'],
                    bibo_results[seq_len]['perplexity'],
                    qwen_results[seq_len]['perplexity'],
                    bibo_results[seq_len]['entropy'],
                    qwen_results[seq_len]['entropy'],
                ]
                for seq_len in seq_lens
            ]
        )
        wandb.log({"validation_comparison": val_table})
        
        # Upload plots
        wandb.log({
            "plots/training_curves": wandb.Image(str(save_dir / 'training_curves.png')),
            "plots/validation_comparison": wandb.Image(str(save_dir / 'long_context_comparison.png')),
            "plots/heatmap_seq64": wandb.Image(str(save_dir / 'attention_heatmap_seq64.png')),
            "plots/heatmap_seq128": wandb.Image(str(save_dir / 'attention_heatmap_seq128.png')),
            "plots/heatmap_seq256": wandb.Image(str(save_dir / 'attention_heatmap_seq256.png')),
        })
        
        # Create custom comparison charts
        # Training efficiency
        wandb.log({
            "charts/steps_to_target": wandb.plot.bar(
                train_table,
                "Model",
                "Steps to Loss 1.5",
                title="Training Efficiency: Steps to Reach Loss 1.5"
            )
        })
        
        # Loss comparison across seq lens
        loss_data = [[seq_len, bibo_results[seq_len]['loss'], qwen_results[seq_len]['loss']] 
                     for seq_len in seq_lens]
        loss_table = wandb.Table(columns=["Seq Len", "BiBo", "Qwen"], data=loss_data)
        wandb.log({
            "charts/loss_vs_seqlen": wandb.plot.line_series(
                xs=[seq_lens, seq_lens],
                ys=[[bibo_results[s]['loss'] for s in seq_lens],
                    [qwen_results[s]['loss'] for s in seq_lens]],
                keys=["BiBo+SSMax", "Qwen3MoE"],
                title="Validation Loss vs Sequence Length",
                xname="Sequence Length"
            )
        })
        
        # Entropy comparison
        wandb.log({
            "charts/entropy_vs_seqlen": wandb.plot.line_series(
                xs=[seq_lens, seq_lens],
                ys=[[bibo_results[s]['entropy'] for s in seq_lens],
                    [qwen_results[s]['entropy'] for s in seq_lens]],
                keys=["BiBo+SSMax", "Qwen3MoE"],
                title="Attention Entropy vs Sequence Length",
                xname="Sequence Length"
            )
        })
        
        print(f"\n✓ All results logged to wandb: {wandb.run.url}")
        wandb.finish()
    
    print("\n" + "="*80)
    print("✓ Long context benchmark complete!")
    print("="*80)


if __name__ == '__main__':
    main()
