"""
Router ablation: noise, scaling, bias effects on expert selection.
Very hard regression task, converge to 0.01 loss.

Ablation modes:
1. baseline: no bias update (bias frozen at 0)
2. bias_scaling: bias update + scaling (λ=1.0)
3. bias_scaling_high: bias update + high scaling (λ=2.0)
4. bias_scaling_noise: bias update + scaling + noise

Noise sweep: [0.0, 0.01, 0.02]
Scaling sweep: [1.0, 2.0]
Router types: [mlp, conv]

Expert types:
- Expert 0, 1: MLP (SwiGLU)
- Expert 2, 3: Causal Conv1D

Task: Predict f(x) = sum of multiple non-linear interactions
Very hard for simple model to fit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import copy

from src.configuration_bibo import BiBoConfig
from src.modeling.ffn.router import BiBoMoERouter
from src.modeling.ffn.mlp import BiBoMLP
from src.modeling.ffn.experts import BiBoCausalConv1D


class VeryHardRegressionDataset(Dataset):
    """Very hard regression: predict complex non-linear function with many interactions.
    
    Requires learning multiple non-linear patterns simultaneously.
    Simple models struggle without proper expert specialization.
    """
    def __init__(self, hidden_size=128, num_samples=512, seq_len=32):
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # Fixed random inputs
        torch.manual_seed(42)
        self.inputs = torch.randn(num_samples, seq_len, hidden_size) * 3.0  # wider range
        
        # Very complex target function with many components
        targets = torch.zeros(num_samples, 1)
        
        for i in range(num_samples):
            x = self.inputs[i]  # (seq_len, hidden_size)
            
            # Component 1: Pairwise sequence interactions (O(seq^2))
            for s1 in range(0, seq_len, 3):
                for s2 in range(s1+2, seq_len, 3):
                    targets[i] += torch.sin(x[s1].mean() * x[s2].std()) * torch.cos(x[s1].max() - x[s2].min())
            
            # Component 2: High-order polynomial terms
            mean_val = x.mean()
            std_val = x.std()
            max_val = x.max()
            min_val = x.min()
            targets[i] += mean_val ** 4 - 3 * mean_val ** 3 + 2 * mean_val ** 2 + 7 * mean_val
            targets[i] += std_val ** 3 + 2 * std_val ** 2 - std_val
            targets[i] += (max_val - min_val) ** 3
            
            # Component 3: Cross-channel non-linear interactions
            for h1 in range(0, hidden_size, 8):
                for h2 in range(h1+4, hidden_size, 8):
                    for h3 in range(h2+4, hidden_size, 8):
                        targets[i] += torch.tanh(x[:, h1].sum() * x[:, h2].sum()) * torch.sigmoid(x[:, h3].mean())
            
            # Component 4: Temporal patterns with phase shifts
            for phase in range(4):
                start = phase * (seq_len // 4)
                end = (phase + 1) * (seq_len // 4)
                chunk = x[start:end]
                targets[i] += 5 * torch.sin(chunk.mean() * (phase + 1)) + 3 * torch.cos(chunk.std() / (phase + 1))
            
            # Component 5: Frequency-domain-like patterns
            for freq in [1, 2, 4, 8]:
                sampled = x[::freq]
                targets[i] += torch.sin(sampled.mean() * freq) * torch.exp(-sampled.std() / freq)
            
            # Component 6: Interaction between first/last quarters
            first_quarter = x[:seq_len//4]
            last_quarter = x[-seq_len//4:]
            targets[i] += 10 * torch.tanh((first_quarter.mean() - last_quarter.mean()) ** 2)
            targets[i] += 5 * torch.sigmoid(first_quarter.std() * last_quarter.std())
        
        self.targets = targets
        
        # Normalize targets
        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std()
        self.targets = (self.targets - self.target_mean) / (self.target_std + 1e-6)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MixedExpertLayer(nn.Module):
    """Mixed expert layer: 2 BiBoMLP experts + 2 BiBoCausalConv1D experts"""
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = 4
        
        # Expert 0, 1: BiBoMLP with SwiGLU
        self.mlp_experts = nn.ModuleList([
            BiBoMLP(config, is_expert=True) for _ in range(2)
        ])
        
        # Expert 2, 3: BiBoCausalConv1D
        self.conv_experts = nn.ModuleList([
            BiBoCausalConv1D(config) for _ in range(2)
        ])
        
    def forward(self, x, top_k_indices, norm_weights):
        # x: (batch, seq, hidden)
        # top_k_indices: (batch, seq, top_k)
        # norm_weights: (batch, seq, top_k)
        batch_size, seq_len, hidden_dim = x.shape
        top_k = top_k_indices.shape[-1]
        
        flat_hidden = x.view(-1, hidden_dim)  # (batch*seq, hidden)
        flat_indices = top_k_indices.view(-1, top_k)  # (batch*seq, top_k)
        flat_weights = norm_weights.view(-1, top_k)  # (batch*seq, top_k)
        
        output = torch.zeros_like(flat_hidden)
        
        # Process MLP experts (0, 1)
        for i in range(2):
            mask = (flat_indices == i).any(dim=1)
            if mask.any():
                mask_indices = torch.where(mask)[0]
                batch_indices = mask_indices // seq_len
                seq_indices = mask_indices % seq_len
                
                unique_batches = torch.unique(batch_indices)
                
                for b in unique_batches:
                    b_mask = (batch_indices == b)
                    if b_mask.any():
                        expert_input = x[b:b+1]  # (1, seq, hidden)
                        expert_output = self.mlp_experts[i](expert_input)  # (1, seq, hidden)
                        
                        positions = seq_indices[b_mask]
                        for pos in positions:
                            flat_idx = b * seq_len + pos
                            for k in range(top_k):
                                if flat_indices[flat_idx, k] == i:
                                    output[flat_idx] += expert_output[0, pos] * flat_weights[flat_idx, k]
        
        # Process Conv experts (2, 3)
        for i in range(2, 4):
            mask = (flat_indices == i).any(dim=1)
            if mask.any():
                mask_indices = torch.where(mask)[0]
                batch_indices = mask_indices // seq_len
                seq_indices = mask_indices % seq_len
                
                unique_batches = torch.unique(batch_indices)
                
                for b in unique_batches:
                    b_mask = (batch_indices == b)
                    if b_mask.any():
                        expert_input = x[b:b+1]  # (1, seq, hidden)
                        expert_output = self.conv_experts[i-2](expert_input)  # (1, seq, hidden)
                        
                        positions = seq_indices[b_mask]
                        for pos in positions:
                            flat_idx = b * seq_len + pos
                            for k in range(top_k):
                                if flat_indices[flat_idx, k] == i:
                                    output[flat_idx] += expert_output[0, pos] * flat_weights[flat_idx, k]
        
        return output.view(batch_size, seq_len, hidden_dim)


class TinyMoEModel(nn.Module):
    """1-layer MoE: router + mixed experts (2 BiBoMLP + 2 BiBoCausalConv1D) + regression head"""
    def __init__(self, config: BiBoConfig, update_bias=True):
        super().__init__()
        self.router = BiBoMoERouter(config)
        self.experts = MixedExpertLayer(config)
        self.head = nn.Linear(config.hidden_size, 1)  # regression output
        self.update_bias = update_bias
        
        # Freeze bias if not updating
        if not update_bias:
            self.router.bias.requires_grad = False
        
    def forward(self, x):
        # x: (batch, seq, hidden)
        top_k_indices, norm_weights = self.router(x)
        expert_out = self.experts(x, top_k_indices, norm_weights)
        # Pool sequence: mean over seq_len
        pooled = expert_out.mean(dim=1)  # (batch, hidden)
        output = self.head(pooled)  # (batch, 1)
        return output, top_k_indices, norm_weights
    
    def update_router_bias(self, top_k_indices):
        """Update bias every step based on expert usage"""
        if not self.update_bias:
            return
        
        with torch.no_grad():
            # Count tokens per expert
            flat_indices = top_k_indices.view(-1)
            counts = torch.bincount(flat_indices, minlength=self.router.num_routed_experts).float()
            
            # Target: uniform distribution
            target_count = counts.sum() / self.router.num_routed_experts
            deviation = target_count - counts
            
            # Update bias: increase for under-utilized, decrease for over-utilized
            step_size = 0.01
            self.router.bias.add_(step_size * deviation.sign())


class RouterLogger:
    """Log router internals: logits, expert selection, bias"""
    def __init__(self):
        self.steps = []
        self.logits_pre_noise = []
        self.logits_post_noise = []
        self.logits_post_bias = []
        self.logits_post_scaling = []
        self.expert_selections = []
        self.bias_values = []
        self.losses = []
        
    def log(self, step, loss, router, hidden_states, top_k_indices):
        """Capture router state at this step"""
        self.steps.append(step)
        self.losses.append(loss)
        
        with torch.no_grad():
            # Recompute logits to capture intermediate states
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_hidden = hidden_states.view(-1, hidden_dim)
            
            # Pre-noise logits (handle both mlp and conv)
            if router.router_type == "mlp":
                logits_raw = router.gate_proj(flat_hidden).float()
            else:  # conv
                x_perm = hidden_states.permute(0, 2, 1)  # (b, h, s)
                x_padded = F.pad(x_perm, (router.causal_padding, 0))
                conv_out = router.gate_conv(x_padded)
                logits_raw = conv_out.permute(0, 2, 1).reshape(-1, router.num_routed_experts).float()
            
            self.logits_pre_noise.append(logits_raw.cpu().numpy())
            
            # Post-noise (if training + noise > 0)
            if router.training and router.router_noise > 0:
                noise_stddev = np.sqrt(router.router_noise)
                noise = torch.randn_like(logits_raw) * noise_stddev
                logits_noisy = logits_raw + noise
            else:
                logits_noisy = logits_raw
            self.logits_post_noise.append(logits_noisy.cpu().numpy())
            
            # Post-bias
            logits_biased = logits_noisy + router.bias
            self.logits_post_bias.append(logits_biased.cpu().numpy())
            
            # Post-scaling
            mean = logits_biased.mean(dim=1, keepdim=True)
            std = logits_biased.std(dim=1, keepdim=True) + 1e-6
            logits_norm = (logits_biased - mean) / std
            logits_scaled = router.router_lambda * logits_norm
            self.logits_post_scaling.append(logits_scaled.cpu().numpy())
            
            # Expert selection distribution
            self.expert_selections.append(top_k_indices.cpu().numpy())
            
            # Bias parameter
            self.bias_values.append(router.bias.detach().cpu().numpy())
    
    def save(self, path):
        """Save all logs to disk"""
        np.savez(
            path,
            steps=np.array(self.steps),
            losses=np.array(self.losses),
            logits_pre_noise=np.array(self.logits_pre_noise),
            logits_post_noise=np.array(self.logits_post_noise),
            logits_post_bias=np.array(self.logits_post_bias),
            logits_post_scaling=np.array(self.logits_post_scaling),
            expert_selections=np.array(self.expert_selections),
            bias_values=np.array(self.bias_values),
        )


def plot_live(logger, save_dir, ablation_name):
    """Live plot: logits, expert dist, bias, loss"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Router Ablation: {ablation_name}', fontsize=16)
    
    steps = logger.steps
    
    # Loss curve
    axes[0, 0].plot(steps, logger.losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Logits distribution over time (last step)
    if len(logger.logits_post_scaling) > 0:
        logits = logger.logits_post_scaling[-1]  # (batch*seq, num_experts)
        axes[0, 1].hist(logits.flatten(), bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Logit Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Logits Distribution (Step {steps[-1]})')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Expert selection distribution (cumulative)
    if len(logger.expert_selections) > 0:
        all_selections = np.concatenate([s.flatten() for s in logger.expert_selections])
        axes[0, 2].hist(all_selections, bins=np.arange(5)-0.5, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Expert ID')
        axes[0, 2].set_ylabel('Selection Count')
        axes[0, 2].set_title('Expert Selection Distribution (Cumulative)')
        axes[0, 2].set_xticks([0, 1, 2, 3])
        axes[0, 2].grid(True, alpha=0.3)
    
    # Bias evolution
    if len(logger.bias_values) > 0:
        bias_arr = np.array(logger.bias_values)  # (steps, num_experts)
        for i in range(bias_arr.shape[1]):
            axes[1, 0].plot(steps, bias_arr[:, i], label=f'Expert {i}', linewidth=2)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Bias Value')
        axes[1, 0].set_title('Bias Parameter Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Logits std over time (measure of noise/scaling effect)
    if len(logger.logits_post_scaling) > 0:
        stds = [np.std(logits) for logits in logger.logits_post_scaling]
        axes[1, 1].plot(steps, stds, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Logits Std Dev')
        axes[1, 1].set_title('Logits Variability Over Time')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Expert selection entropy over time
    if len(logger.expert_selections) > 0:
        entropies = []
        for sel in logger.expert_selections:
            counts = np.bincount(sel.flatten(), minlength=4)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        axes[1, 2].plot(steps, entropies, 'purple', linewidth=2)
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Entropy (nats)')
        axes[1, 2].set_title('Expert Selection Entropy')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{ablation_name}_live.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_ablation(config, ablation_name, update_bias=True, max_steps=30000, log_every=50, plot_every=1000):
    """Train tiny model, log router internals, visualize"""
    save_dir = Path('logs/router_ablation') / ablation_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset + loader (very hard regression task)
    dataset = VeryHardRegressionDataset(hidden_size=config.hidden_size, num_samples=512, seq_len=32)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)  # smaller batch
    
    # Model + optimizer
    model = TinyMoEModel(config, update_bias=update_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # lower LR
    criterion = nn.MSELoss()
    
    logger = RouterLogger()
    
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_name}")
    print(f"Config: noise={config.router_noise}, lambda={config.router_lambda}, bias_update={update_bias}, router={config.router_type}")
    print(f"{'='*60}\n")
    
    step = 0
    converged = False
    best_loss = float('inf')
    patience = 0
    max_patience = 1000
    
    while step < max_steps and not converged:
        for inputs, targets in loader:
            model.train()
            optimizer.zero_grad()
            
            outputs, top_k_indices, norm_weights = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update bias after optimizer step
            model.update_router_bias(top_k_indices)
            
            # Log
            if step % log_every == 0:
                logger.log(step, loss.item(), model.router, inputs, top_k_indices)
                print(f"Step {step:5d} | Loss: {loss.item():.6f}")
            
            # Live plot
            if step % plot_every == 0 and step > 0:
                plot_live(logger, save_dir, ablation_name)
            
            # Check convergence (0.01 loss)
            if loss.item() < 0.01:
                print(f"\n✓ Converged at step {step} (loss < 0.01)")
                converged = True
                break
            
            # Early stopping if stuck
            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                print(f"\n⚠ Early stop at step {step} (no improvement for {max_patience} steps)")
                break
            
            step += 1
            if step >= max_steps:
                break
    
    # Final log + plot
    logger.log(step, loss.item(), model.router, inputs, top_k_indices)
    plot_live(logger, save_dir, ablation_name)
    logger.save(save_dir / 'logs.npz')
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump({
            'ablation': ablation_name,
            'router_type': config.router_type,
            'router_noise': config.router_noise,
            'router_lambda': config.router_lambda,
            'update_bias': update_bias,
            'num_experts': config.num_routed_experts,
            'top_k': config.num_experts_per_tok,
            'hidden_size': config.hidden_size,
            'final_loss': loss.item(),
            'steps': step,
            'converged': converged,
        }, f, indent=2)
    
    print(f"\nSaved logs → {save_dir}")
    return logger


def main():
    """Run ablations: baseline, bias+scaling (λ=1.0, 2.0), bias+scaling+noise"""
    base_config = BiBoConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_routed_experts=4,  # 2 MLP + 2 Conv
        num_experts_per_tok=2,
        intermediate_size=256,
        num_hidden_layers=1,
        vocab_size=100,
        kernel_size=3,
    )
    
    router_types = ['mlp', 'conv']
    noise_levels = [0.0, 0.01, 0.02]
    scaling_factors = [1.0, 2.0]
    
    for router_type in router_types:
        print(f"\n{'#'*80}")
        print(f"# Router Type: {router_type.upper()}")
        print(f"# Experts: 0,1=MLP(SwiGLU), 2,3=CausalConv1D")
        print(f"{'#'*80}")
        
        # Mode 1: Baseline (no bias update, no noise, λ=1.0)
        config = copy.deepcopy(base_config)
        config.router_type = router_type
        config.router_noise = 0.0
        config.router_lambda = 1.0
        train_ablation(config, f'{router_type}_baseline', update_bias=False, max_steps=30000, log_every=50, plot_every=1000)
        
        # Mode 2: Bias + Scaling (test different λ)
        for lambda_val in scaling_factors:
            config = copy.deepcopy(base_config)
            config.router_type = router_type
            config.router_noise = 0.0
            config.router_lambda = lambda_val
            train_ablation(config, f'{router_type}_bias_scaling_lambda_{lambda_val:.1f}', update_bias=True, max_steps=30000, log_every=50, plot_every=1000)
        
        # Mode 3: Bias + Scaling + Noise (λ=1.0, sweep noise)
        for noise in noise_levels:
            if noise == 0.0:
                continue  # already covered in mode 2
            config = copy.deepcopy(base_config)
            config.router_type = router_type
            config.router_noise = noise
            config.router_lambda = 1.0
            train_ablation(config, f'{router_type}_bias_scaling_noise_{noise:.2f}', update_bias=True, max_steps=30000, log_every=50, plot_every=1000)
    
    print("\n" + "="*80)
    print("All ablations complete. Check logs/router_ablation/")
    print("="*80)


if __name__ == '__main__':
    main()
