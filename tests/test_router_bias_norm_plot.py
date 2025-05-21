import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
from typing import List

# Add BiBo root directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration_bibo import BiBoConfig
from modeling_bibo import BiBoMoERouter


def plot_router_distributions(distributions: List[np.ndarray], labels: List[str], baseline: float, title: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for dist, label in zip(distributions, labels):
        plt.plot(dist, marker='o', label=label)
    plt.axhline(baseline, color='r', linestyle='--', label='Uniform Baseline')
    plt.xlabel('Expert Index')
    plt.ylabel('Routing Probability')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_router_and_plot(router_type: str, router_lambda: float, bias_update_threshold: int, num_experts: int = 4, 
                          num_tokens: int = 20000, batch_size: int = 32, seq_len: int = 8, hidden_size: int = 16, 
                          plot_every: int = 2000, device: str = 'cpu'):
    # Minimal BiBoConfig
    config = BiBoConfig(
        hidden_size=hidden_size,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_routed_experts=num_experts,
        num_experts_per_tok=1,
        bias_update_factor=1e-2,
        bias_update_threshold=bias_update_threshold,
        router_lambda=router_lambda,
        router_noise=0.0,
        router_type=router_type,
        kernel_size=3,
        max_position_embeddings=seq_len
    )
    router = BiBoMoERouter(config).to(device)
    router.train()
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-2)

    # Track distributions
    snapshots = []
    snapshot_steps = []
    uniform_baseline = 1.0 / num_experts

    for step in range(0, num_tokens, batch_size * seq_len):
        # Generate random input (simulate token batches)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        # Forward pass
        topk_idx, norm_weights = router(x)
        # For visualization: average routing weights over batch and seq
        if hasattr(router, 'gate_proj'):
            logits = router.gate_proj(x.view(-1, hidden_size)).float() + router.bias
        else:
            x_perm = x.permute(0, 2, 1)
            x_padded = F.pad(x_perm, (router.causal_padding, 0))
            conv_out = router.gate_conv(x_padded)
            logits = conv_out.permute(0, 2, 1).reshape(-1, num_experts).float() + router.bias
        # Apply normalization and scaling
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True) + 1e-6
        logits_norm = (logits - mean) / std
        logits_scaled = router.router_lambda * logits_norm
        probs = F.softmax(logits_scaled, dim=1)
        avg_probs = probs.mean(dim=0).detach().cpu().numpy()
        if step % plot_every == 0 or step == (num_tokens - batch_size * seq_len):
            with torch.no_grad():
                snapshots.append(avg_probs.copy())
                snapshot_steps.append(step)
        # Dummy loss: encourage uniform routing
        loss = ((probs - uniform_baseline) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Plot snapshots
    for idx, (dist, s) in enumerate(zip(snapshots, snapshot_steps)):
        save_path = os.path.join('visuals', f'router_{router_type}_lambda{router_lambda}_biasthresh{bias_update_threshold}_step{s}.png')
        plot_router_distributions(
            [dist],
            [f'{router_type} lambda={router_lambda} Step={s}'],
            baseline=uniform_baseline,
            title=f'Router Distribution ({router_type}, lambda={router_lambda}, bias_thresh={bias_update_threshold}) Step={s}',
            save_path=save_path
        )
    print(f"Saved plots for {router_type} router with lambda={router_lambda} bias_update_threshold={bias_update_threshold}")


def main():
    num_experts = 4
    settings = [
        # (router_type, router_lambda, bias_update_threshold)
        ("mlp", 1.0, 256),
        ("mlp", 2.0, 256),
        ("mlp", 1.0, 32),
        ("conv", 1.0, 256),
        ("conv", 2.0, 256),
        ("conv", 1.0, 32),
    ]
    for router_type, router_lambda, bias_update_threshold in settings:
        train_router_and_plot(
            router_type=router_type,
            router_lambda=router_lambda,
            bias_update_threshold=bias_update_threshold,
            num_experts=num_experts,
            num_tokens=20000,
            batch_size=64,
            seq_len=32,
            hidden_size=32,
            plot_every=40,
            device='cpu'
        )
    print("All router bias/norm/λ experiments complete. Check the generated PNG files.")


if __name__ == "__main__":
    main()
