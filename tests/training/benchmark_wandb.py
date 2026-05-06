"""
Benchmark BiBo vs Qwen3MoE with wandb logging.
Proper training setup: AdamW + Cosine LR scheduler + warmup.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not installed. Install with: pip install wandb")

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from baseline.qwen3moe.config import Qwen3MoeConfig
from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM


class VeryHardLanguageDataset(Dataset):
    """Very hard language modeling task"""
    def __init__(self, vocab_size=5000, num_samples=1000, seq_len=256):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        torch.manual_seed(42)
        self.sequences = []
        
        for _ in range(num_samples):
            seq = []
            
            # Pattern 1: Arithmetic sequences
            start = torch.randint(0, vocab_size // 4, (1,)).item()
            step = torch.randint(1, 10, (1,)).item()
            for i in range(seq_len // 4):
                seq.append((start + i * step) % vocab_size)
            
            # Pattern 2: Repeating motifs
            motif_len = torch.randint(3, 8, (1,)).item()
            motif = torch.randint(0, vocab_size, (motif_len,)).tolist()
            for _ in range(seq_len // 4 // motif_len):
                seq.extend(motif)
            
            # Pattern 3: Fibonacci-like
            a, b = torch.randint(0, 100, (2,)).tolist()
            for _ in range(seq_len // 4):
                seq.append(a % vocab_size)
                a, b = b, (a + b)
            
            # Pattern 4: Local correlations
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
    """Cosine LR schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def benchmark_model(model, model_name, dataloader, device, num_steps=500, 
                   lr=3e-2, warmup_steps=20, weight_decay=0.01, use_wandb=True):
    """Benchmark with proper training setup"""
    model.train()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # LR Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps,
        min_lr_ratio=0.1
    )
    
    results = {
        'losses': [],
        'lrs': [],
        'forward_times': [],
        'backward_times': [],
        'step_times': [],
    }
    
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_name}")
    print(f"{'='*70}")
    print(f"  Optimizer: AdamW (lr={lr}, wd={weight_decay}, betas=(0.9, 0.95))")
    print(f"  Scheduler: Cosine with warmup ({warmup_steps} steps)")
    print(f"  Total steps: {num_steps}")
    print(f"{'='*70}\n")
    
    step = 0
    epoch = 0
    
    while step < num_steps:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if step >= num_steps:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.time()
            
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.time()
            fwd_time = t1 - t0
            
            # Backward
            optimizer.zero_grad()
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t2 = time.time()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t3 = time.time()
            bwd_time = t3 - t2
            
            # Get current LR
            current_lr = scheduler.get_last_lr()[0]
            
            # Record
            results['losses'].append(loss.item())
            results['lrs'].append(current_lr)
            results['forward_times'].append(fwd_time)
            results['backward_times'].append(bwd_time)
            results['step_times'].append(fwd_time + bwd_time)
            
            # Log to wandb with separate step counters per model
            if use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    f'{model_name}/loss': loss.item(),
                    f'{model_name}/lr': current_lr,
                    f'{model_name}/forward_time_ms': fwd_time * 1000,
                    f'{model_name}/backward_time_ms': bwd_time * 1000,
                    f'{model_name}/step_time_ms': (fwd_time + bwd_time) * 1000,
                    f'{model_name}/step': step,  # Model-specific step counter
                }
                # Also log to comparison metrics (same step axis for both models)
                log_dict.update({
                    'comparison/loss': loss.item(),
                    'comparison/step_time_ms': (fwd_time + bwd_time) * 1000,
                    'comparison/model': model_name,
                    'global_step': step,  # Shared step counter for comparison
                })
                wandb.log(log_dict)
            
            # Print
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1:3d}: loss={loss.item():.4f}, lr={current_lr:.2e}, "
                      f"fwd={fwd_time*1000:.0f}ms, bwd={bwd_time*1000:.0f}ms")
            
            step += 1
        
        epoch += 1
    
    # Compute averages
    results['final_loss'] = results['losses'][-1]
    results['avg_forward_time'] = np.mean(results['forward_times'])
    results['avg_backward_time'] = np.mean(results['backward_times'])
    results['avg_step_time'] = np.mean(results['step_times'])
    
    return results


def main():
    """Run benchmark with wandb logging"""
    print("="*80)
    print("BiBo vs Qwen3MoE Benchmark (wandb logging)")
    print("="*80)
    
    # Check wandb
    if not WANDB_AVAILABLE:
        print("\n⚠ wandb not available. Install with: pip install wandb")
        print("  Continuing without wandb logging...\n")
        use_wandb = False
    else:
        use_wandb = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cpu':
        print("⚠ Running on CPU - this will be slow!")
    
    save_dir = Path('logs/benchmark')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    num_steps = 600
    batch_size = 8
    lr = 1e-3  # High LR
    warmup_steps = 50
    weight_decay = 0.01
    use_compile = False  # torch.compile disabled
    
    # Dataset
    print("\nCreating dataset...")
    dataset = VeryHardLanguageDataset(vocab_size=5000, num_samples=500, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: 127")
    
    # BiBo config
    print("\n" + "-"*80)
    print("Creating BiBo model...")
    bibo_config = BiBoConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        router_type='conv',
        router_noise=0.01,
        router_lambda=1.0,
        kernel_size=3,
        bias_update_factor=0.01,
        bias_update_threshold=1,
        mlp_only_layers=[0, 11],
        use_ssmax=True,
        max_position_embeddings=512,
    )
    bibo_model = BiBoForCausalLM(bibo_config).to(device)
    bibo_params = sum(p.numel() for p in bibo_model.parameters())
    print(f"  Total params: {bibo_params:,}")
    print(f"  Config: 8 experts (1 identity, 1 noise, 1 zero, 1 relu, 2 MLP, 2 Conv)")
    
    # Compile model for speedup
    if use_compile:
        print(f"  Compiling model with torch.compile()...")
        try:
            bibo_model = torch.compile(bibo_model, mode='default', backend='inductor')
            print(f"  ✓ Model compiled successfully")
        except Exception as e:
            print(f"  ⚠ Compilation failed: {str(e)[:100]}")
            print(f"  ℹ To enable torch.compile on Windows, install Visual Studio Build Tools:")
            print(f"    https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022")
            print(f"  Continuing without compilation (slower)...")
            use_compile = False
    
    # Qwen3MoE config
    print("\n" + "-"*80)
    print("Creating Qwen3MoE baseline...")
    qwen_config = Qwen3MoeConfig(
        vocab_size=5000,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        mlp_only_layers=[0, 11],
        max_position_embeddings=512,
    )
    qwen_model = Qwen3MoeForCausalLM(qwen_config).to(device)
    qwen_params = sum(p.numel() for p in qwen_model.parameters())
    print(f"  Total params: {qwen_params:,}")
    print(f"  Config: 6 experts (standard Qwen3MoE)")
    
    # Compile model for speedup
    if use_compile:
        print(f"  Compiling model with torch.compile()...")
        try:
            qwen_model = torch.compile(qwen_model, mode='default', backend='inductor')
            print(f"  ✓ Model compiled successfully")
        except Exception as e:
            print(f"  ⚠ Compilation failed: {str(e)[:100]}")
            print(f"  Continuing without compilation (slower)...")
    
    diff_pct = abs(bibo_params - qwen_params) / qwen_params * 100
    print(f"\nParam difference: {abs(bibo_params - qwen_params):,} ({diff_pct:.2f}%)")
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="bibo-benchmark",
            name=f"bibo_vs_qwen3moe_{int(time.time())}",
            config={
                "num_steps": num_steps,
                "batch_size": batch_size,
                "lr": lr,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "bibo_params": bibo_params,
                "qwen_params": qwen_params,
                "bibo_experts": 8,
                "qwen_experts": 6,
                "device": str(device),
                "use_compile": use_compile,
            }
        )
        
        # Define custom charts for comparison
        wandb.define_metric("global_step")
        wandb.define_metric("BiBo/step")
        wandb.define_metric("Qwen3MoE/step")
        
        # Set x-axis for each model's metrics
        wandb.define_metric("BiBo/*", step_metric="BiBo/step")
        wandb.define_metric("Qwen3MoE/*", step_metric="Qwen3MoE/step")
        
        # Comparison metrics use global_step
        wandb.define_metric("comparison/*", step_metric="global_step")
        
        print(f"\n✓ wandb initialized: {wandb.run.url}")
    
    # Benchmark BiBo
    print("\n" + "="*80)
    print("BENCHMARKING BIBO")
    print("="*80)
    bibo_results = benchmark_model(
        bibo_model, 'BiBo', dataloader, device, 
        num_steps=num_steps, lr=lr, warmup_steps=warmup_steps,
        weight_decay=weight_decay, use_wandb=use_wandb
    )
    
    # Benchmark Qwen3MoE
    print("\n" + "="*80)
    print("BENCHMARKING QWEN3MOE")
    print("="*80)
    qwen_results = benchmark_model(
        qwen_model, 'Qwen3MoE', dataloader, device,
        num_steps=num_steps, lr=lr, warmup_steps=warmup_steps,
        weight_decay=weight_decay, use_wandb=use_wandb
    )
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<35} {'BiBo':<20} {'Qwen3MoE':<20} {'Winner':<10}")
    print("-"*85)
    
    # Parameters
    print(f"{'Total Parameters':<35} {bibo_params:>19,} {qwen_params:>19,} "
          f"{'BiBo' if bibo_params < qwen_params else 'Qwen':<10}")
    
    # Final loss
    print(f"{'Final Loss':<35} {bibo_results['final_loss']:>19.6f} {qwen_results['final_loss']:>19.6f} "
          f"{'BiBo' if bibo_results['final_loss'] < qwen_results['final_loss'] else 'Qwen':<10}")
    
    # Forward time
    bibo_fwd = bibo_results['avg_forward_time'] * 1000
    qwen_fwd = qwen_results['avg_forward_time'] * 1000
    print(f"{'Avg Forward Time (ms)':<35} {bibo_fwd:>19.1f} {qwen_fwd:>19.1f} "
          f"{'BiBo' if bibo_fwd < qwen_fwd else 'Qwen':<10}")
    
    # Backward time
    bibo_bwd = bibo_results['avg_backward_time'] * 1000
    qwen_bwd = qwen_results['avg_backward_time'] * 1000
    print(f"{'Avg Backward Time (ms)':<35} {bibo_bwd:>19.1f} {qwen_bwd:>19.1f} "
          f"{'BiBo' if bibo_bwd < qwen_bwd else 'Qwen':<10}")
    
    # Total step time
    bibo_step = bibo_results['avg_step_time'] * 1000
    qwen_step = qwen_results['avg_step_time'] * 1000
    print(f"{'Avg Step Time (ms)':<35} {bibo_step:>19.1f} {qwen_step:>19.1f} "
          f"{'BiBo' if bibo_step < qwen_step else 'Qwen':<10}")
    
    # Speedup
    speedup = qwen_step / bibo_step if bibo_step > 0 else 0
    print(f"{'Speedup (Qwen/BiBo)':<35} {speedup:>19.2f}x {'':>19} "
          f"{'BiBo' if speedup > 1 else 'Qwen':<10}")
    
    print("="*85)
    
    # Save results
    results = {
        'bibo': {
            'params': bibo_params,
            'final_loss': bibo_results['final_loss'],
            'avg_forward_time_ms': bibo_fwd,
            'avg_backward_time_ms': bibo_bwd,
            'avg_step_time_ms': bibo_step,
        },
        'qwen': {
            'params': qwen_params,
            'final_loss': qwen_results['final_loss'],
            'avg_forward_time_ms': qwen_fwd,
            'avg_backward_time_ms': qwen_bwd,
            'avg_step_time_ms': qwen_step,
        },
        'speedup': speedup,
    }
    
    with open(save_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved → {save_dir / 'benchmark_results.json'}")
    
    if use_wandb:
        # Log final summary to wandb
        wandb.log({
            'summary/bibo_final_loss': bibo_results['final_loss'],
            'summary/qwen_final_loss': qwen_results['final_loss'],
            'summary/bibo_avg_step_time_ms': bibo_step,
            'summary/qwen_avg_step_time_ms': qwen_step,
            'summary/speedup': speedup,
        })
        
        # Create comparison table
        comparison_table = wandb.Table(
            columns=["step", "BiBo_loss", "Qwen3MoE_loss"],
            data=[[i, bibo_results['losses'][i], qwen_results['losses'][i]] 
                  for i in range(min(len(bibo_results['losses']), len(qwen_results['losses'])))]
        )
        wandb.log({"comparison_table": comparison_table})
        
        # Create line plot comparing losses
        wandb.log({
            "loss_comparison": wandb.plot.line_series(
                xs=list(range(num_steps)),
                ys=[bibo_results['losses'], qwen_results['losses']],
                keys=["BiBo", "Qwen3MoE"],
                title="Training Loss Comparison",
                xname="Step"
            )
        })
        
        # Create line plot comparing step times
        wandb.log({
            "step_time_comparison": wandb.plot.line_series(
                xs=list(range(num_steps)),
                ys=[[t * 1000 for t in bibo_results['step_times']], 
                    [t * 1000 for t in qwen_results['step_times']]],
                keys=["BiBo", "Qwen3MoE"],
                title="Step Time Comparison (ms)",
                xname="Step"
            )
        })
        
        print(f"\n✓ View results at: {wandb.run.url}")
        wandb.finish()
    
    print("\n" + "="*80)
    print("✓ Benchmark complete!")
    print("="*80)


if __name__ == '__main__':
    main()
