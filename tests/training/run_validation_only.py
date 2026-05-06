"""
Run validation only from saved checkpoints
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
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


def compute_attention_entropy(model, dataloader, device, max_batches=10):
    """Compute avg attention entropy"""
    model.eval()
    entropies = []
    max_probs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            inputs = inputs.to(device)
            outputs = model(input_ids=inputs, output_attentions=True)
            
            if outputs.attentions is None:
                continue
            
            for layer_attn in outputs.attentions:
                if layer_attn is None:
                    continue
                    
                attn_probs = torch.clamp(layer_attn, min=1e-10)
                entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean().item()
                entropies.append(entropy)
                
                max_prob = attn_probs.max(dim=-1)[0].mean().item()
                max_probs.append(max_prob)
    
    return {
        'entropy': np.mean(entropies) if entropies else 0.0,
        'max_prob': np.mean(max_probs) if max_probs else 0.0,
    }


def evaluate_on_seq_lengths(model, model_name, vocab_size, device, seq_lens=[64, 128, 256], 
                            batch_size=8, num_samples=200):
    """Evaluate model on different seq lengths"""
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} on varying seq lengths")
    print(f"{'='*70}\n")
    
    for seq_len in seq_lens:
        print(f"Seq len: {seq_len}")
        
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


print("="*80)
print("Running Validation from Checkpoints")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

save_dir = Path('logs/long_context')

# Load checkpoints
print("\nLoading checkpoints...")
bibo_ckpt = torch.load(save_dir / 'bibo_checkpoint.pt', map_location=device, weights_only=False)
qwen_ckpt = torch.load(save_dir / 'qwen_checkpoint.pt', map_location=device, weights_only=False)

bibo_model = BiBoForCausalLM(bibo_ckpt['config']).to(device)
bibo_model.load_state_dict(bibo_ckpt['model_state_dict'])
print(f"✓ BiBo loaded (trained {bibo_ckpt['steps']} steps)")

qwen_model = Qwen3MoeForCausalLM(qwen_ckpt['config']).to(device)
qwen_model.load_state_dict(qwen_ckpt['model_state_dict'])
print(f"✓ Qwen loaded (trained {qwen_ckpt['steps']} steps)")

# Run validation
seq_lens = [64, 128, 256]
vocab_size = 2000

bibo_results = evaluate_on_seq_lengths(
    bibo_model, 'BiBo+SSMax', vocab_size, device, seq_lens=seq_lens
)

qwen_results = evaluate_on_seq_lengths(
    qwen_model, 'Qwen3MoE', vocab_size, device, seq_lens=seq_lens
)

# Summary
print("\n" + "="*80)
print("VALIDATION RESULTS")
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

# Save results
results = {
    'bibo': bibo_results,
    'qwen': qwen_results,
}

with open(save_dir / 'validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved → {save_dir / 'validation_results.json'}")
print("\n" + "="*80)
print("✓ Validation complete!")
print("="*80)
