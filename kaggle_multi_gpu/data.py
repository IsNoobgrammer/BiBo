"""
Generate and save train/val datasets to disk.
Run FIRST before train.

Usage: python kaggle_multi_gpu/data.py
"""
import torch
import numpy as np
import yaml
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']


def generate_sequences(num_samples, seq_len, vocab_size, seed):
    """
    Complex synthetic sequences — polynomial + periodic + skip + XOR patterns.
    ~30% hard positions, ~10% noise.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((num_samples, seq_len), dtype=np.int64)
    
    for s in range(num_samples):
        seq = data[s]
        seq[:5] = rng.integers(0, vocab_size, size=5)
        pattern_weights = rng.dirichlet(np.ones(5))
        period = rng.integers(5, 18)
        delta = rng.integers(1, vocab_size // 10)
        
        for i in range(5, seq_len):
            r = rng.random()
            cumw = np.cumsum(pattern_weights)
            if r < 0.10:
                seq[i] = rng.integers(0, vocab_size)
            elif r < cumw[0]:
                seq[i] = (seq[i-1] * seq[i-2] + seq[i-1] * 3 + 17) % vocab_size
            elif r < cumw[1]:
                seq[i] = (seq[i % period] + i * 7) % vocab_size
            elif r < cumw[2]:
                skip = 5 if i >= 7 else 2
                seq[i] = (seq[i-1] * 11 + seq[i-skip] * 23 + 5) % vocab_size
            elif r < cumw[3]:
                seq[i] = (seq[i-1] + delta) % vocab_size
            else:
                seq[i] = (seq[i-1] ^ seq[i-3]) % vocab_size
    
    return data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Generating training data...")
    train_data = generate_sequences(T['train_samples'], T['seq_len'], T['vocab_size'], seed=42)
    train_path = os.path.join(DATA_DIR, 'train.npy')
    np.save(train_path, train_data)
    print(f"  Saved: {train_path} | shape={train_data.shape}")
    
    print("Generating validation data...")
    val_data = generate_sequences(T['val_samples'], T['seq_len'], T['vocab_size'], seed=123)
    val_path = os.path.join(DATA_DIR, 'val.npy')
    np.save(val_path, val_data)
    print(f"  Saved: {val_path} | shape={val_data.shape}")
    
    print(f"\nDone. {T['train_samples']} train + {T['val_samples']} val sequences.")
    print(f"  Vocab: {T['vocab_size']} | Seq len: {T['seq_len']}")
    print(f"  Steps/epoch: {T['train_samples'] // T['batch_size']}")
    print(f"  Val steps: {T['val_samples'] // T['batch_size']}")


if __name__ == '__main__':
    main()
