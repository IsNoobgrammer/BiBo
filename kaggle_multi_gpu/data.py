"""
Generate sorting task data.
Input: random sequence of tokens [0, vocab_size)
Target: same tokens sorted in ascending order

Usage: python kaggle_multi_gpu/data.py
"""
import numpy as np
import yaml
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']


def generate_sorting_data(num_samples, seq_len, vocab_size, seed):
    """
    Generate (input, target) pairs for sorting task.
    Input: random tokens from [0, vocab_size)
    Target: sorted version of input
    
    Stored as [num_samples, 2, seq_len]:
        data[:, 0, :] = input (unsorted)
        data[:, 1, :] = target (sorted)
    """
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, vocab_size, size=(num_samples, seq_len)).astype(np.int64)
    targets = np.sort(inputs, axis=1).astype(np.int64)
    
    data = np.stack([inputs, targets], axis=1)  # [N, 2, seq_len]
    return data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    seq_len = T['seq_len']
    vocab_size = T['vocab_size']
    
    print(f"Task: Sort {seq_len} tokens from vocab [0, {vocab_size})")
    
    print("Generating training data...")
    train_data = generate_sorting_data(T['train_samples'], seq_len, vocab_size, seed=42)
    train_path = os.path.join(DATA_DIR, 'train.npy')
    np.save(train_path, train_data)
    print(f"  Saved: {train_path} | shape={train_data.shape}")
    
    print("Generating validation data...")
    val_data = generate_sorting_data(T['val_samples'], seq_len, vocab_size, seed=123)
    val_path = os.path.join(DATA_DIR, 'val.npy')
    np.save(val_path, val_data)
    print(f"  Saved: {val_path} | shape={val_data.shape}")
    
    # Show example
    print(f"\nExample:")
    print(f"  Input:  {train_data[0, 0, :20]}...")
    print(f"  Target: {train_data[0, 1, :20]}...")
    
    print(f"\n{T['train_samples']} train + {T['val_samples']} val")
    print(f"  Steps/epoch: {T['train_samples'] // T['batch_size']}")
    print(f"  Val steps: {T['val_samples'] // T['batch_size']}")


if __name__ == '__main__':
    main()
