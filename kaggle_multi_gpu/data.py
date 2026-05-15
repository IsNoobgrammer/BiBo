"""
Generate sorting task data with mixed sequence lengths.
Creates 3 datasets (seq_len=64, 128, 256), pads to max_len, shuffles and merges.

Input: random tokens [1, vocab_size)  (0 reserved for padding)
Target: sorted version
Padded to max_seq_len with 0s.

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
MAX_SEQ_LEN = T['seq_len']  # 256
VOCAB_SIZE = T['vocab_size']  # 2048
PAD_ID = T.get('pad_token_id', 0)

SEQ_LENS = [64, 128, 256]


def generate_sorting_data(num_samples, seq_len, vocab_size, max_seq_len, rng):
    """
    Generate sorting pairs padded to max_seq_len.
    Tokens from [1, vocab_size) — 0 is pad.
    
    Returns: [num_samples, 2, max_seq_len]
        data[:, 0, :] = input (unsorted, padded)
        data[:, 1, :] = target (sorted, padded)
    """
    data = np.zeros((num_samples, 2, max_seq_len), dtype=np.int64)
    
    for i in range(num_samples):
        # Random tokens from [1, vocab_size) — avoid 0 (pad)
        tokens = rng.integers(1, vocab_size, size=seq_len)
        sorted_tokens = np.sort(tokens)
        
        data[i, 0, :seq_len] = tokens
        data[i, 1, :seq_len] = sorted_tokens
        # Rest stays 0 (pad)
    
    return data


def generate_mixed_dataset(total_samples, seed):
    """Generate equal parts of seq_len=64, 128, 256, shuffle together."""
    rng = np.random.default_rng(seed)
    samples_per_len = total_samples // len(SEQ_LENS)
    
    all_data = []
    all_lengths = []
    
    for seq_len in SEQ_LENS:
        data = generate_sorting_data(samples_per_len, seq_len, VOCAB_SIZE, MAX_SEQ_LEN, rng)
        lengths = np.full(samples_per_len, seq_len, dtype=np.int64)
        all_data.append(data)
        all_lengths.append(lengths)
        print(f"    seq_len={seq_len}: {samples_per_len} samples")
    
    # Concat and shuffle
    all_data = np.concatenate(all_data, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)
    
    shuffle_idx = rng.permutation(len(all_data))
    all_data = all_data[shuffle_idx]
    all_lengths = all_lengths[shuffle_idx]
    
    return all_data, all_lengths


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Task: Sort sequences (mixed lengths: {SEQ_LENS})")
    print(f"  Vocab: [1, {VOCAB_SIZE}) | Pad: {PAD_ID} | Max len: {MAX_SEQ_LEN}")
    
    print("\nGenerating training data...")
    train_data, train_lengths = generate_mixed_dataset(T['train_samples'], seed=42)
    np.save(os.path.join(DATA_DIR, 'train.npy'), train_data)
    np.save(os.path.join(DATA_DIR, 'train_lengths.npy'), train_lengths)
    print(f"  Saved: train.npy shape={train_data.shape}")
    
    print("\nGenerating validation data...")
    val_data, val_lengths = generate_mixed_dataset(T['val_samples'], seed=123)
    np.save(os.path.join(DATA_DIR, 'val.npy'), val_data)
    np.save(os.path.join(DATA_DIR, 'val_lengths.npy'), val_lengths)
    print(f"  Saved: val.npy shape={val_data.shape}")
    
    # Example
    print(f"\nExample (seq_len={train_lengths[0]}):")
    sl = train_lengths[0]
    print(f"  Input:  {train_data[0, 0, :min(20, sl)]}...")
    print(f"  Target: {train_data[0, 1, :min(20, sl)]}...")
    
    print(f"\nTotal: {T['train_samples']} train + {T['val_samples']} val")
    print(f"  Steps/epoch: {T['train_samples'] // T['batch_size']}")


if __name__ == '__main__':
    main()
