"""
Generate sorting task data with mixed sequence lengths.
No padding. Variable length. Token 2047 = separator.

Format per sample: [unsorted_tokens] [SEP=2047] [sorted_tokens]
Tokens from [0, 2047). Separator = 2047.

3 buckets: seq_len = 64, 128, 256
Full sequence lengths: 129, 257, 513

Usage: python kaggle_multi_gpu/data.py
"""
import numpy as np
import yaml
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']
VOCAB_SIZE = T['vocab_size']  # 2048
SEP_TOKEN = VOCAB_SIZE - 1    # 2047

SEQ_LENS = [64, 128, 256]


def generate_bucket(num_samples, seq_len, rng):
    """
    Generate sorting samples for one bucket.
    Each sample: [unsorted (seq_len)] [SEP] [sorted (seq_len)]
    Total length: 2*seq_len + 1
    
    Returns list of np arrays (each shape [2*seq_len + 1])
    """
    samples = []
    for _ in range(num_samples):
        tokens = rng.integers(0, SEP_TOKEN, size=seq_len)  # [0, 2047)
        sorted_tokens = np.sort(tokens)
        full = np.concatenate([tokens, [SEP_TOKEN], sorted_tokens]).astype(np.int64)
        samples.append(full)
    return samples


def generate_dataset(total_samples, seed):
    """Generate mixed-length dataset, save per-bucket."""
    rng = np.random.default_rng(seed)
    samples_per_bucket = total_samples // len(SEQ_LENS)
    
    all_samples = {}  # bucket_name -> list of samples
    
    for seq_len in SEQ_LENS:
        bucket_name = f'len_{seq_len}'
        samples = generate_bucket(samples_per_bucket, seq_len, rng)
        # Shuffle within bucket
        rng.shuffle(samples)
        all_samples[bucket_name] = samples
        full_len = 2 * seq_len + 1
        print(f"    {bucket_name}: {len(samples)} samples, full_seq={full_len}")
    
    return all_samples


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Task: Sort sequences")
    print(f"  Vocab: [0, {SEP_TOKEN}) | SEP token: {SEP_TOKEN}")
    print(f"  Buckets: {SEQ_LENS} → full lengths: {[2*s+1 for s in SEQ_LENS]}")
    
    print("\nGenerating training data...")
    train_data = generate_dataset(T['train_samples'], seed=42)
    for bucket_name, samples in train_data.items():
        path = os.path.join(DATA_DIR, f'train_{bucket_name}.npy')
        np.save(path, np.array(samples))
        print(f"  Saved: {path}")
    
    print("\nGenerating validation data...")
    val_data = generate_dataset(T['val_samples'], seed=123)
    for bucket_name, samples in val_data.items():
        path = os.path.join(DATA_DIR, f'val_{bucket_name}.npy')
        np.save(path, np.array(samples))
        print(f"  Saved: {path}")
    
    # Save metadata
    meta = {
        'vocab_size': VOCAB_SIZE,
        'sep_token': SEP_TOKEN,
        'seq_lens': SEQ_LENS,
        'full_lens': {f'len_{s}': 2*s+1 for s in SEQ_LENS},
        'train_per_bucket': T['train_samples'] // len(SEQ_LENS),
        'val_per_bucket': T['val_samples'] // len(SEQ_LENS),
    }
    with open(os.path.join(DATA_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Example
    print(f"\nExample (seq_len=64):")
    s = train_data['len_64'][0]
    print(f"  Full: {s[:10]}... [{SEP_TOKEN}] ...{s[-10:]}")
    print(f"  Length: {len(s)}")
    
    print(f"\nDone. Run train.py next.")


if __name__ == '__main__':
    main()
