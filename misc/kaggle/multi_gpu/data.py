"""
Generate sorting task data with curriculum learning support.

Format per sample: [unsorted_tokens] [SEP] [sorted_tokens]
Tokens from [0, SEP_TOKEN). Separator = vocab_size - 1.

Curriculum stages: [2, 8, 32, 64, 128, 256]
Each stage gets equal share of total samples.

Usage:
    python misc/kaggle/multi_gpu/data.py           # seed from config (48)
    python misc/kaggle/multi_gpu/data.py --seed 69
"""
import numpy as np
import yaml
import os
import json
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

T = CFG['training']
VOCAB_SIZE = T['vocab_size']
SEP_TOKEN = VOCAB_SIZE - 1

# Use curriculum_stages for bucket sizes (regardless of curriculum mode)
if T.get('curriculum_stages'):
    SEQ_LENS = T['curriculum_stages']
else:
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
        tokens = rng.integers(0, SEP_TOKEN, size=seq_len)  # [0, SEP_TOKEN)
        sorted_tokens = np.sort(tokens)
        full = np.concatenate([tokens, [SEP_TOKEN], sorted_tokens]).astype(np.int64)
        samples.append(full)
    return samples


def generate_val_deduped(val_target, seq_len, rng, train_set):
    """
    Generate val samples ensuring NO overlap with train.
    """
    val_samples = []
    attempts = 0
    max_attempts = val_target * 5

    while len(val_samples) < val_target and attempts < max_attempts:
        tokens = rng.integers(0, SEP_TOKEN, size=seq_len)
        sorted_tokens = np.sort(tokens)
        full = np.concatenate([tokens, [SEP_TOKEN], sorted_tokens]).astype(np.int64)

        sample_hash = full.tobytes()
        if sample_hash not in train_set:
            val_samples.append(full)
        attempts += 1

    if len(val_samples) < val_target:
        print(f"    WARNING: only got {len(val_samples)}/{val_target} unique val samples after {max_attempts} attempts")

    return val_samples


def main():
    parser = argparse.ArgumentParser(description='Generate sorting task data')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config.yaml, default: config value)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else T['seed']

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Task: Sort sequences")
    print(f"  Seed: {seed}")
    print(f"  Vocab: [0, {SEP_TOKEN}) | SEP token: {SEP_TOKEN}")
    print(f"  Buckets: {SEQ_LENS} → full lengths: {[2*s+1 for s in SEQ_LENS]}")
    print(f"  Total train samples: {T['train_samples']}")

    # === Generate training data ===
    print("\nGenerating training data...")
    train_rng = np.random.default_rng(seed)
    samples_per_bucket = T['train_samples'] // len(SEQ_LENS)

    train_data = {}
    train_hashes = {}

    for seq_len in SEQ_LENS:
        bucket_name = f'len_{seq_len}'
        samples = generate_bucket(samples_per_bucket, seq_len, train_rng)
        train_rng.shuffle(samples)
        train_data[bucket_name] = samples
        train_hashes[seq_len] = set(s.tobytes() for s in samples)
        full_len = 2 * seq_len + 1
        print(f"    {bucket_name}: {len(samples)} samples, full_seq={full_len}")

    for bucket_name, samples in train_data.items():
        path = os.path.join(DATA_DIR, f'train_{bucket_name}.npy')
        np.save(path, np.array(samples))
        print(f"  Saved: {path}")

    # === Generate validation data (deduplicated from train) ===
    print("\nGenerating validation data (deduped from train)...")
    val_rng = np.random.default_rng(seed + 1000)
    val_per_bucket = T['val_samples'] // len(SEQ_LENS)

    for seq_len in SEQ_LENS:
        bucket_name = f'len_{seq_len}'
        val_samples = generate_val_deduped(val_per_bucket, seq_len, val_rng, train_hashes[seq_len])
        val_rng.shuffle(val_samples)

        path = os.path.join(DATA_DIR, f'val_{bucket_name}.npy')
        np.save(path, np.array(val_samples))
        print(f"  Saved: {path} ({len(val_samples)} unique samples, 0 overlap with train)")

    # Save metadata
    meta = {
        'vocab_size': VOCAB_SIZE,
        'sep_token': SEP_TOKEN,
        'seq_lens': SEQ_LENS,
        'full_lens': {f'len_{s}': 2*s+1 for s in SEQ_LENS},
        'train_per_bucket': samples_per_bucket,
        'val_per_bucket': val_per_bucket,
        'val_deduped': True,
        'curriculum': T.get('curriculum', False),
    }
    with open(os.path.join(DATA_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Example
    first_bucket = SEQ_LENS[0]
    print(f"\nExample (seq_len={first_bucket}):")
    s = train_data[f'len_{first_bucket}'][0]
    print(f"  Full: {s.tolist()}")
    print(f"  Length: {len(s)}")

    print(f"\nDone. Run train.py next.")


if __name__ == '__main__':
    main()
