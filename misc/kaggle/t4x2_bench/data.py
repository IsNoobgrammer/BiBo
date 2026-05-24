"""
Generate sorting task data for T4x2 benchmark.

Task: given a sequence of random integers, sort them.
Input:  [3, 1, 4, 1, 5, 9, 2, 6] SEP [1, 1, 2, 3, 4, 5, 6, 9]
Target: the sorted sequence (model learns to predict next token)

Format: tokens 0-250 = values, 251 = SEP, 252-255 = padding
"""
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def generate_sorting_data(num_samples, seq_len, vocab_size, seed=42):
    """
    Generate sorting task data.
    
    Each sample: [random_values] SEP [sorted_values]
    Total length = seq_len (split evenly: half input, half output)
    """
    rng = np.random.RandomState(seed)
    sep_token = vocab_size - 5  # 251 for vocab_size=256
    half_len = seq_len // 2
    
    # Generate random sequences and their sorted versions
    inputs = rng.randint(0, sep_token, size=(num_samples, half_len))
    sorted_inputs = np.sort(inputs, axis=1)
    
    # Build full sequences: [input] SEP [sorted]
    data = np.zeros((num_samples, seq_len), dtype=np.int64)
    data[:, :half_len] = inputs
    data[:, half_len] = sep_token
    data[:, half_len + 1:] = sorted_inputs[:, :-1]  # shift by 1 for next-token prediction
    
    return data


def main():
    import yaml
    
    config_path = os.path.join(BASE_DIR, 'config.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    T = cfg['training']
    seq_len = T['seq_len']
    vocab_size = T['vocab_size']
    num_train = T['num_train_samples']
    num_val = T['num_val_samples']
    seed = T['seed']
    
    print(f"Generating sorting data...")
    print(f"  Train samples: {num_train}")
    print(f"  Val samples: {num_val}")
    print(f"  Seq length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    
    # Generate
    train_data = generate_sorting_data(num_train, seq_len, vocab_size, seed)
    val_data = generate_sorting_data(num_val, seq_len, vocab_size, seed + 1)
    
    # Save as numpy arrays
    train_path = os.path.join(DATA_DIR, 'train_sort.npy')
    val_path = os.path.join(DATA_DIR, 'val_sort.npy')
    
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    
    print(f"\nSaved:")
    print(f"  {train_path} ({train_data.nbytes / 1e6:.1f} MB)")
    print(f"  {val_path} ({val_data.nbytes / 1e6:.1f} MB)")
    
    # Verify
    sample = train_data[0]
    half = seq_len // 2
    print(f"\nSample 0:")
    print(f"  Input:  {sample[:half]}")
    print(f"  SEP:    {sample[half]}")
    print(f"  Target: {sample[half+1:]}")


if __name__ == '__main__':
    main()
