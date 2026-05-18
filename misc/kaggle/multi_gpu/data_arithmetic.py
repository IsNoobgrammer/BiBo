"""
Generate arithmetic task data — multi-phase chain-of-thought.

Token encoding:
  - Numbers 1..max_num map to tokens 1..max_num (token 0 = PAD)
  - Special tokens (from end of vocab):
      vocab[-1] = SEP   (phase separator)
      vocab[-2] = ADD   (+)
      vocab[-3] = SUB   (-)
      vocab[-4] = MUL   (*)
      vocab[-5] = DIV   (//, floor division)

Sample format (3 phases):
  Phase 1 (input):  raw expression, e.g. [3, MUL, 4, ADD, 2, SUB, 6, DIV, 3]
  Phase 2 (intermediate): precedence resolved (mul/div computed), e.g. [12, ADD, 2, SUB, 2]
  Phase 3 (answer): final scalar, e.g. [12]

Full sequence: [phase1] [SEP] [phase2] [SEP] [phase3]

Labels: only predict from first SEP onward (phase2 + phase3 are targets).

Usage:
    python misc/kaggle/multi_gpu/data_arithmetic.py
    python misc/kaggle/multi_gpu/data_arithmetic.py --seed 69
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
ARITH = T['arithmetic']
VOCAB_SIZE = T['vocab_size']

# Special tokens (from end of vocab)
SEP_TOKEN = VOCAB_SIZE - 1      # 511
ADD_TOKEN = VOCAB_SIZE - 2      # 510
SUB_TOKEN = VOCAB_SIZE - 3      # 509
MUL_TOKEN = VOCAB_SIZE - 4      # 508
DIV_TOKEN = VOCAB_SIZE - 5      # 507

OP_MAP = {
    '+': ADD_TOKEN,
    '-': SUB_TOKEN,
    '*': MUL_TOKEN,
    '//': DIV_TOKEN,
}
OP_REVERSE = {v: k for k, v in OP_MAP.items()}

MAX_NUM = ARITH['max_num']  # numbers in [1, max_num]

# Validate: max_num must fit in vocab (tokens 1..max_num, special tokens at end)
assert MAX_NUM < DIV_TOKEN, f"max_num={MAX_NUM} too large for vocab_size={VOCAB_SIZE} (DIV_TOKEN={DIV_TOKEN})"


def num_to_token(n):
    """Convert a number to its token. Handles negatives by clamping to valid range."""
    # We allow results outside [1, max_num] — they just use their integer value as token
    # But must be in [1, DIV_TOKEN-1] to not collide with special tokens
    # For negative results or zero, we offset: store as (n + offset) where offset ensures positivity
    # Actually simpler: we'll restrict generation to avoid negative/zero results
    return int(n)


def generate_expression(num_terms, rng, operators):
    """
    Generate a random arithmetic expression with num_terms operands.
    Constructs expressions that are guaranteed to have valid intermediate results.
    
    Strategy:
    - Pick operators first
    - For consecutive * and // groups: construct numbers that produce clean results
    - For + and -: any numbers work
    - For longer expressions: use smaller multipliers to avoid overflow
    
    Returns:
        numbers: list of int (operands)
        ops: list of str (operators between operands)
    """
    ops = []
    for i in range(num_terms - 1):
        op = operators[int(rng.integers(0, len(operators)))]
        ops.append(op)
    
    # For longer expressions, limit multiplier size to prevent overflow
    max_mult = max(2, min(9, 500 // max(num_terms, 1)))
    
    # Group consecutive * and // operations (these get resolved together in precedence)
    # For each group, construct numbers that produce valid results
    numbers = [None] * num_terms
    
    # Find mul/div groups and fill numbers constructively
    i = 0
    while i < num_terms:
        # Find the extent of this mul/div group
        group_start = i
        while i < num_terms - 1 and ops[i] in ('*', '//'):
            i += 1
        group_end = i  # last index in group
        
        if group_end > group_start:
            # This is a mul/div group: numbers[group_start] through numbers[group_end]
            # Generate constructively — start small to leave room
            current = int(rng.integers(2, min(50, MAX_NUM // 2)))
            numbers[group_start] = current
            
            for j in range(group_start, group_end):
                op = ops[j]
                if op == '*':
                    mult = int(rng.integers(2, max_mult + 1))
                    numbers[j + 1] = mult
                    current = current * mult
                elif op == '//':
                    # Pick a divisor that divides current evenly
                    divisors = [d for d in range(2, min(abs(current) + 1, 11)) if current % d == 0]
                    if divisors:
                        div = divisors[int(rng.integers(0, len(divisors)))]
                    else:
                        # Make current even then divide by 2
                        if current % 2 != 0:
                            current += 1
                            numbers[group_start] += 1
                        div = 2
                    numbers[j + 1] = div
                    current = current // div
        else:
            # Single number (surrounded by + or - or at edges)
            # For longer expressions, use smaller numbers to keep final result in range
            num_max = min(MAX_NUM, max(20, 500 // max(num_terms // 3, 1)))
            numbers[i] = int(rng.integers(1, num_max + 1))
        
        i += 1
    
    # Fill any remaining None (safety)
    for i in range(num_terms):
        if numbers[i] is None:
            numbers[i] = int(rng.integers(1, MAX_NUM + 1))
    
    return numbers, ops


def resolve_precedence(numbers, ops):
    """
    Phase 2: resolve * and // first (left to right), leave + and - untouched.
    
    Returns:
        new_numbers: list of int (after mul/div resolved)
        new_ops: list of str (only + and -)
    
    Returns None if result goes out of bounds.
    """
    # Work left to right, collapsing * and // 
    new_numbers = [numbers[0]]
    new_ops = []
    
    for i, op in enumerate(ops):
        if op == '*':
            result = new_numbers[-1] * numbers[i + 1]
            if result > 50000 or result < -50000:  # overflow guard
                return None, None
            new_numbers[-1] = result
        elif op == '//':
            divisor = numbers[i + 1]
            if divisor == 0:
                return None, None
            dividend = new_numbers[-1]
            result = dividend // divisor
            new_numbers[-1] = result
        else:
            # + or - : keep as-is
            new_numbers.append(numbers[i + 1])
            new_ops.append(op)
    
    return new_numbers, new_ops


def compute_final(numbers, ops):
    """
    Phase 3: compute final answer from + and - only (left to right).
    
    Returns:
        result: int (final answer)
    """
    result = numbers[0]
    for i, op in enumerate(ops):
        if op == '+':
            result += numbers[i + 1]
        elif op == '-':
            result -= numbers[i + 1]
    return result


def encode_expression(numbers, ops):
    """Encode numbers and operators as token sequence."""
    tokens = []
    for i, n in enumerate(numbers):
        tokens.append(n)
        if i < len(ops):
            tokens.append(OP_MAP[ops[i]])
    return tokens


def encode_number(n):
    """
    Encode a number as tokens. 
    For numbers in [1, max_num]: single token.
    For larger numbers or negatives: we need multi-token encoding.
    
    Strategy: use a sign token + magnitude, or restrict to single-token range.
    
    For simplicity and to keep it learnable: we restrict ALL intermediate and final
    results to fit in [1, DIV_TOKEN-1]. If they don't, we reject the sample.
    """
    return [int(n)]


def is_valid_result(n):
    """Check if a number can be represented as a single token."""
    return 1 <= n <= (DIV_TOKEN - 1)


def generate_sample(rng, min_terms, max_terms, operators):
    """
    Generate one valid arithmetic sample.
    
    Returns token sequence: [phase1] [SEP] [phase2] [SEP] [phase3]
    Or None if sample is invalid (out of bounds, bad division, etc.)
    """
    num_terms = int(rng.integers(min_terms, max_terms + 1))
    numbers, ops = generate_expression(num_terms, rng, operators)
    
    # Phase 2: resolve precedence
    p2_numbers, p2_ops = resolve_precedence(numbers, ops)
    if p2_numbers is None:
        return None
    
    # Check all phase 2 numbers are valid tokens
    for n in p2_numbers:
        if not is_valid_result(n):
            return None
    
    # Phase 3: compute final
    final = compute_final(p2_numbers, p2_ops)
    if not is_valid_result(final):
        return None
    
    # Encode
    phase1_tokens = encode_expression(numbers, ops)
    phase2_tokens = encode_expression(p2_numbers, p2_ops)
    phase3_tokens = [final]
    
    # Full sequence
    full = phase1_tokens + [SEP_TOKEN] + phase2_tokens + [SEP_TOKEN] + phase3_tokens
    
    return np.array(full, dtype=np.int64)


def generate_bucket(num_samples, min_terms, max_terms, rng, operators):
    """Generate samples for one difficulty bucket."""
    samples = []
    attempts = 0
    max_attempts = num_samples * 50  # higher multiplier for harder buckets
    
    while len(samples) < num_samples and attempts < max_attempts:
        sample = generate_sample(rng, min_terms, max_terms, operators)
        if sample is not None:
            samples.append(sample)
        attempts += 1
    
    if len(samples) < num_samples:
        print(f"    WARNING: only generated {len(samples)}/{num_samples} valid samples "
              f"(terms={min_terms}-{max_terms}, {attempts} attempts, "
              f"acceptance={len(samples)/attempts*100:.1f}%)")
    
    return samples


def generate_val_deduped(val_target, min_terms, max_terms, rng, operators, train_hashes):
    """Generate validation samples with no overlap to training."""
    val_samples = []
    attempts = 0
    max_attempts = val_target * 30
    
    while len(val_samples) < val_target and attempts < max_attempts:
        sample = generate_sample(rng, min_terms, max_terms, operators)
        if sample is not None:
            h = sample.tobytes()
            if h not in train_hashes:
                val_samples.append(sample)
        attempts += 1
    
    if len(val_samples) < val_target:
        print(f"    WARNING: only got {len(val_samples)}/{val_target} unique val samples")
    
    return val_samples


def pad_samples(samples, pad_token=0):
    """Pad variable-length samples to max length in the batch."""
    max_len = max(len(s) for s in samples)
    padded = np.full((len(samples), max_len), pad_token, dtype=np.int64)
    lengths = np.zeros(len(samples), dtype=np.int64)
    for i, s in enumerate(samples):
        padded[i, :len(s)] = s
        lengths[i] = len(s)
    return padded, lengths


def decode_sample(tokens):
    """Decode a token sequence back to human-readable string (for debugging)."""
    parts = []
    for t in tokens:
        if t == SEP_TOKEN:
            parts.append('SEP')
        elif t == ADD_TOKEN:
            parts.append('+')
        elif t == SUB_TOKEN:
            parts.append('-')
        elif t == MUL_TOKEN:
            parts.append('*')
        elif t == DIV_TOKEN:
            parts.append('//')
        elif t == 0:
            parts.append('<PAD>')
        else:
            parts.append(str(t))
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Generate arithmetic task data')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config.yaml)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else T['seed']
    os.makedirs(DATA_DIR, exist_ok=True)

    operators = ARITH['operators']
    buckets = ARITH['buckets']
    ood_buckets = ARITH.get('ood_buckets', [])
    
    print(f"Task: Arithmetic (multi-phase chain-of-thought)")
    print(f"  Seed: {seed}")
    print(f"  Vocab: {VOCAB_SIZE} | Numbers: [1, {MAX_NUM}]")
    print(f"  Special tokens: SEP={SEP_TOKEN}, +={ADD_TOKEN}, -={SUB_TOKEN}, *={MUL_TOKEN}, //={DIV_TOKEN}")
    print(f"  Operators: {operators}")
    print(f"  Train buckets: {buckets}")
    print(f"  OOD val buckets: {ood_buckets}")
    print(f"  Total train samples: {T['train_samples']}")
    print(f"  Total val samples: {T['val_samples']}")

    # === Generate training data ===
    print("\nGenerating training data...")
    train_rng = np.random.default_rng(seed)
    samples_per_bucket = T['train_samples'] // len(buckets)

    train_data = {}
    train_hashes = {}

    for bucket_idx, (min_t, max_t) in enumerate(buckets):
        bucket_name = f'arith_{min_t}_{max_t}'
        samples = generate_bucket(samples_per_bucket, min_t, max_t, train_rng, operators)
        train_rng.shuffle(samples)
        train_data[bucket_name] = samples
        train_hashes[bucket_name] = set(s.tobytes() for s in samples)
        
        # Stats
        lengths = [len(s) for s in samples]
        print(f"    {bucket_name}: {len(samples)} samples, "
              f"seq_len range=[{min(lengths)}, {max(lengths)}], "
              f"mean={np.mean(lengths):.1f}")

    # Save (padded per bucket)
    for bucket_name, samples in train_data.items():
        padded, lengths = pad_samples(samples)
        path = os.path.join(DATA_DIR, f'train_{bucket_name}.npy')
        np.save(path, padded)
        len_path = os.path.join(DATA_DIR, f'train_{bucket_name}_lengths.npy')
        np.save(len_path, lengths)
        print(f"  Saved: {path} (shape={padded.shape})")

    # === Generate validation data (in-distribution, deduped from train) ===
    print("\nGenerating validation data (in-distribution, deduped from train)...")
    val_rng = np.random.default_rng(seed + 1000)
    val_per_bucket = T['val_samples'] // len(buckets)

    for bucket_idx, (min_t, max_t) in enumerate(buckets):
        bucket_name = f'arith_{min_t}_{max_t}'
        val_samples = generate_val_deduped(
            val_per_bucket, min_t, max_t, val_rng, operators, train_hashes[bucket_name]
        )
        val_rng.shuffle(val_samples)
        
        padded, lengths = pad_samples(val_samples)
        path = os.path.join(DATA_DIR, f'val_{bucket_name}.npy')
        np.save(path, padded)
        len_path = os.path.join(DATA_DIR, f'val_{bucket_name}_lengths.npy')
        np.save(len_path, lengths)
        print(f"  Saved: {path} ({len(val_samples)} unique samples)")

    # === Generate OOD validation data (out-of-distribution — generalization test) ===
    if ood_buckets:
        print("\nGenerating OOD validation data (generalization test)...")
        ood_rng = np.random.default_rng(seed + 2000)
        ood_per_bucket = max(200, T['val_samples'] // (len(ood_buckets) * 2))
        
        for bucket_idx, (min_t, max_t) in enumerate(ood_buckets):
            bucket_name = f'arith_{min_t}_{max_t}'
            # OOD: no dedup needed (these ranges don't overlap with train)
            samples = generate_bucket(ood_per_bucket, min_t, max_t, ood_rng, operators)
            ood_rng.shuffle(samples)
            
            if samples:
                padded, lengths = pad_samples(samples)
                path = os.path.join(DATA_DIR, f'ood_{bucket_name}.npy')
                np.save(path, padded)
                len_path = os.path.join(DATA_DIR, f'ood_{bucket_name}_lengths.npy')
                np.save(len_path, lengths)
                seq_lengths = [len(s) for s in samples]
                print(f"  Saved: {path} ({len(samples)} samples, "
                      f"seq_len=[{min(seq_lengths)}, {max(seq_lengths)}])")
            else:
                print(f"  SKIPPED: {bucket_name} — could not generate valid samples")

    # Save metadata
    meta = {
        'task': 'arithmetic',
        'vocab_size': VOCAB_SIZE,
        'special_tokens': {
            'sep': SEP_TOKEN,
            'add': ADD_TOKEN,
            'sub': SUB_TOKEN,
            'mul': MUL_TOKEN,
            'div': DIV_TOKEN,
            'pad': 0,
        },
        'max_num': MAX_NUM,
        'operators': operators,
        'buckets': buckets,
        'ood_buckets': ood_buckets,
        'train_per_bucket': samples_per_bucket,
        'val_per_bucket': val_per_bucket,
        'val_deduped': True,
    }
    with open(os.path.join(DATA_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Examples
    print(f"\nExamples:")
    for bucket_name, samples in list(train_data.items())[:2]:
        print(f"\n  [{bucket_name}]")
        for i in range(min(3, len(samples))):
            s = samples[i]
            print(f"    {decode_sample(s)}")
            sep_positions = [j for j, t in enumerate(s) if t == SEP_TOKEN]
            if len(sep_positions) >= 2:
                phase1 = s[:sep_positions[0]]
                phase2 = s[sep_positions[0]+1:sep_positions[1]]
                phase3 = s[sep_positions[1]+1:]
                print(f"      Phase1: {decode_sample(phase1)}")
                print(f"      Phase2: {decode_sample(phase2)}")
                print(f"      Phase3: {decode_sample(phase3)}")

    print(f"\nDone. Run train.py next.")


if __name__ == '__main__':
    main()
