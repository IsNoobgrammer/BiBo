"""
Parallel training — BiBo on cuda:0, Qwen3MoE on cuda:1.
Supports curriculum learning: train on progressively longer sequences.

Loss: NTIL (Numerical Token Integrity Loss) — arXiv:2505.13077

Usage:
    python misc/kaggle/multi_gpu/data.py      # generate data first
    python misc/kaggle/multi_gpu/train.py     # train both
    python misc/kaggle/multi_gpu/train.py --seed 69
"""
import os
import sys
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from model_utils import CFG, BASE_DIR
from trainer import train_worker, seed_everything

T = CFG['training']
DATA_DIR = os.path.join(BASE_DIR, 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')


def main():
    parser = argparse.ArgumentParser(description='BiBo vs Qwen3MoE parallel training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config.yaml, default: config value)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else T['seed']
    T['seed'] = seed

    seed_everything(seed)
    print(f"\n  Seed: {seed}")
    print(f"  Task: {T.get('task', 'sort')}")
    print(f"  Curriculum: {T.get('curriculum', False)}")
    if T.get('curriculum'):
        print(f"  Stages: {T.get('curriculum_stages')}")
    print(f"  Epochs: {T['epochs']}")
    print(f"  LR: {T['lr']}")
    print(f"  Vocab: {T['vocab_size']}")

    # Verify data exists
    task = T.get('task', 'sort')
    stages = T.get('curriculum_stages', [64, 128, 256])

    if task == 'arithmetic':
        arith_cfg = T.get('arithmetic', {})
        buckets = arith_cfg.get('buckets', [[3, 7], [9, 16], [19, 30], [35, 50]])
        first_bucket = buckets[0]
        check_file = os.path.join(DATA_DIR, f'train_arith_{first_bucket[0]}_{first_bucket[1]}.npy')
        data_script = 'data_arithmetic.py'
    else:
        check_file = os.path.join(DATA_DIR, f'train_len_{stages[0]}.npy')
        data_script = 'data.py'

    if not os.path.exists(check_file):
        print(f"ERROR: Data not found ({check_file}). Run `python misc/kaggle/multi_gpu/{data_script}` first.")
        sys.exit(1)

    # Spawn parallel processes
    print("\nLaunching parallel training...")
    print(f"  BiBo → {CFG['bibo'].get('device', 'cuda:0')}")
    print(f"  Qwen3MoE → {CFG['qwen3moe'].get('device', 'cuda:1')}")
    print()

    mp.set_start_method('spawn', force=True)

    p_bibo = mp.Process(target=train_worker, args=('bibo',))
    p_qwen = mp.Process(target=train_worker, args=('qwen3moe',))

    p_bibo.start()
    p_qwen.start()

    p_bibo.join()
    p_qwen.join()

    print("\n" + "="*60)
    print("BOTH MODELS DONE")
    print("="*60)
    print(f"  Checkpoints: {SAVE_DIR}/")
    print(f"  Metrics: {METRICS_DIR}/")
    print("  Next: python misc/kaggle/multi_gpu/analyze_model.py")


if __name__ == '__main__':
    main()
