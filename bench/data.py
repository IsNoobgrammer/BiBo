"""
BiBo Benchmark — Data Pipeline

Dataset: tinycompany/Instruct-packed-2K-Context-tk-QTK-81K
- Pre-tokenized with QTK-81K tokenizer
- Packed to 2048 token sequences
- Truncated to seq_len (default 1024) for benchmark

Also loads HellaSwag + ARC-Challenge for eval benchmarks.
"""

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset


class PackedLMDataset(Dataset):
    """Wraps HF dataset for language modeling with truncation."""

    def __init__(self, hf_dataset, seq_len=1024):
        self.data = hf_dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"][: self.seq_len]
        labels = input_ids[1:] + [-100]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_benchmark_data(
    dataset_name="tinycompany/Instruct-packed-2K-Context-tk-QTK-81K",
    seq_len=1024,
    val_split=0.05,
    seed=42,
):
    """Load dataset, truncate to seq_len, split train/val."""
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"  Raw dataset size: {len(ds)}")

    ds = PackedLMDataset(ds, seq_len=seq_len)

    total = len(ds)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=generator
    )

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Seq len: {seq_len}")
    return train_ds, val_ds


def create_dataloader(dataset, batch_size, num_workers=0, shuffle=True, seed=42,
                      rank=None, world_size=1):
    """Create DataLoader with proper settings for packed sequences.

    Under DDP (world_size > 1), uses a DistributedSampler so each rank reads a disjoint
    shard — without it both ranks read identical batches and the second GPU is wasted.
    The seed makes the global ordering identical across runs (BiBo vs Qwen comparability).
    """
    if world_size > 1 and rank is not None:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, seed=seed, drop_last=True)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )


# ─────────────────────────────────────────────────────────────
# HellaSwag Benchmark
# ─────────────────────────────────────────────────────────────

def load_hellaswag(max_examples=None):
    """
    Load HellaSwag validation set from HuggingFace.

    Returns list of: {ctx, completions: [str], gold_idx: int}
    Source: Rowan/hellaswag
    """
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    data = []
    for item in ds:
        ctx = item["activity_label"] + ": " + item["ctx_a"] + " " + item["ctx_b"]
        completions = item["endings"]
        gold_idx = int(item["label"])
        data.append({
            "ctx": ctx,
            "completions": completions,
            "gold_idx": gold_idx,
        })
    return data


# ─────────────────────────────────────────────────────────────
# ARC-Challenge Benchmark
# ─────────────────────────────────────────────────────────────

def load_arc_challenge(max_examples=None):
    """
    Load ARC-Challenge from HuggingFace.

    Returns list of: {question, completions: [str], gold_idx: int}
    Source: allenai/ai2_arc (ARC-Challenge split)
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    data = []
    for item in ds:
        question = item["question"]
        choices = item["choices"]
        completions = choices["text"]
        answer_key = choices["label"][0] if isinstance(choices["label"], list) else choices["label"]
        gold_idx = label_to_idx.get(answer_key, 0)
        data.append({
            "question": question,
            "completions": completions,
            "gold_idx": gold_idx,
        })
    return data
