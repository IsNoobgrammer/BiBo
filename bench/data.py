"""
BiBo Benchmark — Data Pipeline

Dataset: tinycompany/Instruct-packed-2K-Context-tk-QTK-81K
- Pre-tokenized with QTK-81K tokenizer
- Packed to 2048 token sequences
- Truncated to seq_len (default 1024) for benchmark
"""

from datasets import load_dataset  # Must import before torch on Windows
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
        # Shift for next-token prediction: predict token[i+1] from token[:i+1]
        # labels[i] = input_ids[i+1], last label is -100 (ignored in loss)
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
    print(f"  Columns: {ds.column_names}")

    # Truncate + shift for NLP
    ds = PackedLMDataset(ds, seq_len=seq_len)

    # Split train/val
    total = len(ds)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=generator)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Seq len: {seq_len}")
    return train_ds, val_ds


def create_dataloader(dataset, batch_size, num_workers=0, shuffle=True):
    """Create DataLoader with proper settings for packed sequences."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Consistent batch sizes for torch.compile
    )


if __name__ == "__main__":
    train_ds, val_ds = load_benchmark_data(seq_len=1024)
    loader = create_dataloader(train_ds, batch_size=2)
    batch = next(iter(loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Sample input_ids[:20]: {batch['input_ids'][0][:20].tolist()}")
