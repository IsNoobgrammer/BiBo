"""Datasets and data loaders for sorting and arithmetic tasks."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model_utils import CFG

__all__ = [
    'SequenceDataset',
    'ArithmeticDataset',
    'CurriculumDataLoader',
    'BucketedDataLoader',
    'arithmetic_collate_fn',
]

T = CFG['training']


class SequenceDataset(Dataset):
    """
    Sorting task — single bucket (fixed length, no padding).
    Format: [unsorted] [SEP] [sorted]
    Labels: [-100 for unsorted+SEP] [sorted tokens]
    """
    def __init__(self, npy_path):
        self.data = np.load(npy_path)
        self.full_len = self.data.shape[1]
        self.seq_len = (self.full_len - 1) // 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data[idx]
        input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)

        labels = torch.tensor(full_seq[1:], dtype=torch.long)
        labels[:self.seq_len] = -100

        return input_ids, labels


class ArithmeticDataset(Dataset):
    """
    Arithmetic task — variable length, padded.
    Format: [phase1] [SEP] [phase2] [SEP] [phase3]
    Labels: [-100 for phase1] [SEP + phase2 + SEP + phase3] (predict from first SEP onward)

    PAD tokens (0) are masked in labels as -100.
    """
    def __init__(self, npy_path, lengths_path=None):
        self.data = np.load(npy_path)  # [N, max_len] (padded)
        if lengths_path and os.path.exists(lengths_path):
            self.lengths = np.load(lengths_path)  # [N] actual lengths
        else:
            # Infer lengths from PAD tokens
            self.lengths = np.array([
                np.max(np.nonzero(row)[0]) + 1 if np.any(row != 0) else 0
                for row in self.data
            ])
        self.sep_token = T['vocab_size'] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data[idx]
        length = int(self.lengths[idx])

        # input_ids = full_seq[:-1], labels = full_seq[1:]
        input_ids = torch.tensor(full_seq[:-1], dtype=torch.long)
        labels = torch.tensor(full_seq[1:], dtype=torch.long)

        # Find first SEP position in the ORIGINAL sequence
        sep_positions = np.where(full_seq == self.sep_token)[0]
        if len(sep_positions) > 0:
            first_sep = sep_positions[0]
            # In shifted labels, mask everything before first_sep (phase1 input)
            # labels[i] corresponds to predicting full_seq[i+1]
            # We want to predict from SEP onward, so mask labels[:first_sep-1]
            # Actually: input_ids[first_sep-1] = last token of phase1
            #           labels[first_sep-1] = SEP (this IS a target — model should predict SEP)
            # So mask labels[:first_sep-1] (everything before the SEP prediction)
            labels[:first_sep - 1] = -100

        # Mask PAD tokens in labels
        labels[labels == 0] = -100
        # Also mask any position beyond actual length
        if length < len(full_seq):
            labels[length - 1:] = -100  # shifted: labels go up to length-1

        return input_ids, labels


class CurriculumDataLoader:
    """
    Curriculum learning: iterates through stages of increasing seq_len,
    then a final mixed stage with 10% of each bucket (all lengths interleaved).

    Strategy: short → long (full data each), then mixed (10% each, round-robin).
    """
    def __init__(self, data_dir, split, batch_size, stages, shuffle=True):
        self.stages = stages
        self.loaders = {}
        self.datasets = {}
        self.mixed_loaders = {}

        for seq_len in stages:
            path = os.path.join(data_dir, f'{split}_len_{seq_len}.npy')
            if os.path.exists(path):
                ds = SequenceDataset(path)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=T['num_workers'], pin_memory=True, drop_last=True)
                self.datasets[seq_len] = ds
                self.loaders[seq_len] = loader

                # Mixed stage: 10% subset of each bucket
                n_mixed = max(1, len(ds) // 10)
                mixed_ds = torch.utils.data.Subset(ds, list(range(n_mixed)))
                mixed_loader = DataLoader(mixed_ds, batch_size=batch_size, shuffle=shuffle,
                                          num_workers=T['num_workers'], pin_memory=True, drop_last=True)
                self.mixed_loaders[seq_len] = mixed_loader
            else:
                print(f"  WARNING: {path} not found, skipping stage seq_len={seq_len}")

        self.available_stages = [s for s in stages if s in self.loaders]
        # Total = all curriculum stages + mixed final stage
        mixed_batches = sum(len(self.mixed_loaders[s]) for s in self.available_stages)
        self.total_batches = sum(len(self.loaders[s]) for s in self.available_stages) + mixed_batches

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        """Curriculum stages (short→long), then mixed final stage (10% each, round-robin)."""
        # Phase 1: curriculum stages in order
        for stage_idx, seq_len in enumerate(self.available_stages):
            loader = self.loaders[seq_len]
            for batch in loader:
                yield batch, seq_len, stage_idx

        # Phase 2: mixed stage — round-robin across all lengths (10% each)
        mixed_stage_idx = len(self.available_stages)
        iterators = {s: iter(self.mixed_loaders[s]) for s in self.available_stages}
        active = list(self.available_stages)
        while active:
            for s in list(active):
                try:
                    batch = next(iterators[s])
                    yield batch, s, mixed_stage_idx
                except StopIteration:
                    active.remove(s)


def arithmetic_collate_fn(batch):
    """
    Custom collate for mixed arithmetic batches (variable lengths).
    Pads input_ids with 0 and labels with -100 to the max length in the batch.
    """
    input_ids_list, labels_list = zip(*batch)
    max_len = max(ids.shape[0] for ids in input_ids_list)

    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = ids.shape[0]
        padded_inputs[i, :seq_len] = ids
        padded_labels[i, :seq_len] = lab

    return (padded_inputs, padded_labels)


class BucketedDataLoader:
    """
    Non-curriculum: for arithmetic, concatenates ALL buckets into one dataset
    and shuffles globally (fully mixed). Uses dynamic padding per batch.
    For sorting, round-robins across buckets (fixed lengths per bucket).
    """
    def __init__(self, data_dir, split, batch_size, stages=None, shuffle=True, task='sort'):
        self.datasets = []
        self.loaders = []

        if task == 'arithmetic':
            # Arithmetic: concatenate all buckets into one mixed dataset
            arith_cfg = T.get('arithmetic', {})
            buckets = arith_cfg.get('buckets', [[3, 7], [9, 16], [19, 30], [35, 50]])
            all_datasets = []
            for min_t, max_t in buckets:
                bucket_name = f'arith_{min_t}_{max_t}'
                path = os.path.join(data_dir, f'{split}_{bucket_name}.npy')
                len_path = os.path.join(data_dir, f'{split}_{bucket_name}_lengths.npy')
                if os.path.exists(path):
                    ds = ArithmeticDataset(path, len_path)
                    all_datasets.append(ds)
                else:
                    print(f"  WARNING: {path} not found, skipping bucket {bucket_name}")

            if all_datasets:
                # Concatenate all buckets — fully mixed when shuffled
                combined = torch.utils.data.ConcatDataset(all_datasets)
                loader = DataLoader(combined, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=T['num_workers'], pin_memory=True, drop_last=True,
                                    collate_fn=arithmetic_collate_fn)
                self.datasets = all_datasets
                self.loaders = [loader]

        else:
            # Sorting buckets: round-robin (different fixed lengths per bucket)
            seq_lens = stages if stages else [64, 128, 256]
            for seq_len in seq_lens:
                path = os.path.join(data_dir, f'{split}_len_{seq_len}.npy')
                if os.path.exists(path):
                    ds = SequenceDataset(path)
                    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                        num_workers=T['num_workers'], pin_memory=True, drop_last=True)
                    self.datasets.append(ds)
                    self.loaders.append(loader)

        self.total_batches = sum(len(l) for l in self.loaders)

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        iterators = [iter(l) for l in self.loaders]
        active = list(range(len(iterators)))
        while active:
            for i in list(active):
                try:
                    batch = next(iterators[i])
                    yield batch, None, None
                except StopIteration:
                    active.remove(i)
