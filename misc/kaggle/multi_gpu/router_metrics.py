"""Router metric computation functions for analyze_router.py"""
import numpy as np
from itertools import combinations

__all__ = [
    'compute_gini',
    'compute_load_balance_metrics',
    'compute_expert_coselection',
    'compute_specialization_score',
]


def compute_gini(counts):
    """Gini coefficient: 0=perfect equality, 1=max inequality."""
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    if sorted_c.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_c) - (n + 1) * np.sum(sorted_c)) / (n * np.sum(sorted_c))


def compute_load_balance_metrics(indices, n_experts, top_k):
    """Compute comprehensive load balance metrics from routing indices."""
    counts = np.bincount(indices.flatten(), minlength=n_experts).astype(float)
    total = counts.sum()
    if total == 0:
        return {'entropy': 0, 'gini': 0, 'cv': 0, 'max_load': 0, 'min_load': 0, 'balance_ratio': 0}

    dist = counts / total
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    max_entropy = np.log(n_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    gini = compute_gini(counts)
    cv = np.std(counts) / (np.mean(counts) + 1e-10)

    return {
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'gini': float(gini),
        'cv': float(cv),
        'max_load': float(counts.max() / total),
        'min_load': float(counts.min() / total),
        'balance_ratio': float(counts.min() / (counts.max() + 1e-10)),
        'counts': counts.tolist(),
    }


def compute_expert_coselection(indices, n_experts):
    """Co-selection matrix: how often pairs of experts are selected together."""
    cosel = np.zeros((n_experts, n_experts))
    for row in indices:
        for i, j in combinations(row, 2):
            cosel[i, j] += 1
            cosel[j, i] += 1
    total_pairs = len(indices) * (indices.shape[1] * (indices.shape[1] - 1) / 2)
    if total_pairs > 0:
        cosel /= total_pairs
    return cosel


def compute_specialization_score(indices, n_experts, seq_len):
    """Expert specialization: KL divergence of position distribution from uniform."""
    scores = np.zeros(n_experts)
    uniform = np.ones(seq_len) / seq_len

    for exp_id in range(n_experts):
        mask = (indices == exp_id).any(axis=-1)
        if mask.sum() == 0:
            scores[exp_id] = 0
            continue
        pos_counts = np.zeros(seq_len)
        token_positions = np.arange(len(mask)) % seq_len
        selected_positions = token_positions[mask]
        for p in selected_positions:
            pos_counts[p] += 1
        pos_dist = pos_counts / (pos_counts.sum() + 1e-10)
        kl = np.sum(pos_dist * np.log((pos_dist + 1e-10) / uniform))
        scores[exp_id] = kl

    return scores
