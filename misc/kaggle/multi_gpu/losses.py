"""NTIL Loss — Numerical Token Integrity Loss
Ref: arXiv:2505.13077
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NTILLoss']


class NTILLoss(nn.Module):
    """
    Numerical Token Integrity Loss for sorting tasks.

    Combines three components:
    1. CCE: Standard cross-entropy (primary gradient signal)
    2. EMD (token-level): Wasserstein-1 distance on predicted distribution
       - Penalizes "off by 1" less than "off by 10"
       - For 1D ordinal tokens: EMD = Σ|CDF_pred - CDF_target|
    3. Sequence-level: L1 distance between predicted values and target values
       - Captures "whole sequence correctness" signal
       - Penalizes cascading errors proportionally

    All components are differentiable. Loss computed externally (not inside model).

    Args:
        vocab_size: Total vocabulary size (includes SEP token)
        sep_token: SEP token id (excluded from ordinal distance)
        alpha_emd: Weight for EMD token-level loss
        alpha_seq: Weight for sequence-level loss
        max_ordinal: Maximum ordinal value for normalization (= sep_token)
    """
    def __init__(self, vocab_size=512, sep_token=511, alpha_emd=0.3, alpha_seq=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.sep_token = sep_token
        self.alpha_emd = alpha_emd
        self.alpha_seq = alpha_seq
        # Ordinal values: token i has value i (for tokens 0..sep_token-1)
        # SEP token has no ordinal meaning — excluded from EMD
        self.max_ordinal = sep_token  # normalize distances to [0, 1]

    def emd_1d(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute 1D Earth Mover's Distance (Wasserstein-1) between predicted
        distribution and one-hot target for ordinal tokens.

        For 1D ordinal labels, EMD = Σᵢ |CDF_pred(i) - CDF_target(i)|
        This is exact and O(V) — no need for Sinkhorn or LP solver.

        Args:
            log_probs: [N, vocab_size] log probabilities
            targets: [N] target token ids (ordinal values)
        Returns:
            emd: [N] per-token EMD values
        """
        # Only use ordinal tokens [0, sep_token) for EMD
        # Exclude SEP token from distance computation
        probs = log_probs[:, :self.sep_token].exp()  # [N, sep_token]

        # CDF of predicted distribution
        cdf_pred = probs.cumsum(dim=-1)  # [N, sep_token]

        # CDF of target (one-hot → step function)
        # CDF_target[i] = 0 if i < target, 1 if i >= target
        target_expanded = targets.unsqueeze(1)  # [N, 1]
        positions = torch.arange(self.sep_token, device=targets.device).unsqueeze(0)  # [1, sep_token]
        cdf_target = (positions >= target_expanded).float()  # [N, sep_token]

        # EMD = sum of |CDF_pred - CDF_target| / max_ordinal (normalized)
        emd = (cdf_pred - cdf_target).abs().sum(dim=-1) / self.max_ordinal

        return emd

    def sequence_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Sequence-level loss: L1 between predicted token values and target values.

        This captures "how wrong is the whole sequence" — not just per-token.
        Predicted value = expected value under predicted distribution (soft argmax).

        Args:
            logits: [B, S, V] raw logits
            targets: [B, S] target token ids
            mask: [B, S] bool mask (True = compute loss here)
        Returns:
            seq_loss: scalar
        """
        # Soft predicted value: E[token_value] under predicted distribution
        # Only over ordinal tokens [0, sep_token)
        probs = F.softmax(logits[:, :, :self.sep_token], dim=-1)  # [B, S, sep_token]
        ordinal_values = torch.arange(self.sep_token, device=logits.device, dtype=logits.dtype)  # [sep_token]
        predicted_values = (probs * ordinal_values).sum(dim=-1)  # [B, S] — expected value

        # Target values (already ordinal)
        target_values = targets.float()  # [B, S]

        # L1 distance normalized by max_ordinal
        l1 = (predicted_values - target_values).abs() / self.max_ordinal  # [B, S]

        # Only on masked positions
        if mask.any():
            seq_loss = l1[mask].mean()
        else:
            seq_loss = torch.tensor(0.0, device=logits.device)

        return seq_loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Compute NTIL loss.

        Args:
            logits: [B, S, V] raw logits from model (no loss computed inside model)
            labels: [B, S] target ids (-100 = ignore)
        Returns:
            total_loss: scalar (combined CCE + optional EMD + seq)
            loss_dict: dict with individual components for logging
        """
        B, S, V = logits.shape

        # Mask: only compute loss where labels != -100
        mask = (labels != -100)  # [B, S]

        if not mask.any():
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return zero, {'cce': 0.0, 'emd': 0.0, 'seq': 0.0, 'total': 0.0}

        # Flatten for per-token losses
        flat_logits = logits[mask]  # [N, V]
        flat_labels = labels[mask]  # [N]

        # 1. CCE (standard cross-entropy)
        cce_loss = F.cross_entropy(flat_logits, flat_labels)

        # 2. EMD (token-level ordinal distance) — skip entirely if alpha_emd == 0
        if self.alpha_emd > 0:
            log_probs = F.log_softmax(flat_logits, dim=-1)
            emd_per_token = self.emd_1d(log_probs, flat_labels)  # [N]
            emd_loss = emd_per_token.mean()
        else:
            emd_loss = torch.tensor(0.0, device=logits.device)

        # 3. Sequence-level loss
        seq_loss = self.sequence_loss(logits, labels.clamp(min=0), mask)

        # Combined
        total_loss = cce_loss + self.alpha_emd * emd_loss + self.alpha_seq * seq_loss

        loss_dict = {
            'cce': cce_loss.item(),
            'emd': emd_loss.item(),
            'seq': seq_loss.item(),
            'emd_scaled': (self.alpha_emd * emd_loss).item(),
            'seq_scaled': (self.alpha_seq * seq_loss).item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict
