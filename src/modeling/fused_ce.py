"""
Chunked Fused Linear Cross-Entropy Loss.

Computes lm_head projection + cross-entropy loss without materializing the full
logit tensor. For vocab=81K, batch=4, seq=1024, this saves ~1.3GB of memory.

Approach: Process the vocabulary in chunks (e.g., 4096 tokens at a time),
compute partial log-sum-exp, and accumulate the loss.

This is a pure PyTorch implementation that works on any GPU.
For maximum speed on Linux/Kaggle, use liger-kernel's Triton version instead.
"""

import torch
import torch.nn.functional as F


class ChunkedFusedLinearCrossEntropy(torch.autograd.Function):
    """
    Fused linear + cross-entropy that never materializes full logits.
    
    Forward: chunks the vocab dimension, computes loss via log-sum-exp trick.
    Backward: recomputes logit chunks on-the-fly (no stored logits).
    """
    
    @staticmethod
    def forward(ctx, hidden_states, weight, labels, chunk_size=4096, ignore_index=-100):
        """
        Args:
            hidden_states: (N, hidden_size) — flattened sequence
            weight: (vocab_size, hidden_size) — lm_head weight
            labels: (N,) — target token IDs
            chunk_size: vocab chunk size for memory efficiency
        Returns:
            loss: scalar
        """
        N, H = hidden_states.shape
        V = weight.shape[0]
        
        # Mask for valid (non-ignored) positions
        valid_mask = labels != ignore_index
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            ctx.save_for_backward(hidden_states, weight, labels, valid_mask)
            ctx.chunk_size = chunk_size
            ctx.ignore_index = ignore_index
            return hidden_states.new_zeros((), requires_grad=True)
        
        # Only compute loss for valid positions (saves compute on padding)
        valid_hidden = hidden_states[valid_mask]  # (num_valid, H)
        valid_labels = labels[valid_mask]          # (num_valid,)
        
        # Compute loss using log-sum-exp trick in chunks
        # loss = -log_softmax(logits)[target] = -logits[target] + log_sum_exp(logits)
        
        # First pass: compute log-sum-exp and target logits
        max_logits = torch.full((num_valid,), float('-inf'), device=hidden_states.device, dtype=torch.float32)
        sum_exp = torch.zeros(num_valid, device=hidden_states.device, dtype=torch.float32)
        target_logits = torch.zeros(num_valid, device=hidden_states.device, dtype=torch.float32)
        
        for chunk_start in range(0, V, chunk_size):
            chunk_end = min(chunk_start + chunk_size, V)
            # Compute logits for this vocab chunk: (num_valid, chunk_size)
            logits_chunk = F.linear(valid_hidden, weight[chunk_start:chunk_end]).float()
            
            # Update max for numerical stability
            chunk_max = logits_chunk.max(dim=-1).values
            new_max = torch.maximum(max_logits, chunk_max)
            
            # Rescale previous sum_exp
            sum_exp = sum_exp * torch.exp(max_logits - new_max)
            # Add current chunk's contribution
            sum_exp += torch.exp(logits_chunk - new_max.unsqueeze(-1)).sum(dim=-1)
            max_logits = new_max
            
            # Extract target logits if they fall in this chunk
            in_chunk_mask = (valid_labels >= chunk_start) & (valid_labels < chunk_end)
            if in_chunk_mask.any():
                local_indices = valid_labels[in_chunk_mask] - chunk_start
                target_logits[in_chunk_mask] = logits_chunk[in_chunk_mask].gather(
                    1, local_indices.unsqueeze(1)
                ).squeeze(1)
        
        # loss = mean(-target_logit + max + log(sum_exp))
        log_sum_exp = max_logits + torch.log(sum_exp)
        per_token_loss = -target_logits + log_sum_exp
        loss = per_token_loss.mean()
        
        # Save for backward
        ctx.save_for_backward(hidden_states, weight, labels, valid_mask)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.num_valid = num_valid
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, labels, valid_mask = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        num_valid = ctx.num_valid
        
        N, H = hidden_states.shape
        V = weight.shape[0]
        
        if num_valid == 0:
            return torch.zeros_like(hidden_states), torch.zeros_like(weight), None, None, None
        
        valid_hidden = hidden_states[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Recompute softmax probabilities in chunks and accumulate gradients
        # d_loss/d_logits = softmax(logits) - one_hot(target)  (scaled by grad_output/N)
        
        grad_hidden = torch.zeros_like(hidden_states)
        grad_weight = torch.zeros_like(weight)
        
        # First: compute full log-sum-exp (needed for softmax)
        max_logits = torch.full((num_valid,), float('-inf'), device=hidden_states.device, dtype=torch.float32)
        sum_exp = torch.zeros(num_valid, device=hidden_states.device, dtype=torch.float32)
        
        for chunk_start in range(0, V, chunk_size):
            chunk_end = min(chunk_start + chunk_size, V)
            logits_chunk = F.linear(valid_hidden, weight[chunk_start:chunk_end]).float()
            chunk_max = logits_chunk.max(dim=-1).values
            new_max = torch.maximum(max_logits, chunk_max)
            sum_exp = sum_exp * torch.exp(max_logits - new_max)
            sum_exp += torch.exp(logits_chunk - new_max.unsqueeze(-1)).sum(dim=-1)
            max_logits = new_max
        
        log_sum_exp = max_logits + torch.log(sum_exp)
        
        # Second pass: compute gradients chunk by chunk
        scale = grad_output / num_valid
        
        for chunk_start in range(0, V, chunk_size):
            chunk_end = min(chunk_start + chunk_size, V)
            chunk_weight = weight[chunk_start:chunk_end]  # (chunk, H)
            
            logits_chunk = F.linear(valid_hidden, chunk_weight).float()
            # softmax probs for this chunk
            probs_chunk = torch.exp(logits_chunk - log_sum_exp.unsqueeze(-1))
            
            # Subtract 1 at target positions
            in_chunk_mask = (valid_labels >= chunk_start) & (valid_labels < chunk_end)
            if in_chunk_mask.any():
                local_indices = valid_labels[in_chunk_mask] - chunk_start
                probs_chunk[in_chunk_mask, local_indices] -= 1.0
            
            # probs_chunk is now d_loss/d_logits (before scaling)
            probs_chunk = probs_chunk * scale
            probs_chunk = probs_chunk.to(hidden_states.dtype)
            
            # grad_hidden += probs_chunk @ chunk_weight
            grad_valid_chunk = probs_chunk @ chunk_weight
            grad_hidden[valid_mask] += grad_valid_chunk
            
            # grad_weight[chunk] += probs_chunk.T @ valid_hidden
            grad_weight[chunk_start:chunk_end] += probs_chunk.t() @ valid_hidden
        
        return grad_hidden, grad_weight, None, None, None


def chunked_fused_linear_cross_entropy(hidden_states, weight, labels, chunk_size=4096, ignore_index=-100):
    """
    Compute fused linear + cross-entropy loss without materializing full logits.
    
    Args:
        hidden_states: (N, hidden_size) or (B, S, hidden_size)
        weight: (vocab_size, hidden_size)
        labels: (N,) or (B, S)
        chunk_size: vocab chunk size (default 4096, tune for your GPU)
        ignore_index: label value to ignore (default -100)
    
    Returns:
        loss: scalar tensor
    """
    return ChunkedFusedLinearCrossEntropy.apply(
        hidden_states, weight, labels, chunk_size, ignore_index
    )
