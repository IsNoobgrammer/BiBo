"""SSMax scaling"""
import torch

__all__ = ['apply_ssmax_query_scaling']


def apply_ssmax_query_scaling(query_states: torch.Tensor, kv_len: int, ssmax_scale: torch.nn.Parameter,
                              context_lens: torch.Tensor = None) -> torch.Tensor:
    """
    Apply SSMax query scaling — PER CAUSAL POSITION (paper: arXiv:2501.19399, Eq. 2).

    SSMax: learnable, seq-len adaptive temperature per head. Prevents attention fading
    in long contexts by multiplying each logit by C = scale * log(n).

    Standard softmax ratio: exp(z_i) / exp(z_k) = exp(z_i - z_k)
    SSMax ratio: exp(C*z_i) / exp(C*z_k) = (exp(z_i - z_k))^C

    CRITICAL: n is the CAUSAL context length of EACH query, not the global sequence length.
    Under a causal mask, query at absolute position p attends to p+1 keys, so n varies along
    the sequence. Using one global log(kv_len) for all positions collapses SSMax to a constant
    temperature during fixed-length training (the per-position log(n) signal is never exercised)
    — that is a bug. We compute n per query position directly from the shapes:

      query j (j = 0..q_len-1) has causal context length  n_j = (kv_len - q_len) + j + 1

    - Training (no cache, q_len == kv_len == L):  n = 1, 2, ..., L   (the real SSMax signal)
    - Single-token decode (q_len == 1):           n = [kv_len]       (unchanged vs old behavior)
    - Prefill with cache:                         correct per-position
    Assumes causal attention (BiBo is a causal LM).

    Args:
        query_states: Query tensor (B, H, q_len, D)
        kv_len: total key/value length (past + current)
        ssmax_scale: Learnable per-head scale param, shape (1, H, 1, 1)
        context_lens: optional (B, q_len) REAL causal context length per query — used with a
            padding mask, where masked pad keys must not count toward n (grid positions would
            over-count by the pad width and shift the temperature)

    Returns:
        Scaled query states (B, H, q_len, D)
    """
    q_len = query_states.shape[-2]
    if context_lens is not None:
        n = context_lens.to(torch.float32).view(context_lens.shape[0], 1, q_len, 1)
    else:
        # Causal context length per query position: n_j = (kv_len - q_len) + j + 1
        n = torch.arange(kv_len - q_len + 1, kv_len + 1,
                         device=query_states.device, dtype=torch.float32).view(1, 1, q_len, 1)
    log_n = torch.log(n.clamp(min=1.0)).to(query_states.dtype)
    return query_states * ssmax_scale * log_n
