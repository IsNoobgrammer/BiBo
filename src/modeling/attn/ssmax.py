"""SSMax scaling"""
import torch

__all__ = ['apply_ssmax_query_scaling']


def apply_ssmax_query_scaling(query_states: torch.Tensor, kv_len: int, ssmax_scale: torch.nn.Parameter) -> torch.Tensor:
    """
    Apply SSMax query scaling.
    
    SSMax: learnable, seq-len adaptive temperature per head.
    Prevents attention fading in long contexts.
    
    Standard softmax ratio: exp(z_i) / exp(z_k) = exp(z_i - z_k)
    SSMax ratio: exp(C*z_i) / exp(C*z_k) = (exp(z_i - z_k))^C
    where C = scale * log(seq_len)
    
    Args:
        query_states: Query tensor
        kv_len: Key/value seq len
        ssmax_scale: Learnable scale param
    
    Returns:
        Scaled query states
    """
    log_n = torch.log(
        torch.clamp(
            torch.tensor(kv_len, device=query_states.device, dtype=ssmax_scale.dtype),
            min=2.0,
        )
    )
    return query_states * ssmax_scale * log_n
