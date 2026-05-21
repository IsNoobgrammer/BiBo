"""Causal attention mask utilities"""
import torch
from typing import List, Optional, Tuple, Union

__all__ = [
    'make_causal_mask',
    'expand_mask',
    'prepare_4d_causal_attention_mask',
]


def make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """Make causal mask for self-attention.

    Args:
        input_ids_shape: (batch_size, tgt_len)
        dtype: Output dtype
        device: Output device
        past_key_values_length: Length of cached key/values

    Returns:
        Causal mask of shape (bsz, 1, tgt_len, tgt_len + past_len)
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len]."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
):
    """Create causal 4D mask, optionally incorporating an existing attention_mask.

    Args:
        attention_mask: Optional (bsz, seq_len) mask
        input_shape: (batch_size, seq_len)
        inputs_embeds: Embedded inputs (for dtype/device)
        past_key_values_length: Length of cached key/values

    Returns:
        4D causal mask of shape (bsz, 1, seq_len, seq_len + past_len)
    """
    bsz, seq_len = input_shape
    causal_mask = make_causal_mask(
        (bsz, seq_len),
        inputs_embeds.dtype,
        device=inputs_embeds.device,
        past_key_values_length=past_key_values_length,
    )
    if attention_mask is not None:
        expanded_attn_mask = expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len).to(
            inputs_embeds.device
        )
        causal_mask = causal_mask + expanded_attn_mask
    return causal_mask
