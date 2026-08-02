"""SSMax scaling"""
import torch

__all__ = ['apply_ssmax_query_scaling']

# log(n) per query position depends only on (kv_len, q_len, device) -- never on the data -- and
# training calls this with one shape forever. Caching it drops an arange + clamp + log from every
# attention layer of every micro-batch. ponytail: unbounded dict, but the keys are shapes; the
# variable-length eval tops out around 150 entries of <=2048 floats. Bound it if that ever changes.
_LOGN = {}


def _log_n(kv_len, q_len, device):
    key = (kv_len, q_len, device)
    v = _LOGN.get(key)
    if v is None:
        n = torch.arange(kv_len - q_len + 1, kv_len + 1, device=device, dtype=torch.float32)
        v = torch.log(n.clamp(min=1.0)).view(1, 1, q_len, 1)
        _LOGN[key] = v
    return v


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
        log_n = torch.log(n.clamp(min=1.0))
    else:
        # Causal context length per query position: n_j = (kv_len - q_len) + j + 1
        log_n = _log_n(kv_len, q_len, query_states.device)
    # Collapse s and log(n) into ONE broadcast scale, IN q's dtype, before touching q.
    #
    # Two separate costs hide in the naive `q * ssmax_scale * log_n`:
    #   1. two passes over the full (B,H,q_len,D) tensor instead of one. The combined scale is
    #      (1,H,q_len,1) -- 8k elements -- so folding it is free.
    #   2. dtype promotion. ssmax_scale is an fp32 Parameter and q is bf16 under autocast, so the
    #      product is fp32: a full-size fp32 q gets materialized, handed to SDPA against bf16 k/v
    #      (measured: sdpa_in = (float32, bfloat16, bfloat16)), and immediately cast back down to
    #      bf16 -- double the bytes and an extra kernel for a value that is discarded.
    # Together: 174.7k -> 168.1k tok/s at 64x4x1024.
    #
    # What casting costs on ds/dL, measured rather than assumed: rel 1.9e-3, worst head 8.5e-3.
    # d(q*s)/ds = q, so the scale's own rounding never enters the gradient. The error is that a
    # bf16 output makes each grad_out*q PRODUCT round to bf16 before the (still fp32) reduction.
    # Accepted: grad_out is already bf16-valued anyway -- SDPA emits bf16 and its backward hands
    # back bf16 gradients that the autocast cast-node merely re-types -- so 0.2% puts ds/dL on
    # exactly the same footing as every other gradient in the model, all of which come off bf16
    # GEMMs, and AdamW normalizes by sqrt(v) before it reaches s. An exact-gradient version means
    # a custom Function that upcasts inside the reduction; not worth 0.2%.
    # (In an fp32 run .to() is a no-op and everything stays fp32, as before.)
    return query_states * (ssmax_scale * log_n).to(query_states.dtype)
