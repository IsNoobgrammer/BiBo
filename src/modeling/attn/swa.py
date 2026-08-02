"""Sliding-window attention (SWA) — FlexAttention fast path, eager reference core.

The eager core in `utils.eager_attention_forward` remains the numerics target and still serves
every case FlexAttention cannot (padding mask, output_attentions, CPU). It is not the hot path
any more: it materializes the full (B,H,q,kv) score matrix and masks it, so a window saves it
nothing. Measured at 64x4x1024, block3 pattern: global 169.3k tok/s -> SWA eager 135.4k (-20%)
and +9.2 GB, with **w128 and w512 costing the same 135.4k** -- the tell that no work was being
skipped, since a 4x smaller window should be far cheaper.

FlexAttention is block-sparse: `create_block_mask` marks whole 128x128 blocks outside the band as
skippable and the kernel never visits them, so cost actually tracks the window.

The sink is applied in closed form rather than as an extra column. With one value-less column of
logit beta appended to the softmax, the real weights are
    w_i = exp(z_i) / (sum_j exp(z_j) + exp(beta))
        = [exp(z_i) / sum_j exp(z_j)] * sum_j exp(z_j) / (sum_j exp(z_j) + exp(beta))
        = p_i * sigmoid(lse - beta)
so the sink output is just the ordinary attention output scaled per (batch, head, query) by
sigmoid(lse - beta). `flex_attention(..., return_lse=True)` returns that lse directly. This is
exact -- not an approximation of the eager path -- and avoids appending a column that would push
KV_LEN off a block multiple. beta is compared against SCALED logits, matching the eager core where
the sink is concatenated after `* scaling`.
"""
import torch
from .utils import causal_band_mask, padding_bias, eager_attention_forward

__all__ = ['swa_attention']

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
except ImportError:                                    # torch < 2.5
    _HAS_FLEX = False

# create_block_mask is expensive and depends only on shape, never on data. Training calls this with
# one shape forever; the variable-length eval would otherwise rebuild it per length -- the same trap
# that made a Triton autotune key on S cost 4.1x on the eval path.
_BLOCK_MASK_CACHE = {}
_FLEX = None


def _flex_call():
    """torch.compile'd flex_attention. Compiled lazily and once -- flex is substantially slower
    interpreted, but compiling at import time would pay the cost even for CPU/eager-only runs."""
    global _FLEX
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def _block_mask(q_len, kv_len, window, device):
    key = (q_len, kv_len, window, str(device))
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is None:
        offset = kv_len - q_len          # absolute query position, as in causal_band_mask

        def band(b, h, q_idx, kv_idx):
            i = q_idx + offset
            return (kv_idx <= i) & ((i - kv_idx) < window)

        bm = create_block_mask(band, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device)
        _BLOCK_MASK_CACHE[key] = bm
    return bm


def swa_attention(query, key, value, sinks, *, sliding_window, num_key_value_groups, scaling,
                  padding_mask=None, dropout=0.0, training=False):
    """SWA forward. `query` (B,H,q,d); `key`/`value` GROUPED (B,H_kv,kv,d); `sinks` per-head
    bias (H,) or None; `padding_mask` 2D (B,K_total) 1=real/0=pad or None.
    Returns (attn_output (B,H,q,d), probs (B,H,q,kv) or None).

    FlexAttention handles the hot path. Anything it cannot express falls back to the eager core:
    a padding mask (data-dependent, and packed training never has one), dropout, or CPU."""
    q_len, kv_len = query.shape[-2], key.shape[-2]
    use_flex = (_HAS_FLEX and padding_mask is None and query.is_cuda
                and not (training and dropout > 0.0))
    if use_flex:
        # enable_gqa keeps key/value GROUPED -- no repeat_kv copy, matching the eager core's inputs.
        out, lse = _flex_call()(
            query, key, value, block_mask=_block_mask(q_len, kv_len, sliding_window, query.device),
            scale=scaling, enable_gqa=True, return_lse=True)
        if sinks is not None:
            # lse is natural-log and already over SCALED scores, the same units as beta.
            beta = sinks.to(torch.float32).reshape(1, -1, 1)
            out = out * torch.sigmoid(lse.to(torch.float32) - beta).unsqueeze(-1).to(out.dtype)
        return out, None

    # Band mask is built even at q_len==1 decode so the window is enforced when the cache
    # holds more than sliding_window keys (uncropped/external caches).
    attn_mask = causal_band_mask(q_len, kv_len, sliding_window, query.dtype, query.device)
    if padding_mask is not None:
        attn_mask = attn_mask + padding_bias(padding_mask, kv_len, query.dtype)
    return eager_attention_forward(
        query, key, value, attn_mask, scaling, num_key_value_groups,
        dropout=dropout, training=training, sinks=sinks)
