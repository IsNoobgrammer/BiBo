"""Flat re-export of src/modeling/ for backward compatibility. Prefer the submodules directly."""
from src.modeling.norm import BiBoRMSNorm
from src.modeling.embed import BiBoRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from src.modeling.attn import BiBoAttention, repeat_kv, apply_ssmax_query_scaling
from src.modeling.ffn import (
    BiBoMLP,
    BiBoPolyGLUExpert,
    BiBoCausalConv1D,
    BiBoMoERouter,
    BiBoFusedExperts,
    BiBoMoELayer,
)
from src.modeling.layers import BiBoDecoderLayer
from src.modeling.models import BiBoPreTrainedModel, BiBoModel, BiBoForCausalLM

__all__ = [
    'BiBoRMSNorm',
    'BiBoRotaryEmbedding', 'apply_rotary_pos_emb', 'rotate_half',
    'BiBoAttention', 'repeat_kv', 'apply_ssmax_query_scaling',
    'BiBoMLP', 'BiBoPolyGLUExpert', 'BiBoCausalConv1D',
    'BiBoMoERouter', 'BiBoFusedExperts', 'BiBoMoELayer',
    'BiBoDecoderLayer',
    'BiBoPreTrainedModel', 'BiBoModel', 'BiBoForCausalLM',
]
