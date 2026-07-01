"""
BiBo modeling - Modular implementation

This file provides a flat import interface for backward compatibility.
All components are now organized in src/modeling/ submodules.

Structure:
    modeling/
    ├── norm.py              # BiBoRMSNorm
    ├── embed.py             # BiBoRotaryEmbedding, apply_rotary_pos_emb
    ├── masks.py             # Causal mask utilities
    ├── attn/                # Attention
    │   ├── base.py          # BiBoAttention (standard softmax + SSMax)
    │   ├── ssmax.py         # SSMax scaling
    │   └── utils.py         # repeat_kv
    ├── ffn/                 # Feed-forward networks
    │   ├── mlp.py           # BiBoMLP (standard MLP)
    │   ├── experts.py       # Special experts (Identity/Zero) + PolyGLU experts
    │   ├── router.py        # BiBoMoERouter
    │   └── moe.py           # BiBoFusedExperts, BiBoMoELayer
    ├── layers.py            # BiBoDecoderLayer
    └── models.py            # BiBoPreTrainedModel, BiBoModel, BiBoForCausalLM

Usage:
    # Import from this file (backward compatible)
    from src.modeling_bibo import BiBoModel, BiBoForCausalLM

    # Or import from submodules (recommended)
    from src.modeling.models import BiBoModel, BiBoForCausalLM
    from src.modeling.attn import BiBoAttention
    from src.modeling.ffn import BiBoMLP, BiBoMoELayer
"""

# Re-export all components for backward compatibility
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
    # Normalization
    'BiBoRMSNorm',
    # Embeddings
    'BiBoRotaryEmbedding',
    'apply_rotary_pos_emb',
    'rotate_half',
    # Attention
    'BiBoAttention',
    'repeat_kv',
    'apply_ssmax_query_scaling',
    # FFN
    'BiBoMLP',
    'BiBoPolyGLUExpert',
    'BiBoCausalConv1D',
    'BiBoMoERouter',
    'BiBoFusedExperts',
    'BiBoMoELayer',
    # Layers
    'BiBoDecoderLayer',
    # Models
    'BiBoPreTrainedModel',
    'BiBoModel',
    'BiBoForCausalLM',
]
