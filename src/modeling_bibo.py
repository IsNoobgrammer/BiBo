"""
BiBo modeling - Modular implementation

This file provides a flat import interface for backward compatibility.
All components are now organized in src/modeling/ submodules.

Structure:
    modeling/
    ├── norm.py              # BiBoRMSNorm
    ├── embed.py             # BiBoRotaryEmbedding, apply_rotary_pos_emb
    ├── attn/                # Attention
    │   ├── base.py          # BiBoAttention (standard softmax + SSMax)
    │   ├── ssmax.py         # SSMax scaling
    │   └── utils.py         # repeat_kv
    ├── ffn/                 # Feed-forward networks
    │   ├── mlp.py           # BiBoMLP (standard MLP)
    │   ├── experts.py       # Special experts (Identity/ReLU/Zero/Noise/Conv)
    │   ├── router.py        # BiBoMoERouter
    │   └── moe.py           # BiBoMoELayer
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
from src.modeling.attn import (
    BiBoAttention,
    repeat_kv,
    apply_ssmax_query_scaling,
)
from src.modeling.ffn import (
    BiBoMLP,
    BiBoIdentityExpert,
    BiBoReLUExpert,
    BiBoZeroExpert,
    BiBoNoiseExpert,
    BiBoCausalConv1D,
    BiBoMoERouter,
    BiBoMoELayer,
)
from src.modeling.layers import BiBoDecoderLayer
from src.modeling.models import (
    BiBoPreTrainedModel,
    BiBoModel,
    BiBoForCausalLM,
)

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
    'BiBoIdentityExpert',
    'BiBoReLUExpert',
    'BiBoZeroExpert',
    'BiBoNoiseExpert',
    'BiBoCausalConv1D',
    'BiBoMoERouter',
    'BiBoMoELayer',
    
    # Layers
    'BiBoDecoderLayer',
    
    # Models
    'BiBoPreTrainedModel',
    'BiBoModel',
    'BiBoForCausalLM',
]
