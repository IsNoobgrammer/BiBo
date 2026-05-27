"""
BiBo Benchmark — Model Configuration

Baseline BiBo ~50M params:
- MLP router, sigmoid gate, shared expert ON
- Uniform SwiGLU experts (no PolyGLU, no Identity/Zero)
- GQA attention with SSMax
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configuration_bibo import BiBoConfig
from src.modeling_bibo import BiBoForCausalLM


# ─────────────────────────────────────────────────────────────
# Baseline BiBo ~50M (MLP router, shared expert, uniform SwiGLU)
# ─────────────────────────────────────────────────────────────

BIBO_50M_BASELINE = BiBoConfig(
    vocab_size=81000,               # QTK-81K tokenizer
    hidden_size=320,
    intermediate_size=1024,         # 3.2x hidden
    num_hidden_layers=12,           # first+last dense, 10 MoE
    num_attention_heads=5,          # hidden/64
    num_key_value_heads=1,          # GQA 5:1 (aggressive)
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — baseline (no PolyGLU)
    polyglu_expert_multiplier=1,    # 1 group = 3 SiLU experts
    special_expert_pairs=0,         # No Identity/Zero
    num_experts_per_tok=2,          # Top-2 routing
    use_shared_expert=True,
    shared_expert_type="mlp",       # SwiGLU shared expert
    # Router
    router_type="mlp",
    router_lambda=1.0,
    router_noise=0.0,               # Disabled for bench
    bias_update_threshold=100_000,
    bias_update_factor=1e-2,
    # Other
    moe_shared_scaling=0.40,         # Pre-computed: skip 10K MC simulation
    tie_word_embeddings=True,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    attention_bias=False,
)


def count_params(config):
    """Count total params for a BiBoConfig (quick, no training overhead)."""
    model = BiBoForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.model.embed_tokens.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())
    # If tied, lm_head shares weights with embed
    unique = total  # params() already deduplicates tied weights
    del model
    return {
        "total": total,
        "total_m": total / 1e6,
        "embed": embed,
        "lm_head": lm_head,
    }


def build_model(config=None):
    """Build BiBoForCausalLM from config. Default: BIBO_50M_BASELINE."""
    if config is None:
        config = BIBO_50M_BASELINE
    model = BiBoForCausalLM(config)
    return model, config


if __name__ == "__main__":
    stats = count_params(BIBO_50M_BASELINE)
    print(f"BiBo Baseline Config:")
    print(f"  Total params: {stats['total']:,} ({stats['total_m']:.2f}M)")
    print(f"  Embed params: {stats['embed']:,}")
    print(f"  LM head params: {stats['lm_head']:,}")
    print(f"  Hidden: {BIBO_50M_BASELINE.hidden_size}")
    print(f"  Layers: {BIBO_50M_BASELINE.num_hidden_layers}")
    print(f"  Experts: {BIBO_50M_BASELINE.num_routed_experts} routed + {BIBO_50M_BASELINE.num_shared_experts} shared")
    print(f"  Top-K: {BIBO_50M_BASELINE.num_experts_per_tok}")
