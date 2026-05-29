"""
BiBo Benchmark — Model Configuration

Baseline BiBo ~50M params:
- PolyGLU layout: 6 routed (SiLU, ReLU², Tanh × 2) + Identity + Zero = 8 routed
- MLP router, sigmoid gate, shared expert ON
- GQA attention with SSMax
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.modeling.ffn.moe import BiBoMoELayer


# ─────────────────────────────────────────────────────────────
# Baseline BiBo ~50M (8 routed experts, top-2, shared expert)
# ─────────────────────────────────────────────────────────────

BIBO_50M_BASELINE = BiBoConfig(
    vocab_size=81000,               # QTK-81K tokenizer
    hidden_size=320,
    intermediate_size=1024,         # 3.2x hidden (dense MLP)
    num_hidden_layers=10,           # first 2 + last dense, 7 MoE
    num_attention_heads=5,          # hidden/64
    num_key_value_heads=1,          # GQA 5:1 (aggressive)
    max_position_embeddings=2048,
    use_ssmax=True,
    # MoE — PolyGLU (8 routed experts)
    polyglu_expert_multiplier=2,    # 2 groups × 3 = 6 GLU (SiLU, ReLU², Tanh)
    special_expert_pairs=1,         # + Identity + Zero = 8 routed total
    num_experts_per_tok=2,          # Top-2 routing
    moe_intermediate_size=768,      # Per-expert FFN size (tuned for ~50M)
    use_shared_expert=True,
    shared_expert_type="mlp",       # SwiGLU shared expert
    mlp_only_layers=[0, 9],          # First + last layer dense, rest MoE
    # Router
    router_type="mlp",
    router_lambda=1.0,
    router_noise=0.0,               # Disabled for bench
    bias_update_threshold=100_000,
    bias_update_factor=1e-2,
    # Other
    moe_shared_scaling=0.40,        # Pre-computed: skip 10K MC simulation
    tie_word_embeddings=True,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
    attention_bias=False,
)


def count_params(config):
    """Count total and active params for a BiBoConfig."""
    model = BiBoForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.model.embed_tokens.parameters())

    attn_total = 0
    dense_total = 0
    moe_routed_total = 0
    moe_shared_total = 0
    moe_router_total = 0
    num_moe = 0
    num_dense = 0

    for layer in model.model.layers:
        # Attention + norms
        attn_total += sum(p.numel() for p in layer.self_attn.parameters())
        attn_total += sum(p.numel() for p in layer.input_layernorm.parameters())
        attn_total += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

        if isinstance(layer.mlp, BiBoMoELayer):
            num_moe += 1
            moe = layer.mlp
            moe_router_total += sum(p.numel() for p in moe.gate.parameters())
            # Shared expert(s)
            for se in moe.shared_experts_list:
                moe_shared_total += sum(p.numel() for p in se.parameters())
            # Fused routed experts (gate_up_proj + down_proj)
            moe_routed_total += sum(p.numel() for p in moe.experts.parameters())
        else:
            num_dense += 1
            dense_total += sum(p.numel() for p in layer.mlp.parameters())

    # Active params per token:
    #   embed + all_attn + dense_mlp + (top_k/num_routed × routed_per_layer × moe_layers) + shared + router
    routed_per_layer = moe_routed_total / max(num_moe, 1)
    active_routed = routed_per_layer * (config.num_experts_per_tok / config.num_routed_experts) * num_moe
    active_total = embed + attn_total + dense_total + active_routed + moe_shared_total + moe_router_total

    del model
    return {
        "total": total,
        "total_m": total / 1e6,
        "active": int(active_total),
        "active_m": active_total / 1e6,
        "ratio": total / max(active_total, 1),
        "embed": embed,
        "attn": attn_total,
        "dense": dense_total,
        "moe_routed": moe_routed_total,
        "moe_shared": moe_shared_total,
        "moe_router": moe_router_total,
        "num_moe_layers": num_moe,
        "num_dense_layers": num_dense,
        "routed_per_layer": int(routed_per_layer),
    }


def build_model(config=None):
    """Build BiBoForCausalLM from config. Default: BIBO_50M_BASELINE."""
    if config is None:
        config = BIBO_50M_BASELINE
    model = BiBoForCausalLM(config)
    return model, config


if __name__ == "__main__":
    stats = count_params(BIBO_50M_BASELINE)
    cfg = BIBO_50M_BASELINE
    print(f"BiBo Baseline Config:")
    print(f"  Total params:     {stats['total']:>12,} ({stats['total_m']:.2f}M)")
    print(f"  Active params:    {stats['active']:>12,} ({stats['active_m']:.2f}M)")
    print(f"  Ratio total/active: {stats['ratio']:.2f}x")
    print()
    print(f"  Hidden: {cfg.hidden_size}")
    print(f"  Layers: {cfg.num_hidden_layers} ({stats['num_moe_layers']} MoE + {stats['num_dense_layers']} dense)")
    print(f"  Experts: {cfg.num_routed_experts} routed + {cfg.num_shared_experts} shared = {cfg.num_experts} total")
    print(f"    Layout: {cfg.polyglu_expert_multiplier}×(SiLU,ReLU²,Tanh) + {cfg.special_expert_pairs}×(Identity,Zero)")
    print(f"  Top-K: {cfg.num_experts_per_tok}")
    print(f"  Router: {cfg.router_type}, λ={cfg.router_lambda}")
    print()
    print(f"  Breakdown:")
    print(f"    Embedding:      {stats['embed']:>12,}")
    print(f"    Attention:      {stats['attn']:>12,}")
    print(f"    Dense MLP:      {stats['dense']:>12,}")
    print(f"    MoE router:     {stats['moe_router']:>12,}")
    print(f"    MoE routed(all):{stats['moe_routed']:>12,}  ({stats['routed_per_layer']:,}/layer)")
    print(f"    MoE shared(all):{stats['moe_shared']:>12,}")
