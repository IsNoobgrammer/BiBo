"""Arm configs for the BiBo-min vs Qwen ablation — parameter-matched by construction.

Both arms share identical dims / experts / top_k. Because PolyGLU==SwiGLU in params and
partial-vs-full RoPE is parameter-free, the two models have the SAME parameter count exactly.

Arms (2, bundled per the design):
  'qwen'     : stock Qwen3MoE (SwiGLU experts, full RoPE, softmax router).
  'bibo_min' : BiBo stripped to Qwen-equivalence EXCEPT PolyGLU experts + partial RoPE.
               no SWA, no conv router, no XSA, no SSMax, no sinks, no shared expert,
               softmax router with NO load-balancing (matches Qwen with aux-loss OFF).

Every knob here is a plain dict field -> swappable. Flip PARTIAL_ROPE / router / balancing
to re-scope the ablation (e.g. a 3rd 'isolate RoPE' arm) without touching model code.
"""
from . import _paths  # noqa: F401  (sys.path bootstrap)

# ---- shared, matched dimensions (team-standard 137M total / 71M active) ----
SHARED = dict(
    vocab_size=81000,             # QTK-81K tokenizer
    hidden_size=512,
    num_hidden_layers=10,
    num_attention_heads=4,
    num_key_value_heads=2,        # GQA 2:1
    intermediate_size=1024,       # dense MLP (mlp_only_layers)
    moe_intermediate_size=768,
    num_experts=9,                # == BiBo num_routed_experts
    num_experts_per_tok=2,        # top-k
    max_position_embeddings=2048,
    mlp_only_layers=[0, 9],       # first + last dense, rest MoE
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,     # SAME on both -> param match holds; tied -> ~137M total / ~71M active
    norm_topk_prob=False,
)

PARTIAL_ROPE = 0.334              # BiBo-min partial rotary; 1.0 == Qwen full RoPE (flip to isolate)


def make_qwen_config(attn_impl="sdpa"):
    from baseline.qwen3moe.config import Qwen3MoeConfig
    cfg = Qwen3MoeConfig(
        vocab_size=SHARED["vocab_size"], hidden_size=SHARED["hidden_size"],
        intermediate_size=SHARED["intermediate_size"], num_hidden_layers=SHARED["num_hidden_layers"],
        num_attention_heads=SHARED["num_attention_heads"], num_key_value_heads=SHARED["num_key_value_heads"],
        num_experts=SHARED["num_experts"], num_experts_per_tok=SHARED["num_experts_per_tok"],
        moe_intermediate_size=SHARED["moe_intermediate_size"], norm_topk_prob=SHARED["norm_topk_prob"],
        max_position_embeddings=SHARED["max_position_embeddings"], mlp_only_layers=SHARED["mlp_only_layers"],
        rms_norm_eps=SHARED["rms_norm_eps"], rope_theta=SHARED["rope_theta"],
        tie_word_embeddings=SHARED["tie_word_embeddings"], router_aux_loss_coef=0.0,   # aux loss OFF (clean match)
    )
    cfg._attn_implementation = attn_impl        # "sdpa" | "flash_attention_4" (native HF dispatch)
    return cfg


def make_bibo_min_config():
    from src.configuration_bibo import BiBoConfig
    return BiBoConfig(
        vocab_size=SHARED["vocab_size"], hidden_size=SHARED["hidden_size"],
        intermediate_size=SHARED["intermediate_size"], num_hidden_layers=SHARED["num_hidden_layers"],
        num_attention_heads=SHARED["num_attention_heads"], num_key_value_heads=SHARED["num_key_value_heads"],
        moe_intermediate_size=SHARED["moe_intermediate_size"], num_experts_per_tok=SHARED["num_experts_per_tok"],
        max_position_embeddings=SHARED["max_position_embeddings"], mlp_only_layers=SHARED["mlp_only_layers"],
        rms_norm_eps=SHARED["rms_norm_eps"], rope_theta=SHARED["rope_theta"],
        tie_word_embeddings=SHARED["tie_word_embeddings"], norm_topk_prob=SHARED["norm_topk_prob"],
        # --- the ablation delta: PolyGLU experts + partial RoPE ---
        polyglu_expert_multiplier=3,      # 9 PolyGLU experts (silu/relu2/normsilu cycled) == Qwen's 9
        special_expert_pairs=0,           # pure PolyGLU, no Identity/Zero specials
        partial_rotary_factor=PARTIAL_ROPE,
        # --- everything else stripped to Qwen-equivalence ---
        use_xsa=False, use_ssmax=False,
        add_full_attention_sink_bias=False, add_swa_attention_sink_bias=False,
        hybrid_layer_pattern=None,        # all-global attention (no SWA)
        router_type="mlp", gate_type="softmax", router_activation="none",
        load_balance_strategy="none", routed_scaling_factor=1.0,
        use_shared_expert=False,
    )


ARMS = {"qwen": make_qwen_config, "bibo_min": make_bibo_min_config}
