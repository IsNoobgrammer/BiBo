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
    vocab_size=81920,             # QTK-81K tokenizer: real len(tokenizer)=81920 (ids up to 81919 in the
                                  # packed corpus). NOT 81000 (that old-bench value overflows the embedding).
    hidden_size=512,
    num_hidden_layers=10,
    num_attention_heads=4,
    num_key_value_heads=2,        # GQA 2:1
    intermediate_size=1024,       # dense MLP (mlp_only_layers)
    moe_intermediate_size=768,
    num_experts=6,                # == BiBo num_routed_experts (polyglu_mult=2 -> 6 GLU experts; LCM of 2,3
                                  # so any enabled act subset {silu,relu2,normsilu} cycles evenly)
    num_experts_per_tok=2,        # top-k
    max_position_embeddings=2048,
    mlp_only_layers=[0, 9],       # first + last dense, rest MoE
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,     # SAME on both -> param match holds; tied -> ~137M total / ~71M active
    norm_topk_prob=False,
)

PARTIAL_ROPE = 0.334              # BiBo-min partial rotary; 1.0 == Qwen full RoPE (flip to isolate)


def make_qwen_config(attn_impl="sdpa", aux_coef=0.001, num_experts=None):
    from baseline.qwen3moe.config import Qwen3MoeConfig
    cfg = Qwen3MoeConfig(
        vocab_size=SHARED["vocab_size"], hidden_size=SHARED["hidden_size"],
        intermediate_size=SHARED["intermediate_size"], num_hidden_layers=SHARED["num_hidden_layers"],
        num_attention_heads=SHARED["num_attention_heads"], num_key_value_heads=SHARED["num_key_value_heads"],
        num_experts=num_experts or SHARED["num_experts"],   # == BiBo GLU count (polyglu_mult*3) -> param-matched
        num_experts_per_tok=SHARED["num_experts_per_tok"],
        moe_intermediate_size=SHARED["moe_intermediate_size"], norm_topk_prob=SHARED["norm_topk_prob"],
        max_position_embeddings=SHARED["max_position_embeddings"], mlp_only_layers=SHARED["mlp_only_layers"],
        rms_norm_eps=SHARED["rms_norm_eps"], rope_theta=SHARED["rope_theta"],
        tie_word_embeddings=SHARED["tie_word_embeddings"], router_aux_loss_coef=aux_coef,  # Switch aux LB loss (Qwen native)
    )
    cfg._attn_implementation = attn_impl        # "sdpa" | "flash_attention_4" (native HF dispatch)
    return cfg


def make_bibo_min_config(load_balance="bias", bias_update_threshold=10240, bias_update_factor=None,
                         polyglu_mult=2, special_pairs=0, router_type="mlp", kernel_size=3,
                         use_ssmax=False, use_xsa=False, balance_exclude_specials=False,
                         identity_expert=True, zero_expert=True):
    from src.configuration_bibo import BiBoConfig
    # DeepSeek-style aux-loss-free balancing pairs with SIGMOID gating (bias added to sigmoid scores);
    # with no balancing we use softmax (Qwen-matched). So gate_type follows load_balance.
    gate = "sigmoid" if load_balance == "bias" else "softmax"
    return BiBoConfig(
        load_balance_strategy=load_balance,         # "bias" (DeepSeek-style, sigmoid) | "none" (softmax, Qwen-matched)
        bias_update_threshold=bias_update_threshold,  # tokens between router-bias updates (only if load_balance="bias")
        bias_update_factor=bias_update_factor,        # None = auto Hill (~0.08 for 9 experts)
        balance_exclude_specials=balance_exclude_specials,  # ablation: freeze Identity/Zero bias (router learns their use)
        vocab_size=SHARED["vocab_size"], hidden_size=SHARED["hidden_size"],
        intermediate_size=SHARED["intermediate_size"], num_hidden_layers=SHARED["num_hidden_layers"],
        num_attention_heads=SHARED["num_attention_heads"], num_key_value_heads=SHARED["num_key_value_heads"],
        moe_intermediate_size=SHARED["moe_intermediate_size"], num_experts_per_tok=SHARED["num_experts_per_tok"],
        max_position_embeddings=SHARED["max_position_embeddings"], mlp_only_layers=SHARED["mlp_only_layers"],
        rms_norm_eps=SHARED["rms_norm_eps"], rope_theta=SHARED["rope_theta"],
        tie_word_embeddings=SHARED["tie_word_embeddings"], norm_topk_prob=SHARED["norm_topk_prob"],
        # --- the ablation delta: PolyGLU experts + partial RoPE ---
        polyglu_expert_multiplier=polyglu_mult,  # GLU experts = polyglu_mult*3 (silu/relu2/normsilu); == Qwen num_experts
        special_expert_pairs=special_pairs,      # per-type count of param-FREE special experts
        identity_expert=identity_expert,         # ablation: include Identity special expert(s) (code 3)
        zero_expert=zero_expert,                 # ablation: include Zero special expert(s)     (code 4)
        partial_rotary_factor=PARTIAL_ROPE,
        # --- everything else stripped to Qwen-equivalence ---
        use_xsa=use_xsa, use_ssmax=use_ssmax,    # ablation axes (default OFF): XSA + scalable-softmax
        add_full_attention_sink_bias=False, add_swa_attention_sink_bias=False,
        hybrid_layer_pattern=None,        # all-global attention (no SWA)
        router_type=router_type, gate_type=gate, router_activation="none",
        kernel_size=kernel_size,                 # conv-router kernel width (only used when router_type="conv")
        routed_scaling_factor=1.0,
        use_shared_expert=False,
    )


ARMS = {"qwen": make_qwen_config, "bibo_min": make_bibo_min_config}
