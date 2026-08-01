"""Arm configs for the BiBo-min vs Qwen ablation -- parameter-matched by construction.

Both arms share identical dims / experts / top_k. Because a GLU expert == SwiGLU in params and
partial-vs-full RoPE is parameter-free, the two models have the SAME parameter count exactly.

  'qwen'     : stock Qwen3MoE (SwiGLU experts, full RoPE, softmax router).
  'bibo_min' : BiBo stripped to Qwen-equivalence EXCEPT radial-NormSiLU experts + partial RoPE.

EXPERT COUNTING CHANGED Aug 1 2026. num_routed_experts is now the TOTAL and the GLU count is
DERIVED as total - 2*special_pairs. It used to be the other way round (polyglu_mult*3 GLU experts
PLUS the specials), so the same numbers now build a DIFFERENT model: at num_experts=6,
special_pairs=1 you get 4 GLU + 2 specials, where the old code gave 6 GLU + 2 specials = 8 routed.
Qwen's num_experts is matched to BiBo's GLU count, so the param match still holds -- but a run tagged
with an expert count before this date is not comparable to one after it.
"""
from . import _paths  # noqa: F401  (sys.path bootstrap)

SHARED = dict(
    vocab_size=81920,             # QTK-81K tokenizer: real len(tokenizer)=81920. NOT 81000 (overflows).
    hidden_size=512,
    num_hidden_layers=10,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=1024,
    moe_intermediate_size=768,
    num_experts=6,                # TOTAL routed experts (GLU + specials)
    num_experts_per_tok=2,
    max_position_embeddings=2048,
    mlp_only_layers=[0, 9],
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    norm_topk_prob="sum",         # "sum" (MiMo div-sum) | "softmax" | False
)

PARTIAL_ROPE = 0.334              # BiBo-min partial rotary; 1.0 == Qwen full RoPE


def glu_count(num_experts, special_pairs, pos_identity=True, neg_identity=True):
    """GLU experts left after the param-free specials take their slots. This is the number Qwen has
    to be built with for the param match, and it is what a run's expert count should be read as."""
    n_special = (special_pairs if pos_identity else 0) + (special_pairs if neg_identity else 0)
    return num_experts - n_special


def make_qwen_config(attn_impl="sdpa", aux_coef=0.001, num_experts=None):
    from baseline.qwen3moe.config import Qwen3MoeConfig
    cfg = Qwen3MoeConfig(
        vocab_size=SHARED["vocab_size"], hidden_size=SHARED["hidden_size"],
        intermediate_size=SHARED["intermediate_size"], num_hidden_layers=SHARED["num_hidden_layers"],
        num_attention_heads=SHARED["num_attention_heads"], num_key_value_heads=SHARED["num_key_value_heads"],
        num_experts=num_experts or SHARED["num_experts"],
        num_experts_per_tok=SHARED["num_experts_per_tok"],
        moe_intermediate_size=SHARED["moe_intermediate_size"],
        norm_topk_prob=bool(SHARED["norm_topk_prob"]),
        max_position_embeddings=SHARED["max_position_embeddings"], mlp_only_layers=SHARED["mlp_only_layers"],
        rms_norm_eps=SHARED["rms_norm_eps"], rope_theta=SHARED["rope_theta"],
        tie_word_embeddings=SHARED["tie_word_embeddings"], router_aux_loss_coef=aux_coef,
    )
    cfg._attn_implementation = attn_impl
    return cfg


def make_bibo_min_config(bias_update_threshold=10240, bias_update_factor=None,
                         num_experts=None, special_pairs=0,
                         use_ssmax=False, use_xsa=False,
                         pos_identity_expert=True, neg_identity_expert=True,
                         top_k=None, moe_intermediate_size=None, num_shared_experts=0):
    from src.configuration_bibo import BiBoConfig
    return BiBoConfig(
        bias_update_threshold=bias_update_threshold,
        bias_update_factor=bias_update_factor,      # None -> BiBoConfig default (0.4, proportional)
        vocab_size=SHARED["vocab_size"], hidden_size=SHARED["hidden_size"],
        intermediate_size=SHARED["intermediate_size"], num_hidden_layers=SHARED["num_hidden_layers"],
        num_attention_heads=SHARED["num_attention_heads"], num_key_value_heads=SHARED["num_key_value_heads"],
        # Raising top_k WITHOUT shrinking moe_intermediate_size multiplies active expert FLOPs by the
        # same factor -- pass both to hold compute constant.
        moe_intermediate_size=(moe_intermediate_size or SHARED["moe_intermediate_size"]),
        num_experts_per_tok=(top_k or SHARED["num_experts_per_tok"]),
        max_position_embeddings=SHARED["max_position_embeddings"], mlp_only_layers=SHARED["mlp_only_layers"],
        rms_norm_eps=SHARED["rms_norm_eps"], rope_theta=SHARED["rope_theta"],
        tie_word_embeddings=SHARED["tie_word_embeddings"], norm_topk_prob=SHARED["norm_topk_prob"],
        # --- the ablation delta: radial-NormSiLU experts + partial RoPE ---
        num_routed_experts=(num_experts or SHARED["num_experts"]),
        special_expert_pairs=special_pairs,
        pos_identity_expert=pos_identity_expert,
        neg_identity_expert=neg_identity_expert,
        partial_rotary_factor=PARTIAL_ROPE,
        # --- everything else stripped to Qwen-equivalence ---
        use_xsa=use_xsa, use_ssmax=use_ssmax,
        add_full_attention_sink_bias=False, add_swa_attention_sink_bias=False,
        hybrid_layer_pattern=None,
        use_shared_expert=bool(num_shared_experts),
        num_shared_experts=max(int(num_shared_experts), 1),
    )


ARMS = {"qwen": make_qwen_config, "bibo_min": make_bibo_min_config}
