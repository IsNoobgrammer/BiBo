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


def swa_block_pattern(n_layers):
    """[global, swa, swa] x N, plus a final global layer.

    n_layers=10 -> [0,1,1, 0,1,1, 0,1,1, 0]: three blocks then a global tail. Global layers land on
    0 and 9, which are exactly SHARED['mlp_only_layers'] -- the dense-FFN layers are also the
    full-attention ones, so every block is (global attn + dense FFN) followed by two windowed MoE
    layers. Requires n_layers % 3 == 1 to come out even."""
    if n_layers % 3 != 1:
        raise ValueError(f"swa_block_pattern wants n_layers % 3 == 1 (got {n_layers}); the "
                         f"[G,S,S] block plus a global tail does not tile otherwise")
    return [0 if i % 3 == 0 else 1 for i in range(n_layers - 1)] + [0]


def hswa_windows(pattern, windows):
    """Per-layer sliding_window list for HIERARCHICAL SWA.

    `windows` is the cycle applied to the windowed layers WITHIN each block, in order. With
    pattern [0,1,1]*3+[0] and windows (128, 512): the first windowed layer of every block gets
    128, the second 512 -- refine locally, then widen. Global layers get 0 (ignored).

    Returned per LAYER, not per windowed layer, so the index stays layer_idx and cannot drift
    out of step with hybrid_layer_pattern.
    """
    out, k = [], 0
    for v in pattern:
        if not v:
            out.append(0)
            continue
        out.append(int(windows[k % len(windows)]))
        k += 1
    return out


def resolve_swa(swa_pattern, sliding_window, n_layers):
    """(--swa_pattern, --sliding_window) -> (hybrid_layer_pattern, sliding_window) for build_arm.

    Lives here rather than in train.py because run_eval.py has to reproduce a checkpoint's
    architecture EXACTLY. Two copies of this drift, and the failure is silent: the eval builds a
    subtly different model, load_state_dict swallows the mismatch, and the numbers are garbage
    with no error anywhere.
    """
    if swa_pattern in (None, "none"):
        pattern = None
    elif swa_pattern == "block3":
        pattern = swa_block_pattern(n_layers)
    else:
        pattern = [int(v) for v in str(swa_pattern).split(",")]
        if len(pattern) != n_layers:
            raise ValueError(f"swa_pattern has {len(pattern)} entries, model has {n_layers} layers")
    win = [int(v) for v in str(sliding_window).split(",")]
    if len(win) == 1:
        win = win[0]                                  # uniform: keep the plain int
    elif pattern is None:
        raise ValueError("sliding_window is a list but swa_pattern is none")
    else:
        win = hswa_windows(pattern, win)              # hierarchical: per-LAYER list
    return pattern, win


def make_bibo_min_config(bias_update_threshold=10240, bias_update_factor=None,
                         num_experts=None, special_pairs=0,
                         use_xsa=False, xsa_alpha_init=0.0,
                         pos_identity_expert=True, neg_identity_expert=True,
                         top_k=None, moe_intermediate_size=None, num_shared_experts=0,
                         hybrid_layer_pattern=None, sliding_window=128,
                         swa_qk_norm=True, attn_res="off", attn_res_sites=2):
    # attn_res: "off" = stable src model. Anything else routes to exp/ (Kimi K3 Attention
    # Residuals): "control" builds exp's model with residuals DISABLED, an int is the block size
    # in decoder layers (1 = per-layer / Full AttnRes, 3 = one block per [G,S,S]).
    if attn_res == "off":
        from src.configuration_bibo import BiBoConfig
        extra = {}
    else:
        from exp.configuration_bibo import BiBoConfig
        extra = {"attn_res_block_size": None if attn_res == "control" else int(attn_res),
                 "attn_res_sites": attn_res_sites}
    return BiBoConfig(
        **extra,
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
        use_xsa=use_xsa, xsa_alpha_init=xsa_alpha_init,
        hybrid_layer_pattern=hybrid_layer_pattern,
        sliding_window=sliding_window,
        # Only meaningful with SWA; global layers keep QK-norm regardless.
        swa_qk_norm=swa_qk_norm,
        use_shared_expert=bool(num_shared_experts),
        num_shared_experts=max(int(num_shared_experts), 1),
    )


ARMS = {"qwen": make_qwen_config, "bibo_min": make_bibo_min_config}
