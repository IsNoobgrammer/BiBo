"""
BiBo Benchmark — Unified Model Builder

Builds BiBo or Qwen3MoE from YAML config dict (PyTorch eager; torch.compile handled in train.py).
"""

import sys
import os
import yaml
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.modeling.ffn.moe import BiBoMoELayer


def build_model_from_config(cfg: dict):
    """
    Build model from YAML config dict.

    Returns (model, config) where config is BiBoConfig or Qwen3MoeConfig.
    """
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]

    if model_type == "bibo":
        config = BiBoConfig(
            vocab_size=model_cfg["vocab_size"],
            hidden_size=model_cfg["hidden_size"],
            intermediate_size=model_cfg["intermediate_size"],
            num_hidden_layers=model_cfg["num_hidden_layers"],
            num_attention_heads=model_cfg["num_attention_heads"],
            num_key_value_heads=model_cfg["num_key_value_heads"],
            max_position_embeddings=model_cfg["max_position_embeddings"],
            mlp_only_layers=model_cfg.get("mlp_only_layers", None),
            use_ssmax=model_cfg.get("use_ssmax", True),
            use_xsa=model_cfg.get("use_xsa", True),
            partial_rotary_factor=model_cfg.get("partial_rotary_factor", 0.334),
            hybrid_layer_pattern=model_cfg.get("hybrid_layer_pattern", None),
            sliding_window=model_cfg.get("sliding_window", 128),
            add_swa_attention_sink_bias=model_cfg.get("add_swa_attention_sink_bias", True),
            add_full_attention_sink_bias=model_cfg.get("add_full_attention_sink_bias", False),
            polyglu_expert_multiplier=model_cfg.get("polyglu_expert_multiplier", 2),
            special_expert_pairs=model_cfg.get("special_expert_pairs", 1),
            num_experts_per_tok=model_cfg.get("num_experts_per_tok", 2),
            moe_intermediate_size=model_cfg.get("moe_intermediate_size", None),
            use_shared_expert=model_cfg.get("use_shared_expert", False),
            router_type=model_cfg.get("router_type", "mlp"),
            router_noise=model_cfg.get("router_noise", 0.0),
            bias_update_threshold=model_cfg.get("bias_update_threshold", 100000),
            bias_update_factor=model_cfg.get("bias_update_factor", 0.01),
            tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
            rope_theta=model_cfg.get("rope_theta", 1e7),
            rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-6),
            attention_dropout=model_cfg.get("attention_dropout", 0.0),
            attention_bias=model_cfg.get("attention_bias", False),
        )
        model = BiBoForCausalLM(config)

    elif model_type == "qwen3moe":
        from baseline.qwen3moe.config import Qwen3MoeConfig
        from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(
            vocab_size=model_cfg["vocab_size"],
            hidden_size=model_cfg["hidden_size"],
            intermediate_size=model_cfg["intermediate_size"],
            num_hidden_layers=model_cfg["num_hidden_layers"],
            num_attention_heads=model_cfg["num_attention_heads"],
            num_key_value_heads=model_cfg["num_key_value_heads"],
            max_position_embeddings=model_cfg["max_position_embeddings"],
            num_experts=model_cfg.get("num_experts", 8),
            num_experts_per_tok=model_cfg.get("num_experts_per_tok", 2),
            moe_intermediate_size=model_cfg.get("moe_intermediate_size", 768),
            decoder_sparse_step=model_cfg.get("decoder_sparse_step", 1),
            norm_topk_prob=model_cfg.get("norm_topk_prob", False),
            router_aux_loss_coef=model_cfg.get("router_aux_loss_coef", 0.01),
            tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-6),
            attention_dropout=model_cfg.get("attention_dropout", 0.0),
            attention_bias=model_cfg.get("attention_bias", False),
            use_sliding_window=model_cfg.get("use_sliding_window", False),
            mlp_only_layers=model_cfg.get("mlp_only_layers", None),
        )
        # Enable SDPA with GQA (same as BiBo's native SDPA)
        config._attn_implementation = "sdpa"
        model = Qwen3MoeForCausalLM(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def count_params(model, config) -> dict:
    """
    Count total and active params for any model type.
    Handles both BiBo (PolyGLU, special experts) and Qwen3MoE.
    """
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.model.embed_tokens.parameters())

    attn_total = 0
    dense_total = 0
    moe_routed_total = 0
    moe_router_total = 0
    num_moe = 0
    num_dense = 0

    model_type = getattr(config, 'model_type', 'bibo')

    for layer in model.model.layers:
        attn_total += sum(p.numel() for p in layer.self_attn.parameters())
        attn_total += sum(p.numel() for p in layer.input_layernorm.parameters())
        attn_total += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

        if isinstance(layer.mlp, BiBoMoELayer):
            num_moe += 1
            moe_router_total += sum(p.numel() for p in layer.mlp.gate.parameters())
            moe_routed_total += sum(p.numel() for p in layer.mlp.experts.parameters())
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            num_moe += 1
            moe_router_total += sum(p.numel() for p in layer.mlp.gate.parameters())
            moe_routed_total += sum(p.numel() for p in layer.mlp.experts.parameters())
        else:
            num_dense += 1
            dense_total += sum(p.numel() for p in layer.mlp.parameters())

    routed_per_layer = moe_routed_total // max(num_moe, 1)

    if model_type == "bibo":
        # Denominator = GLU-expert count, NOT num_routed_experts. Params live only in the GLU experts
        # (routed_per_layer above sums just those); the param-free Identity/Zero specials must be
        # excluded or active under-counts vs an all-GLU MoE — apples-to-apples, BiBo's 2-of-9-GLU ==
        # Qwen's 2-of-9. This is an UPPER BOUND: real active is lower whenever the router picks a
        # free special (that capability is BiBo's, and it only ever makes BiBo cheaper, not costlier).
        num_glu = config.polyglu_expert_multiplier * 3
        top_k = config.num_experts_per_tok
        active_routed = routed_per_layer * (top_k / max(num_glu, 1)) * num_moe
    else:
        num_routed = getattr(config, 'num_experts', 8)
        top_k = config.num_experts_per_tok
        active_routed = routed_per_layer * (top_k / num_routed) * num_moe

    active_total = embed + attn_total + dense_total + active_routed + moe_router_total

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
        "moe_router": moe_router_total,
        "num_moe_layers": num_moe,
        "num_dense_layers": num_dense,
        "routed_per_layer": int(routed_per_layer),
    }


def resize_embeddings(model, config, target_vocab_size):
    """Resize embeddings to cover all token IDs in dataset."""
    current_vocab = config.vocab_size
    if target_vocab_size != current_vocab:
        model.resize_token_embeddings(target_vocab_size)
        config.vocab_size = target_vocab_size
        print(f"  Resized embeddings: {current_vocab} -> {target_vocab_size}")
