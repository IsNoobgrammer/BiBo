"""
Monkey-patch BiBo/Qwen3 models to use Liger-Kernel Triton ops.

Uses linkedin/Liger-Kernel (production-grade, battle-tested):
- LigerRMSNormFunction: Fused RMSNorm (fp32 intermediate, output in input dtype)
- LigerRopeFunction: Fused RoPE (eliminates rotate_half intermediate)

These patches apply ONLY Liger ops (RMSNorm + RoPE) — BiBo's custom Triton kernels (MoE, dense
MLP, conv router, XSA, fused-CE) are applied by their own patch functions.

Usage:
    from src.kernels.patch import patch_bibo_with_liger
    model = BiBoForCausalLM(config).cuda()
    patch_bibo_with_liger(model)  # Done — 8-9x faster norms, 2-3x faster RoPE
    # (patch_bibo_with_triton is a deprecated alias of patch_bibo_with_liger)

Disable with --no_triton flag in bench scripts.
"""

from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction

__all__ = [
    'patch_bibo_with_liger', 'patch_qwen3_with_liger',
    'patch_bibo_with_triton', 'patch_qwen3_with_triton',  # deprecated aliases (Liger only)
    'unpatch_bibo', 'unpatch_qwen3',
]


# ─────────────────────────────────────────────────────────────
# Patched forward methods
# ─────────────────────────────────────────────────────────────

def _liger_rmsnorm_forward(self, hidden_states):
    """Drop-in replacement for BiBoRMSNorm/Qwen3RMSNorm using Liger-Kernel."""
    return LigerRMSNormFunction.apply(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        0.0,       # offset (0 for standard RMSNorm)
        "llama",   # casting_mode: upcast to fp32, output in input dtype
        False,     # in_place: False to avoid mutating input
    )


def _liger_rope_apply(q, k, cos, sin, unsqueeze_dim=1):
    """Drop-in replacement for apply_rotary_pos_emb using Liger-Kernel."""
    return LigerRopeFunction.apply(q, k, cos, sin, None, unsqueeze_dim)


# ─────────────────────────────────────────────────────────────
# BiBo patching
# ─────────────────────────────────────────────────────────────

def patch_bibo_with_liger(model):
    """
    Patch a BiBoForCausalLM to use Liger-Kernel Triton ops.

    Replaces:
      - All BiBoRMSNorm.forward -> LigerRMSNormFunction
      - apply_rotary_pos_emb -> LigerRopeFunction

    Args:
        model: BiBoForCausalLM or BiBoModel instance

    Returns:
        model (same instance, modified in-place)
    """
    from src.modeling.norm import BiBoRMSNorm
    import src.modeling.attn.base as attn_module
    import src.modeling.embed as embed_module

    # Patch all RMSNorm instances
    for module in model.modules():
        if isinstance(module, BiBoRMSNorm):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _liger_rmsnorm_forward.__get__(module, BiBoRMSNorm)

    # Patch RoPE in the attn module (where it's called from)
    if not hasattr(attn_module, '_original_apply_rotary_pos_emb'):
        attn_module._original_apply_rotary_pos_emb = attn_module.apply_rotary_pos_emb
    attn_module.apply_rotary_pos_emb = _liger_rope_apply

    # Also patch embed module
    if not hasattr(embed_module, '_original_apply_rotary_pos_emb'):
        embed_module._original_apply_rotary_pos_emb = embed_module.apply_rotary_pos_emb
    embed_module.apply_rotary_pos_emb = _liger_rope_apply

    model._triton_patched = True
    return model


def unpatch_bibo(model):
    """Restore original PyTorch implementations."""
    from src.modeling.norm import BiBoRMSNorm
    import src.modeling.attn.base as attn_module
    import src.modeling.embed as embed_module

    for module in model.modules():
        if isinstance(module, BiBoRMSNorm) and hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward

    if hasattr(attn_module, '_original_apply_rotary_pos_emb'):
        attn_module.apply_rotary_pos_emb = attn_module._original_apply_rotary_pos_emb
        del attn_module._original_apply_rotary_pos_emb

    if hasattr(embed_module, '_original_apply_rotary_pos_emb'):
        embed_module.apply_rotary_pos_emb = embed_module._original_apply_rotary_pos_emb
        del embed_module._original_apply_rotary_pos_emb

    model._triton_patched = False
    return model


# ─────────────────────────────────────────────────────────────
# Qwen3 Fused Cross-Entropy patch
# ─────────────────────────────────────────────────────────────

def patch_qwen3_fused_ce(model):
    """
    Patch Qwen3MoeForCausalLM to use BiBo's OWN fused-linear-CE Triton kernel
    (src/kernels/fused_ce.py) — NOT Liger's chunked CE (3-4x slower at vocab<=81k).

    Replaces standard logits+CE with the fused kernel that never materializes the (N,V) logits:
    forward 2.1x, comparable-to-faster fwd+bwd, ~0.1-0.5x memory, enabling at the 16k step.
    fp16 (matches autocast); the kernel does fp32 accumulation internally.
    """
    import torch
    from src.kernels.fused_ce import fused_linear_cross_entropy

    def _fused_ce_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                          past_key_values=None, inputs_embeds=None, labels=None,
                          use_cache=None, output_router_logits=None,
                          logits_to_keep=0, **kwargs):
        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else self.config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        loss = None
        logits = None

        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_hidden = shift_hidden.view(-1, hidden_states.shape[-1])
            shift_labels = shift_labels.view(-1).to(shift_hidden.device)
            # weight cast to hidden dtype (fp16 under autocast); our kernel does fp32 accum inside.
            lm_weight = self.lm_head.weight.to(shift_hidden.dtype)
            loss = fused_linear_cross_entropy(shift_hidden, lm_weight, shift_labels)
        else:
            # No labels → eval/inference: must materialize logits (fused CE only runs WITH labels).
            # logits_to_keep>0 = generation (last-k); else full-sequence scoring (HellaSwag/eval).
            if logits_to_keep and (not isinstance(logits_to_keep, int) or logits_to_keep > 0):
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:
                logits = self.lm_head(hidden_states)

        aux_loss = None
        if output_router_logits:
            from baseline.qwen3moe.modeling import load_balancing_loss_func
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        from baseline.qwen3moe.modeling import MoeCausalLMOutputWithPast
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
    if not hasattr(Qwen3MoeForCausalLM, '_original_forward'):
        Qwen3MoeForCausalLM._original_forward = Qwen3MoeForCausalLM.forward
    Qwen3MoeForCausalLM.forward = _fused_ce_forward
    return model
# ─────────────────────────────────────────────────────────────
# Qwen3 / Qwen3MoE patching
# ─────────────────────────────────────────────────────────────

def patch_qwen3_with_liger(model):
    """
    Patch Qwen3ForCausalLM or Qwen3MoeForCausalLM to use Liger-Kernel.

    Handles both Qwen3RMSNorm and Qwen3MoeRMSNorm.
    """
    norm_classes = []
    try:
        from baseline.qwen3.modeling import Qwen3RMSNorm
        norm_classes.append(Qwen3RMSNorm)
    except ImportError:
        pass
    try:
        from baseline.qwen3moe.modeling import Qwen3MoeRMSNorm
        norm_classes.append(Qwen3MoeRMSNorm)
    except ImportError:
        pass

    if not norm_classes:
        raise ImportError("Could not import any Qwen3 RMSNorm class")

    norm_classes = tuple(norm_classes)

    for module in model.modules():
        if isinstance(module, norm_classes):
            if not hasattr(module, '_original_forward'):
                module._original_forward = module.forward
            module.forward = _liger_rmsnorm_forward.__get__(module, type(module))

    # Patch RoPE in qwen3 modeling modules
    try:
        import baseline.qwen3.modeling as qwen3_mod
        if not hasattr(qwen3_mod, '_original_apply_rotary_pos_emb'):
            qwen3_mod._original_apply_rotary_pos_emb = qwen3_mod.apply_rotary_pos_emb
        qwen3_mod.apply_rotary_pos_emb = _liger_rope_apply
    except (ImportError, AttributeError):
        pass

    try:
        import baseline.qwen3moe.modeling as qwen3moe_mod
        if not hasattr(qwen3moe_mod, '_original_apply_rotary_pos_emb'):
            qwen3moe_mod._original_apply_rotary_pos_emb = qwen3moe_mod.apply_rotary_pos_emb
        qwen3moe_mod.apply_rotary_pos_emb = _liger_rope_apply
    except (ImportError, AttributeError):
        pass

    model._triton_patched = True
    return model


def unpatch_qwen3(model):
    """Restore original Qwen3 implementations."""
    norm_classes = []
    try:
        from baseline.qwen3.modeling import Qwen3RMSNorm
        norm_classes.append(Qwen3RMSNorm)
    except ImportError:
        pass
    try:
        from baseline.qwen3moe.modeling import Qwen3MoeRMSNorm
        norm_classes.append(Qwen3MoeRMSNorm)
    except ImportError:
        pass

    norm_classes = tuple(norm_classes) if norm_classes else ()

    for module in model.modules():
        if norm_classes and isinstance(module, norm_classes) and hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            del module._original_forward

    try:
        import baseline.qwen3.modeling as qwen3_mod
        if hasattr(qwen3_mod, '_original_apply_rotary_pos_emb'):
            qwen3_mod.apply_rotary_pos_emb = qwen3_mod._original_apply_rotary_pos_emb
            del qwen3_mod._original_apply_rotary_pos_emb
    except ImportError:
        pass

    try:
        import baseline.qwen3moe.modeling as qwen3moe_mod
        if hasattr(qwen3moe_mod, '_original_apply_rotary_pos_emb'):
            qwen3moe_mod.apply_rotary_pos_emb = qwen3moe_mod._original_apply_rotary_pos_emb
            del qwen3moe_mod._original_apply_rotary_pos_emb
    except ImportError:
        pass

    model._triton_patched = False
    return model


# ─────────────────────────────────────────────────────────────
# Deprecated aliases — these patches apply ONLY Liger ops (RMSNorm + RoPE),
# not BiBo's custom Triton kernels. Renamed to *_with_liger for honesty;
# old *_with_triton names kept for back-compat. Prefer the new names.
# ─────────────────────────────────────────────────────────────
patch_bibo_with_triton = patch_bibo_with_liger
patch_qwen3_with_triton = patch_qwen3_with_liger
