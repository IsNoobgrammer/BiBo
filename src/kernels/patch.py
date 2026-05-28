"""
Monkey-patch BiBo/Qwen3 models to use Liger-Kernel Triton ops.

Uses linkedin/Liger-Kernel (production-grade, battle-tested):
- LigerRMSNormFunction: Fused RMSNorm (fp32 intermediate, output in input dtype)
- LigerRopeFunction: Fused RoPE (eliminates rotate_half intermediate)

Usage:
    from src.kernels.patch import patch_bibo_with_triton
    model = BiBoForCausalLM(config).cuda()
    patch_bibo_with_triton(model)  # Done — 8-9x faster norms, 2-3x faster RoPE

Disable with --no_triton flag in bench scripts.
"""

from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction

__all__ = ['patch_bibo_with_triton', 'patch_qwen3_with_triton', 'unpatch_bibo', 'unpatch_qwen3']


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

def patch_bibo_with_triton(model):
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
# Qwen3 / Qwen3MoE patching
# ─────────────────────────────────────────────────────────────

def patch_qwen3_with_triton(model):
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
