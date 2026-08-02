"""Model builder + param counter for the ablation. Swappable: build_arm(name) is the only entry."""
from . import _paths  # noqa: F401
import torch
from .configs import ARMS, SHARED, glu_count, make_qwen_config, make_bibo_min_config
from . import patches


def build_arm(arm, device="cuda", dtype=torch.float32, attn_impl="sdpa",
              bias_update_threshold=10240, bias_update_factor=None, aux_coef=0.001,
              num_experts=None, special_pairs=0,
              use_xsa=False, xsa_alpha_init=0.0,
              pos_identity_expert=True, neg_identity_expert=True,
              top_k=None, moe_intermediate_size=None, num_shared_experts=0,
              hybrid_layer_pattern=None, sliding_window=128, swa_sink=True,
              swa_qk_norm=True, attn_res="off"):
    """arm in {'qwen','bibo_min'} -> (model, config). Params in `dtype` (fp32 master; bf16 via autocast).
    Balancing, each native: BiBo router-bias updates; Qwen Switch aux loss (aux_coef).
    PARAM MATCH: Qwen's num_experts is set to BiBo's GLU count, which is num_experts MINUS the
    param-free ±Identity specials -- see configs.glu_count."""
    eff = patches.resolve_attn(attn_impl)
    n_total = num_experts or SHARED["num_experts"]
    n_glu = glu_count(n_total, special_pairs, pos_identity_expert, neg_identity_expert)
    if n_glu < 1:
        raise ValueError(f"{n_total} routed experts minus {special_pairs} special pair(s) leaves "
                         f"{n_glu} GLU experts; raise --experts or lower --special_pairs")
    if arm == "qwen":
        from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
        cfg = make_qwen_config(eff, aux_coef=aux_coef, num_experts=n_glu)
        model = Qwen3MoeForCausalLM(cfg)
    elif arm == "bibo_min":
        # exp/ reimplements only the residual topology (decoder layer + trunk); it imports
        # BiBoAttention/BiBoMoELayer from src, so SWA, per-layer windows, XSA and swa_qk_norm
        # compose with AttnRes for free. The 'moe' Triton patch is installed on the src class,
        # so it applies to the exp model too.
        if attn_res == "off":
            from src.modeling.models import BiBoForCausalLM
        else:
            from exp.modeling_bibo import BiBoForCausalLM
        cfg = make_bibo_min_config(bias_update_threshold, bias_update_factor,
                                   num_experts=n_total, special_pairs=special_pairs,
                                   use_xsa=use_xsa,
                                   xsa_alpha_init=xsa_alpha_init,
                                   pos_identity_expert=pos_identity_expert,
                                   neg_identity_expert=neg_identity_expert,
                                   top_k=top_k, moe_intermediate_size=moe_intermediate_size,
                                   num_shared_experts=num_shared_experts,
                                   hybrid_layer_pattern=hybrid_layer_pattern,
                                   sliding_window=sliding_window, swa_sink=swa_sink,
                                   swa_qk_norm=swa_qk_norm, attn_res=attn_res)
        model = BiBoForCausalLM(cfg)
        if eff.startswith("flash"):
            patches.patch_bibo_flash()
    else:
        raise ValueError(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    return model.to(device=device, dtype=dtype), cfg


def count_params(model, top_k=None, num_experts=None):
    """Return (total, trainable, active). trainable excludes inert (requires_grad=False) params like
    BiBo's zero-init router bias; the ablation is matched on trainable/active params. active discounts
    inactive experts (3D stacked expert tensors).

    top_k comes from the MODEL'S OWN CONFIG, not from SHARED. It used to fall back to
    SHARED["num_experts_per_tok"], so every --top_k run reported the top_k=2 active count.

    NOTE it is an UPPER BOUND when param-free special experts are present: a token routed to a
    ±Identity expert activates one fewer GLU expert, so real active params are lower by roughly
    (special_load * top_k) experts' worth. Conservative in the right direction for param-matching."""
    cfg = getattr(model, "config", None)
    top_k = top_k or getattr(cfg, "num_experts_per_tok", None) or SHARED["num_experts_per_tok"]
    num_experts = num_experts or SHARED["num_experts"]
    total = trainable = inactive = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        if p.ndim == 3 and ("expert" in n or "gate_up_proj" in n or "down_proj" in n):
            e = p.shape[0]
            inactive += int(p.numel() * (1.0 - top_k / e))
    return total, trainable, total - inactive
