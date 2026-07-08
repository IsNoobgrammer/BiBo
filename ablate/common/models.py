"""Model builder + param counter for the ablation. Swappable: build_arm(name) is the only entry."""
from . import _paths  # noqa: F401
import torch
from .configs import ARMS, SHARED, make_qwen_config, make_bibo_min_config
from . import patches


def build_arm(arm, device="cuda", dtype=torch.float32, attn_impl="sdpa",
              load_balance="bias", bias_update_threshold=10240, bias_update_factor=None, aux_coef=0.001,
              polyglu_mult=3, special_pairs=0):
    """arm in {'qwen','bibo_min'} -> (model, config). Params in `dtype` (fp32 master; bf16 via autocast).
    Balancing (fair, each native): BiBo bias updates (load_balance/bias_update_*); Qwen Switch aux loss (aux_coef).
    PARAM MATCH: BiBo GLU experts = polyglu_mult*3; Qwen num_experts is set to the SAME so they're matched, and
    BiBo's special_pairs Identity/Zero experts (param-free) are the extra we test."""
    eff = patches.resolve_attn(attn_impl)
    n_glu = polyglu_mult * 3
    if arm == "qwen":
        from baseline.qwen3moe.modeling import Qwen3MoeForCausalLM
        cfg = make_qwen_config(eff, aux_coef=aux_coef, num_experts=n_glu)
        model = Qwen3MoeForCausalLM(cfg)
    elif arm == "bibo_min":
        from src.modeling.models import BiBoForCausalLM
        cfg = make_bibo_min_config(load_balance, bias_update_threshold, bias_update_factor,
                                   polyglu_mult=polyglu_mult, special_pairs=special_pairs)
        model = BiBoForCausalLM(cfg)
        if eff.startswith("flash"):
            patches.patch_bibo_flash()
    else:
        raise ValueError(f"unknown arm {arm!r}; valid: {list(ARMS)}")
    return model.to(device=device, dtype=dtype), cfg


def count_params(model, top_k=None, num_experts=None):
    """Return (total, trainable, active). trainable excludes inert (requires_grad=False) params like
    BiBo's zero-init router bias; the ablation is matched on trainable/active params. active discounts
    inactive experts (3D stacked expert tensors)."""
    top_k = top_k or SHARED["num_experts_per_tok"]
    num_experts = num_experts or SHARED["num_experts"]
    total = trainable = inactive = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        # stacked expert weights are 3D (E, *, *) under a *.experts.* / gate_up_proj / down_proj name;
        # infer E from the tensor's leading dim so it's correct for any polyglu_mult
        if p.ndim == 3 and ("expert" in n or "gate_up_proj" in n or "down_proj" in n):
            e = p.shape[0]
            inactive += int(p.numel() * (1.0 - top_k / e))
    return total, trainable, total - inactive
