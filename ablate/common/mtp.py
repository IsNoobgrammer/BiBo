"""Multi-token prediction (MTP), depth 1: one extra layer predicting t+2. Variants (--mtp_variant):

    1  x -> AR read -> SWA attention -> fused carry -> MLP     (a full 11th SWA layer, below)
    2  x -> SWA attention -> MLP                               plain pre-norm block, no AttnRes
    3  x -> AR read -> MLP                                     no attention: s = x + Ens(norm AR(x, A))
    4  x -> MLP                                                s = x + Ens(norm x)

All share the input W_p[norm(lhs); norm(Emb(t+1))], the 8 x 576 all-active ensemble, the output
norm and the tied lm_head. Unused sub-modules are deleted so params/FLOPs are honest.

Variant 1 in detail:

    x_i   = W_p [ RMSNorm(lhs_i) ; RMSNorm(Emb(t_{i+1})) ]        lhs = final hidden state (post-norm)
    x     = BiBoDecoderLayer_L(x, archive)                          L = num_hidden_layers (an 11th layer)
    logit = lm_head(RMSNorm(x))            -> CE against t_{i+2}   (Emb / lm_head SHARED, tied)

The layer is an ordinary windowed (SWA) layer of this model -- attention residuals (reads the MAIN
model's block archive, "ar"), SWA attention, carry ("ar + swa"), MLP -- except its MLP is the
all-active 8 x 576 ensemble (L0's geometry) instead of the routed MoE. Emb(t_{i+1}) is the real next
token: known input at train time, never the t_{i+2} target, so the head stays causal.

Runs at full length S: position i uses t_{i+1} = ids[:, i+1] (exists for every i); the last position
has no t_{i+2} and is ignored in the loss. Val loss / BPB never see this head.
"""
import copy

import torch
import torch.nn as nn

from src.modeling.norm import BiBoRMSNorm


def mtp_layer_config(config, ffn="ensemble"):
    """(config copy, layer_idx) for an extra windowed layer. ffn: 'ensemble' = L0-style 8 x 576
    all-active; 'dense' = ONE all-active expert of width 8*576 = 4608 -- a dense GLU FFN with the same
    params AND the same radial-normsilu activation/kernel (BiBoMLP would also switch the act to SiLU)."""
    idx = config.num_hidden_layers
    cfg = copy.copy(config)
    pat = getattr(config, "hybrid_layer_pattern", None)
    if pat is not None:
        cfg.hybrid_layer_pattern = list(pat) + [1]
    per = getattr(config, "sliding_window_per_layer", None)
    if per is not None:
        cfg.sliding_window_per_layer = list(per) + [next(w for w, p in zip(per, pat) if p)]
    over = dict(getattr(config, "moe_overrides", None) or {})
    ens = dict(over.get(0) or {"num_routed_experts": 8, "num_experts_per_tok": 8, "moe_intermediate_size": 576})
    assert ens["num_experts_per_tok"] == ens["num_routed_experts"], "MTP MLP must be all-active"
    if ffn == "dense":
        ens = {"num_routed_experts": 1, "num_experts_per_tok": 1,
               "moe_intermediate_size": ens["num_routed_experts"] * ens["moe_intermediate_size"]}
    over[idx] = ens
    cfg.moe_overrides = over
    return cfg, idx


class MTP(nn.Module):
    HAS_ATTN = {1: True, 2: True, 3: False, 4: False}

    def __init__(self, config, variant=1, init_fn=None, ffn="ensemble", use_emb=True, use_proj=True):
        super().__init__()
        from exp.modeling_bibo import BiBoDecoderLayer
        assert variant in self.HAS_ATTN, f"--mtp_variant {variant}: valid 1-4"
        self.variant = variant
        H, eps = config.hidden_size, config.rms_norm_eps
        self.use_emb = bool(use_emb)       # False: x = W_p norm(lhs) -- t(i+1) is NOT given to the head
        self.norm_h = BiBoRMSNorm(H, eps=eps)
        self.norm_e = BiBoRMSNorm(H, eps=eps) if self.use_emb else None
        # use_proj=False (only without the embedding): no W_p at all -- the head's residual stream
        # starts from norm(lhs), exactly what the main head sees, and the layer learns the t+2 correction
        assert use_proj or not self.use_emb, "W_p is required to merge [lhs; Emb(t+1)] (2H -> H)"
        self.proj = nn.Linear((2 if self.use_emb else 1) * H, H, bias=False) if use_proj else None
        cfg, idx = mtp_layer_config(config, ffn)
        if variant == 2:
            cfg.attn_res_block_size = None       # the layer takes its standard pre-norm residual path
        self.layer = BiBoDecoderLayer(cfg, idx)
        if variant in (3, 4):                    # no attention sublayer
            del self.layer.self_attn, self.layer.input_layernorm
            self.layer.attn_res_carry_theta = None
        if variant == 4:                         # no depth read either
            del self.layer.self_attention_res_proj, self.layer.self_attention_res_norm
        self.norm_out = BiBoRMSNorm(H, eps=eps)
        if init_fn is not None:              # the model's own _init_weights, so this layer starts
            self.apply(init_fn)              # exactly like the main ones would

    def forward(self, lhs, e_next, position_embeddings, block_residual):
        if self.use_emb:
            x = self.proj(torch.cat([self.norm_h(lhs), self.norm_e(e_next).to(lhs.dtype)], dim=-1))
        else:
            x = self.norm_h(lhs)
            if self.proj is not None:
                x = self.proj(x)
        L = self.layer
        if self.variant == 1:
            s = L(x, position_embeddings=position_embeddings, block_residual=block_residual)[0]
        elif self.variant == 2:
            s = L(x, position_embeddings=position_embeddings)[0]
        else:
            if self.variant == 3:
                from exp.modeling_bibo import apply_attention_residual
                B, S, H = x.shape
                r = apply_attention_residual(x.reshape(-1, H), block_residual,
                                             L.self_attention_res_proj, L.self_attention_res_norm,
                                             L.attn_res_score_mode, L.attn_res_topk).reshape(B, S, H)
            else:
                r = x
            s = x + L._attn_res_mlp_forward(r)   # post-attention norm + ensemble (megakernel-patched)
        return self.norm_out(s)
