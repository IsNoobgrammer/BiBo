"""Multi-token prediction (MTP), depth 1 -- variant 1: one extra SWA decoder layer predicting t+2.

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


def mtp_layer_config(config):
    """(config copy, layer_idx) for an extra windowed layer with the L0-style all-active ensemble."""
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
    over[idx] = ens
    cfg.moe_overrides = over
    return cfg, idx


class MTP(nn.Module):
    def __init__(self, config, init_fn=None):
        super().__init__()
        from exp.modeling_bibo import BiBoDecoderLayer
        H, eps = config.hidden_size, config.rms_norm_eps
        self.norm_h = BiBoRMSNorm(H, eps=eps)
        self.norm_e = BiBoRMSNorm(H, eps=eps)
        self.proj = nn.Linear(2 * H, H, bias=False)
        cfg, idx = mtp_layer_config(config)
        self.layer = BiBoDecoderLayer(cfg, idx)
        self.norm_out = BiBoRMSNorm(H, eps=eps)
        if init_fn is not None:              # the model's own _init_weights, so this layer starts
            self.apply(init_fn)              # exactly like the main ones would

    def forward(self, lhs, e_next, position_embeddings, block_residual):
        x = self.proj(torch.cat([self.norm_h(lhs), self.norm_e(e_next).to(lhs.dtype)], dim=-1))
        out = self.layer(x, position_embeddings=position_embeddings, block_residual=block_residual)
        return self.norm_out(out[0])
