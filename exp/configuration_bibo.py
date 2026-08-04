"""Experimental BiBo configuration with Kimi K3 Block AttnRes.

The stable configuration remains in ``src.configuration_bibo``. This module is
intentionally a one-way extension: ``exp`` may reuse ``src``, while ``src`` has
no dependency on the experimental package.
"""

from src.configuration_bibo import BiBoConfig as _StableBiBoConfig

__all__ = ["BiBoConfig"]


class BiBoConfig(_StableBiBoConfig):
    """BiBo configuration for the experimental Attention Residuals model.

    ``attn_res_block_size`` is measured in complete transformer decoder layers,
    matching the official Kimi K3 Hugging Face implementation. K3 uses 12.
    Passing ``None`` disables AttnRes and provides a standard-residual control.
    """

    model_type = "bibo_attn_res"

    def __init__(self, attn_res_block_size=12, attn_res_sites=2, attn_res_carry=False,
                 attn_res_fp32_stream=False, attn_res_carry_scale="none",
                 attn_res_emb_term=False, attn_res_emb_scale="none",
                 attn_res_emb_site="mlp", **kwargs):
        if attn_res_sites not in (1, 2):
            raise ValueError(
                f"attn_res_sites must be 2 (K3 faithful: a depth-mix before the attention "
                f"sublayer AND before the MLP sublayer) or 1 (one mix per layer at the layer "
                f"input; the MLP takes an ordinary PreNorm residual), got {attn_res_sites!r}")
        self.attn_res_sites = attn_res_sites
        self.attn_res_carry = bool(attn_res_carry)
        self.attn_res_fp32_stream = bool(attn_res_fp32_stream)
        _csm = "none" if attn_res_carry_scale in (False, None) else str(attn_res_carry_scale)
        if _csm not in ("none", "unbounded", "sigmoid", "tanh"):
            raise ValueError(f"attn_res_carry_scale must be one of none/unbounded/sigmoid/tanh, "
                             f"got {attn_res_carry_scale!r}")
        self.attn_res_carry_scale = _csm
        # A THIRD, OFF-SIMPLEX term on the carry write: h = attn_read + c*attn_out + d*embedding,
        # d learnable per layer, init 0 (so it is a strict generalization of plain carry).
        # The embedding is ALREADY a candidate inside attn_read, but that read is a convex
        # combination -- weighting the embedding costs prefix_sum weight one for one, so the model
        # cannot ask for MORE total token identity, only for a different split. d is additive and
        # unconstrained, which is the same reason c exists.
        # Motivation: layer 1 runs XSA BACKWARDS (tanh(alpha) to -0.95, i.e. it nearly doubles the
        # self-value component) in every s1 arm measured, while setting its carry c to ~0.03-0.09,
        # the smallest in the model. Reading: it wants token identity in the STREAM and has no
        # channel for it, so it abuses XSA. If that is right, d spikes at layer 1 and alpha there
        # returns positive. The depth profile of d IS the experiment -- do not hard-code it to L1.
        self.attn_res_emb_term = bool(attn_res_emb_term)
        self.attn_res_emb_scale = str(attn_res_emb_scale)
        self.attn_res_emb_site = str(attn_res_emb_site)
        if attn_res_block_size is not None and (
            isinstance(attn_res_block_size, bool)
            or not isinstance(attn_res_block_size, int)
            or attn_res_block_size <= 0
        ):
            raise ValueError(
                "attn_res_block_size must be a positive integer number of decoder layers "
                f"or None, got {attn_res_block_size!r}"
            )
        self.attn_res_block_size = attn_res_block_size
        super().__init__(**kwargs)
