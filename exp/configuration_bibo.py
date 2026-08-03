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

    def __init__(self, attn_res_block_size=12, attn_res_sites=2, **kwargs):
        if attn_res_sites not in (1, 2):
            raise ValueError(
                f"attn_res_sites must be 2 (K3 faithful: a depth-mix before the attention "
                f"sublayer AND before the MLP sublayer) or 1 (one mix per layer at the layer "
                f"input; the MLP takes an ordinary PreNorm residual), got {attn_res_sites!r}")
        self.attn_res_sites = attn_res_sites
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
