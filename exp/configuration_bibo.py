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

    ``use_typed_attn_res`` enables BiBo's experimental extension: attention and
    MLP outputs remain separate thought/memory streams inside a block, while
    each sublayer learns a token-conditioned read over typed and canonical K3
    candidates. ``typed_attn_res_long_memory`` additionally archives the
    memory-only stream at block boundaries. The extra streams start with a
    relative prior of ``typed_attn_res_extra_init`` so opt-in initialization is
    close to ordinary K3 AttnRes rather than an equal five-way mixture.

    ``use_typed_attn_res_fast_slow_memory`` replaces the current memory sum with
    a fast decayed state that resets per block, and adds a slow decayed state
    that persists across all layers. Their initial decay rates are configured by
    ``typed_attn_res_fast_decay_init`` and ``typed_attn_res_slow_decay_init``;
    the learned parameterization preserves ``slow > fast``. Attention controls
    the slow-memory write gain. ``use_typed_attn_res_innovation_write`` removes
    a learned fraction of each MLP output parallel to the current thought stream
    before writing typed memory; ``typed_attn_res_innovation_init`` initializes
    that fraction. Neither option modifies the canonical residual write.
    """

    model_type = "bibo_attn_res"

    def __init__(
        self,
        attn_res_block_size=12,
        attn_res_sites=2,
        attn_res_carry=False,
        use_typed_attn_res=False,
        typed_attn_res_long_memory=True,
        typed_attn_res_extra_init=0.01,
        use_typed_attn_res_fast_slow_memory=False,
        typed_attn_res_fast_decay_init=0.5,
        typed_attn_res_slow_decay_init=0.95,
        use_typed_attn_res_innovation_write=False,
        typed_attn_res_innovation_init=0.01,
        **kwargs,
    ):
        if attn_res_sites not in (1, 2):
            raise ValueError(
                f"attn_res_sites must be 2 (K3 faithful: a depth-mix before the attention "
                f"sublayer AND before the MLP sublayer) or 1 (one mix per layer at the layer "
                f"input; the MLP takes an ordinary PreNorm residual), got {attn_res_sites!r}")
        self.attn_res_sites = attn_res_sites
        self.attn_res_carry = bool(attn_res_carry)
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

        if not isinstance(use_typed_attn_res, bool):
            raise ValueError(
                f"use_typed_attn_res must be a bool, got {use_typed_attn_res!r}"
            )
        if not isinstance(typed_attn_res_long_memory, bool):
            raise ValueError(
                "typed_attn_res_long_memory must be a bool, got "
                f"{typed_attn_res_long_memory!r}"
            )
        if (
            isinstance(typed_attn_res_extra_init, bool)
            or not isinstance(typed_attn_res_extra_init, (int, float))
            or not 0.0 < typed_attn_res_extra_init <= 1.0
        ):
            raise ValueError(
                "typed_attn_res_extra_init must be a number in (0, 1], got "
                f"{typed_attn_res_extra_init!r}"
            )
        if use_typed_attn_res and attn_res_block_size is None:
            raise ValueError(
                "use_typed_attn_res=True requires attn_res_block_size to be enabled"
            )
        if use_typed_attn_res and attn_res_sites != 2:
            raise ValueError(
                "use_typed_attn_res=True requires attn_res_sites=2 so attention and "
                "MLP have independent typed reads"
            )
        self.use_typed_attn_res = use_typed_attn_res
        self.typed_attn_res_long_memory = typed_attn_res_long_memory
        self.typed_attn_res_extra_init = float(typed_attn_res_extra_init)

        for name, value in {
            "use_typed_attn_res_fast_slow_memory": use_typed_attn_res_fast_slow_memory,
            "use_typed_attn_res_innovation_write": use_typed_attn_res_innovation_write,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool, got {value!r}")
        for name, value in {
            "typed_attn_res_fast_decay_init": typed_attn_res_fast_decay_init,
            "typed_attn_res_slow_decay_init": typed_attn_res_slow_decay_init,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 < value < 1.0
            ):
                raise ValueError(f"{name} must be a number in (0, 1), got {value!r}")
        if typed_attn_res_slow_decay_init <= typed_attn_res_fast_decay_init:
            raise ValueError(
                "typed_attn_res_slow_decay_init must be > "
                "typed_attn_res_fast_decay_init"
            )
        if (
            isinstance(typed_attn_res_innovation_init, bool)
            or not isinstance(typed_attn_res_innovation_init, (int, float))
            or not 0.0 < typed_attn_res_innovation_init < 1.0
        ):
            raise ValueError(
                "typed_attn_res_innovation_init must be a number in (0, 1), got "
                f"{typed_attn_res_innovation_init!r}"
            )
        if use_typed_attn_res_fast_slow_memory and not use_typed_attn_res:
            raise ValueError(
                "use_typed_attn_res_fast_slow_memory=True requires "
                "use_typed_attn_res=True"
            )
        if use_typed_attn_res_innovation_write and not use_typed_attn_res:
            raise ValueError(
                "use_typed_attn_res_innovation_write=True requires "
                "use_typed_attn_res=True"
            )
        self.use_typed_attn_res_fast_slow_memory = (
            use_typed_attn_res_fast_slow_memory
        )
        self.typed_attn_res_fast_decay_init = float(
            typed_attn_res_fast_decay_init
        )
        self.typed_attn_res_slow_decay_init = float(
            typed_attn_res_slow_decay_init
        )
        self.use_typed_attn_res_innovation_write = (
            use_typed_attn_res_innovation_write
        )
        self.typed_attn_res_innovation_init = float(
            typed_attn_res_innovation_init
        )
        super().__init__(**kwargs)
