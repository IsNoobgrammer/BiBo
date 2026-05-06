"""Experimental config defaults and validation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

from transformers.utils import logging

logger = logging.get_logger(__name__)

EXPERIMENTAL_DEFAULTS: Dict[str, Any] = {
    # Attention experiments
    "use_ssmax": False,
    "attention_type": "softmax",
    "linear_attention_feature_map": "elu",
    "linear_attention_eps": 1e-6,
    "use_sliding_window": False,
    "sliding_window": 512,
    "max_window_layers": None,
    # Residual/mHC experiments
    "residual_gate_type": "none",
    "residual_gate_init": 0.95,
    "residual_mixer_type": "none",
    "residual_conv_kernel_size": 4,
    "residual_conv_init": 0.95,
    "residual_history_include_input": False,
    "residual_num_streams": 1,
    "residual_stream_mode": "independent",
    "residual_stream_gate_type": "token",
    "residual_stream_init": "copy",
    "residual_stream_read_init": 0.99,
    "residual_stream_write_init": 0.99,
}

EXPERIMENTAL_CONFIG_KEYS = frozenset(EXPERIMENTAL_DEFAULTS)


def pop_legacy_experimental_kwargs(kwargs: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Move old top-level experimental kwargs into the canonical exp dict."""
    legacy_exp = {}
    for key in list(kwargs):
        if key in EXPERIMENTAL_CONFIG_KEYS:
            legacy_exp[key] = kwargs.pop(key)
    return legacy_exp


def apply_experimental_config(
    config: Any,
    exp: Optional[Mapping[str, Any]],
    legacy_exp: Optional[Mapping[str, Any]],
    *,
    num_hidden_layers: int,
) -> None:
    """Attach experimental config values to a BiBoConfig instance.

    ``config.exp`` is the canonical serialized form. For the model code, values
    are also mirrored as attributes on the config object to keep call-sites tidy.
    """
    exp_values = dict(EXPERIMENTAL_DEFAULTS)
    if exp is not None:
        unknown = set(exp) - EXPERIMENTAL_CONFIG_KEYS
        if unknown:
            unknown_keys = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown experimental config option(s): {unknown_keys}")
        exp_values.update(exp)
    if legacy_exp:
        exp_values.update(legacy_exp)

    if exp_values["max_window_layers"] is None:
        exp_values["max_window_layers"] = num_hidden_layers

    for key, value in exp_values.items():
        setattr(config, key, value)
    config.exp = dict(exp_values)

    validate_experimental_config(config)


def validate_experimental_config(config: Any) -> None:
    if config.attention_type not in {"softmax", "sliding_window", "linear", "gdn", "kda"}:
        raise ValueError(
            "exp.attention_type must be one of: 'softmax', 'sliding_window', 'linear', 'gdn', 'kda'"
        )
    if config.linear_attention_feature_map not in {"elu", "relu"}:
        raise ValueError("exp.linear_attention_feature_map must be one of: 'elu', 'relu'")
    if config.linear_attention_eps <= 0.0:
        raise ValueError("exp.linear_attention_eps must be positive")
    if config.sliding_window is not None and config.sliding_window <= 0:
        raise ValueError("exp.sliding_window must be positive if specified")
    if config.max_window_layers is not None and config.max_window_layers < 0:
        raise ValueError("exp.max_window_layers must be non-negative if specified")
    if config.residual_gate_type not in {"none", "scalar", "token", "channel"}:
        raise ValueError("exp.residual_gate_type must be one of: 'none', 'scalar', 'token', 'channel'")
    if not (0.0 < config.residual_gate_init < 1.0):
        raise ValueError("exp.residual_gate_init must be between 0 and 1")
    if config.residual_mixer_type not in {"none", "causal_conv", "dynamic_causal_conv"}:
        raise ValueError(
            "exp.residual_mixer_type must be one of: 'none', 'causal_conv', 'dynamic_causal_conv'"
        )
    if config.residual_conv_kernel_size <= 1:
        raise ValueError("exp.residual_conv_kernel_size must be greater than 1")
    if not (0.0 < config.residual_conv_init < 1.0):
        raise ValueError("exp.residual_conv_init must be between 0 and 1")
    if not isinstance(config.residual_history_include_input, bool):
        raise ValueError("exp.residual_history_include_input must be a boolean")
    if config.residual_num_streams < 1:
        raise ValueError("exp.residual_num_streams must be at least 1")
    if config.residual_stream_mode not in {"independent", "delay_line"}:
        raise ValueError("exp.residual_stream_mode must be one of: 'independent', 'delay_line'")
    if config.residual_stream_gate_type not in {"scalar", "token"}:
        raise ValueError("exp.residual_stream_gate_type must be one of: 'scalar', 'token'")
    if config.residual_stream_init not in {"copy", "zero"}:
        raise ValueError("exp.residual_stream_init must be one of: 'copy', 'zero'")
    if not (0.0 < config.residual_stream_read_init < 1.0):
        raise ValueError("exp.residual_stream_read_init must be between 0 and 1")
    if not (0.0 < config.residual_stream_write_init < 1.0):
        raise ValueError("exp.residual_stream_write_init must be between 0 and 1")
    if (
        config.residual_num_streams > 1
        and config.residual_stream_mode == "independent"
        and config.residual_gate_type != "none"
    ):
        logger.warning_once(
            "Both residual branch gates and multi-stream write gates are enabled. "
            "This is valid, but their effects compose; use exp.residual_gate_type='none' "
            "for cleaner mHC-style stream-gate attribution."
        )
