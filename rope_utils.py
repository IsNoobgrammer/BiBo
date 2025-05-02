def rope_config_validation(config):
    """
    Validates the rope_theta and rope_scaling parameters for rotary embeddings.
    """
    if config.rope_theta is not None and config.rope_theta <= 0:
        raise ValueError("rope_theta must be positive if specified")
    if config.rope_scaling is not None:
        if not isinstance(config.rope_scaling, dict):
            raise ValueError("rope_scaling must be a dict or None")
        # Backward compatibility: 'type' -> 'rope_type'
        if "type" in config.rope_scaling:
            config.rope_scaling["rope_type"] = config.rope_scaling.pop("type")
        if 'rope_type' not in config.rope_scaling:
            raise ValueError("rope_scaling dict must have 'rope_type' key")
        if 'factor' in config.rope_scaling and config.rope_scaling['factor'] <= 1.0:
            raise ValueError("rope_scaling 'factor' must be > 1.0 if specified")
