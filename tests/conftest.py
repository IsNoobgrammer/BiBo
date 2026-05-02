import sys
import types

import torch


def _install_transformers_stub():
    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    transformers.PretrainedConfig = PretrainedConfig

    activations = types.ModuleType("transformers.activations")
    activations.ACT2FN = {
        "silu": torch.nn.functional.silu,
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
    }

    modeling_outputs = types.ModuleType("transformers.modeling_outputs")
    for name in (
        "BaseModelOutputWithPast",
        "CausalLMOutputWithPast",
        "MoeModelOutputWithPast",
        "MoeCausalLMOutputWithPast",
    ):
        setattr(modeling_outputs, name, type(name, (), {}))

    modeling_utils = types.ModuleType("transformers.modeling_utils")

    class PreTrainedModel(torch.nn.Module):
        config_class = None

        def __init__(self, config=None):
            super().__init__()
            self.config = config

    modeling_utils.PreTrainedModel = PreTrainedModel

    utils = types.ModuleType("transformers.utils")

    class Logging:
        @staticmethod
        def get_logger(name):
            class Logger:
                def warning_once(self, *args, **kwargs):
                    pass

                def warning(self, *args, **kwargs):
                    pass

                def info(self, *args, **kwargs):
                    pass

            return Logger()

    utils.logging = Logging

    rope_utils = types.ModuleType("transformers.modeling_rope_utils")
    rope_utils.rope_config_validation = lambda config: config

    cache_utils = types.ModuleType("transformers.cache_utils")
    for name in ("Cache", "DynamicCache", "SlidingWindowCache", "StaticCache"):
        setattr(cache_utils, name, type(name, (), {}))

    sys.modules.update(
        {
            "transformers": transformers,
            "transformers.activations": activations,
            "transformers.modeling_outputs": modeling_outputs,
            "transformers.modeling_utils": modeling_utils,
            "transformers.utils": utils,
            "transformers.modeling_rope_utils": rope_utils,
            "transformers.cache_utils": cache_utils,
        }
    )


try:
    import transformers  # noqa: F401
except Exception:
    _install_transformers_stub()


sys.path.insert(0, "src")
