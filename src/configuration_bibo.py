from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

BIBO_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class BiBoConfig(PretrainedConfig):
    r"""
    Configuration class for the BiBo model.
    """
    model_type = "bibo"

    def __init__(
        self,
        vocab_size=128000,
        hidden_size=1536,
        intermediate_size=4104,  # 2nd Ramanujan-Hardy number (4104 = 16³+2³ = 15³+9³)
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        layer_norm_type="rms",
        use_cache=True,
        use_ssmax=True,  # SSMax: scaling softmax for long context
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        attention_bias=False,
        # MoE
        mlp_only_layers=None,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        num_routed_experts=16,
        num_shared_experts=1,
        num_experts_per_tok=6,
        num_experts=None,
        # Router
        router_type="mlp",  # "mlp" or "conv"
        kernel_size=3,
        router_lambda=1.0,  # Skywork-MoE logit normalization scaling
        router_noise=0.5,  # Exploration noise during training
        bias_update_factor=1e-2,  # Step size for load balancing bias updates
        bias_update_threshold=100_000,  # Tokens before bias update
        router_temperature=1.3,  # Legacy (not used; kept for compat)
        # Shared expert
        moe_shared_scaling=1.0,  # Auto-computed if 1.0 (DeepSeek-V2/V3 style)
        norm_topk_prob=False,
        output_router_logits=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_type = layer_norm_type
        self.use_cache = use_cache
        self.use_ssmax = use_ssmax
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts if num_experts is not None else (num_routed_experts + num_shared_experts)
        self.router_temperature = router_temperature
        self.bias_update_factor = bias_update_factor
        self.bias_update_threshold = bias_update_threshold
        self.router_noise = router_noise
        self.router_type = router_type
        self.kernel_size = kernel_size
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_lambda = router_lambda

        # --- Auto-estimate scaling factor for shared expert if left as 1.0 ---
        self.moe_shared_scaling = moe_shared_scaling
        if moe_shared_scaling == 1.0:
            try:
                import numpy as np
                def softmax(x):
                    p = np.exp(x)
                    return p / p.sum()
                n = self.num_routed_experts
                k = self.num_experts_per_tok
                s = getattr(self, 'num_shared_experts', 1) or 1
                factors = []
                for _ in range(10000):
                    logits = np.random.randn(n - s)
                    p = np.sort(softmax(logits))[::-1][:k - s]
                    factors.append(s**0.5 / (np.sum(p**2)**0.5))
                approx_lambda = float(np.mean(factors))
                self.moe_shared_scaling = round(approx_lambda, 2)
            except Exception as e:
                print(f"[BiBoConfig] Could not auto-set moe_shared_scaling: {e}")

        if self.rope_scaling is None:
            self.rope_scaling = {"type": "linear", "factor": 1.0}

        if self.layer_norm_type != "rms":
            raise ValueError(f"Only 'rms' layer_norm_type is supported. Got: {self.layer_norm_type}")

        if mlp_only_layers is None:
            self.mlp_only_layers = [0, num_hidden_layers - 1]
        else:
            self.mlp_only_layers = mlp_only_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # --- Validations ---
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.attention_dropout < 0.0 or self.attention_dropout > 1.0:
            raise ValueError("attention_dropout must be between 0.0 and 1.0")
        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")
        if self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.bias_update_factor < 0.0:
            raise ValueError("bias_update_factor must be non-negative")
        if self.router_noise < 0.0:
            raise ValueError("router_noise must be non-negative")
        if self.num_routed_experts < 5:
            raise ValueError("num_routed_experts must be >= 5 (need at least 1 MLP + identity + zero + noise + relu)")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed total number of experts")
        for idx in self.mlp_only_layers:
            if not (0 <= idx < self.num_hidden_layers):
                raise ValueError(f"mlp_only_layers index {idx} is out of range for {self.num_hidden_layers} layers")
