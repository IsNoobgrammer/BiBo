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
        rms_norm_eps=1e-6,
        layer_norm_type="rms",
        use_cache=True,
        use_ssmax=True,  # SSMax: scaling softmax for long context
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        rope_theta=None,  # Auto-derived from max_position_embeddings if None
        rope_scaling=None,
        attention_dropout=0.0,
        attention_bias=False,
        # MoE
        mlp_only_layers=None,
        decoder_sparse_step=1,
        moe_intermediate_size=None,  # Auto: intermediate_size // top_k
        use_shared_expert=False,  # Disable shared expert (match Qwen3MoE — no shared expert)
        num_shared_experts=1,
        num_experts_per_tok=6,
        num_experts=None,
        # PolyGLU expert layout
        polyglu_expert_multiplier=2,  # Groups of 3 (SiLU, ReLU², Tanh) GLU experts
        special_expert_pairs=1,  # Pairs of (Identity, Zero) experts
        # Router
        router_type="mlp",  # "mlp" or "conv"
        kernel_size=3,
        router_lambda=1.0,  # Skywork-MoE logit normalization scaling
        router_noise=0,  # Auto: log(num_experts) * 0.1
        bias_update_factor=None,  # Auto: (1 - exp(-n/48)) * 0.5
        bias_update_threshold=8000,  # User knob: tokens between bias updates
        router_temperature=1.3,  # Legacy (not used; kept for compat)
        # Router logit normalization (Skywork-MoE style)
        use_router_logit_norm=False,  # z-score normalize logits before softmax
        # Load balancing strategy: "none", "bias" (heuristic bias updates), "aux_loss" (Switch Transformer / Qwen style)
        load_balance_strategy="bias",
        aux_loss_coef=0.01,  # aux load-balancing loss coef (when strategy="aux_loss"); 1e-2 = ablated consensus (Switch-T, ST-MoE, OLMoE)
        # Router activation: applied to raw logits before softmax/selection
        # "none" (standard softmax), "relu" (DECO-style), "silu"
        router_activation="none",
        # Shared expert
        shared_expert_type="mlp",  # "mlp" (SwiGLU, like Qwen) or "conv" (CausalConv1D)
        moe_shared_scaling=1.0,  # Auto-computed if 1.0 (DeepSeek-V2/V3 style)
        norm_topk_prob=False,
        gate_type="sigmoid",  # "sigmoid" (DeepSeek-V3, independent) or "softmax" (legacy, competitive)
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
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.decoder_sparse_step = decoder_sparse_step
        self.use_shared_expert = use_shared_expert
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        # PolyGLU layout: experts = polyglu_multiplier * 3 (SiLU, ReLU², Tanh) + special_pairs * 2 (Identity, Zero)
        self.polyglu_expert_multiplier = polyglu_expert_multiplier
        self.special_expert_pairs = special_expert_pairs
        self.num_routed_experts = (polyglu_expert_multiplier * 3) + (special_expert_pairs * 2)
        self.num_experts = num_experts if num_experts is not None else (self.num_routed_experts + num_shared_experts)
        self.router_temperature = router_temperature
        self.router_type = router_type
        self.kernel_size = kernel_size
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.gate_type = gate_type
        self.router_lambda = router_lambda
        self.router_noise = router_noise
        self.shared_expert_type = shared_expert_type
        self.use_router_logit_norm = use_router_logit_norm
        self.load_balance_strategy = load_balance_strategy
        self.aux_loss_coef = aux_loss_coef
        self.router_activation = router_activation

        # ============================================================
        # Auto-derived hyperparameters
        # ============================================================

        # rope_theta: default 10000 (matches Qwen3MoE / standard LLaMA)
        if rope_theta is not None:
            self.rope_theta = rope_theta
        else:
            self.rope_theta = 10000.0

        # moe_intermediate_size: maintain compute parity with dense FFN
        # dense active params/token = 2 * hidden * intermediate
        # MoE active params/token = 2 * hidden * moe_intermediate * top_k
        # parity → moe_intermediate = intermediate // top_k
        if moe_intermediate_size is not None:
            self.moe_intermediate_size = moe_intermediate_size
        else:
            self.moe_intermediate_size = self.intermediate_size // self.num_experts_per_tok

        # # router_noise: exploration noise, scales with log(num_experts)
        # # more experts → more need for exploration to discover all of them
        # if router_noise is not None:
        #     self.router_noise = router_noise
        # else:
        #     import math as _math
        #     self.router_noise = round(min(1.0, 0.1 * _math.log(self.num_routed_experts)), 4)

        # bias_update_factor: scales with num_experts
        # Small n → small step (few experts, imbalance less harmful on single GPU)
        # Large n → large step (many experts, need strong balancing for EP)
        # Bounded [0, 0.5], smooth S-curve via exponential saturation
        if bias_update_factor is not None:
            self.bias_update_factor = bias_update_factor
        else:
            import math as _math
            n = self.num_routed_experts
            # Hill function: A * n^α / (n^α + C)
            # Derived from: f(8)=0.07, f(16)=0.1417, f(∞)=0.35
            _alpha = 1.445
            _C     = 81.0
            _n_pow = n ** _alpha
            self.bias_update_factor = round(0.35 * _n_pow / (_n_pow + _C), 4)

        # bias_update_threshold: user-controlled frequency knob
        # How many tokens to accumulate before applying a bias update
        if bias_update_threshold is not None:
            self.bias_update_threshold = bias_update_threshold
        else:
            self.bias_update_threshold = 8000

        # --- Auto-estimate scaling factor for shared expert if left as 1.0 ---
        self.moe_shared_scaling = moe_shared_scaling
        if moe_shared_scaling == 1.0:
            try:
                import numpy as np
                def softmax(x):
                    e = np.exp(x - x.max())
                    return e / e.sum()
                n = self.num_routed_experts
                k = self.num_experts_per_tok
                s = getattr(self, 'num_shared_experts', 1) or 1
                lam = self.router_lambda  # Account for Skywork-MoE logit normalization
                factors = []
                for _ in range(10000):
                    logits = np.random.randn(n - s)
                    # Simulate actual router: normalize then scale by router_lambda
                    logits_norm = (logits - logits.mean()) / (logits.std() + 1e-6)
                    logits_scaled = lam * logits_norm
                    p = np.sort(softmax(logits_scaled))[::-1][:k - s]
                    factors.append(s**0.5 / (np.sum(p**2)**0.5))
                approx_lambda = float(np.mean(factors))
                self.moe_shared_scaling = round(approx_lambda, 2)
            except Exception as e:
                print(f"[BiBoConfig] Could not auto-set moe_shared_scaling: {e}")

        # Dynamic NTK-aware RoPE on by default: identity within the trained window
        # (seq_len <= max_position_embeddings), smooth base growth beyond it. Set
        # type="none" for plain RoPE. factor scales extension aggressiveness (1.0 = pure dynamic).
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "dynamic", "factor": 1.0}

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
        if self.shared_expert_type not in ("mlp", "conv"):
            raise ValueError(f"shared_expert_type must be 'mlp' or 'conv', got '{self.shared_expert_type}'")
        if self.polyglu_expert_multiplier < 1:
            raise ValueError("polyglu_expert_multiplier must be >= 1 (need at least one group of SiLU/ReLU²/Tanh GLU experts)")
        if self.special_expert_pairs < 0:
            raise ValueError("special_expert_pairs must be >= 0")
        if self.num_routed_experts < 3:
            raise ValueError(f"num_routed_experts must be >= 3 (got {self.num_routed_experts}). Increase polyglu_expert_multiplier or special_expert_pairs.")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed total number of experts")
        if self.load_balance_strategy not in ("none", "bias", "aux_loss"):
            raise ValueError(f"load_balance_strategy must be 'none', 'bias', or 'aux_loss', got '{self.load_balance_strategy}'")
        if self.router_activation not in ("none", "relu", "silu"):
            raise ValueError(f"router_activation must be 'none', 'relu', or 'silu', got '{self.router_activation}'")
        if self.rope_scaling.get("type") not in ("none", "dynamic"):
            raise ValueError(f"rope_scaling['type'] must be 'none' or 'dynamic', got {self.rope_scaling.get('type')!r}")
        if self.rope_scaling.get("factor", 1.0) <= 0:
            raise ValueError("rope_scaling['factor'] must be positive")
        for idx in self.mlp_only_layers:
            if not (0 <= idx < self.num_hidden_layers):
                raise ValueError(f"mlp_only_layers index {idx} is out of range for {self.num_hidden_layers} layers")
