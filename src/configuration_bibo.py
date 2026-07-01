from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

BIBO_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class BiBoConfig(PretrainedConfig):
    r"""
    Configuration class for the BiBo model.

    Fields are grouped: core dims → norm → attention → RoPE → MoE layout →
    router → shared expert. A few values are auto-derived in __init__ when left
    as None (rope_theta, moe_intermediate_size, bias_update_factor).
    """

    model_type = "bibo"

    def __init__(
        self,
        # ── Core dimensions ──────────────────────────────────────
        vocab_size=128000,
        hidden_size=1536,
        intermediate_size=4104,  # 2nd Ramanujan-Hardy number (4104 = 16³+2³ = 15³+9³)
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        hidden_act="silu",
        initializer_range=0.02,
        # ── Norm ─────────────────────────────────────────────────
        rms_norm_eps=1e-6,
        layer_norm_type="rms",
        exp_post_embed_norm=False,  # EXPERIMENTAL: extra RMSNorm after embeddings, before block 0 (BLOOM-style). Final pre-LM-head norm always on regardless.
        # ── Attention ────────────────────────────────────────────
        use_xsa=True,   # Exclusive Self Attention (https://arxiv.org/abs/2603.09078)
        use_ssmax=True,  # SSMax: scaling softmax for long context
        attention_dropout=0.0,
        attention_bias=False,
        # ── RoPE ─────────────────────────────────────────────────
        rope_theta=None,    # Auto-derived (10000.0) if None
        rope_scaling=None,  # Auto: {"type": "dynamic", "factor": 1.0} if None
        rope_nope_ratio=0.5,  # Fraction of heads that are NoPE. 0.5 = 2:2 (6 RoPE+NTK heads, 6 NoPE content heads at 12h/2kv). 0.0 = all RoPE.
        # ── Tokens / embeddings ──────────────────────────────────
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        use_cache=True,
        # ── MoE layout ───────────────────────────────────────────
        mlp_only_layers=None,  # Auto: [0, num_hidden_layers - 1] (first + last dense)
        moe_intermediate_size=None,  # Auto: intermediate_size // num_experts_per_tok
        num_experts_per_tok=6,
        num_experts=None,  # Auto: num_routed_experts + num_shared_experts
        polyglu_expert_multiplier=2,  # Groups of 3 (SiLU, ReLU², Tanh) GLU experts
        special_expert_pairs=1,       # Pairs of (Identity, Zero) experts
        # ── Shared expert ────────────────────────────────────────
        use_shared_expert=False,    # Off by default (param-match Qwen3MoE — no shared expert)
        num_shared_experts=1,
        shared_expert_type="mlp",   # "mlp" (SwiGLU, like Qwen) or "conv" (CausalConv1D)
        # ── Router ───────────────────────────────────────────────
        router_type="mlp",   # "mlp" or "conv"
        kernel_size=3,        # conv-router / conv-expert kernel width
        gate_type="sigmoid",  # "sigmoid" (DeepSeek-V3, independent) or "softmax" (legacy, competitive)
        router_activation="none",  # applied to raw logits before gating: "none", "relu" (DECO), "silu"
        norm_topk_prob=True,        # MiMo-V2.5 / DeepSeek-V3: renormalize the top-k weights to sum to 1
        routed_scaling_factor=1.0,  # MiMo/DeepSeek-V3 final routed-weight scale; 1.0 = no-op (MiMo-V2.5)
        router_noise=0,             # DEPRECATED (injection commented out in router.py); kept for re-enable
        load_balance_strategy="bias",  # "none" or "bias" (heuristic aux-loss-free bias updates)
        bias_update_factor=None,    # Auto: Hill function of num_routed_experts
        bias_update_threshold=8000,  # tokens between bias updates
        output_router_logits=False,
        **kwargs,
    ):
        # ── Core dimensions ──────────────────────────────────────
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        # ── Norm ─────────────────────────────────────────────────
        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_type = layer_norm_type
        self.exp_post_embed_norm = exp_post_embed_norm

        # ── Attention ────────────────────────────────────────────
        self.use_xsa = use_xsa
        self.use_ssmax = use_ssmax
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        # ── RoPE ─────────────────────────────────────────────────
        self.rope_scaling = rope_scaling
        self.rope_nope_ratio = rope_nope_ratio

        # ── Tokens / embeddings ──────────────────────────────────
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        # ── MoE layout ───────────────────────────────────────────
        self.num_experts_per_tok = num_experts_per_tok
        self.polyglu_expert_multiplier = polyglu_expert_multiplier
        self.special_expert_pairs = special_expert_pairs
        # experts = polyglu_multiplier * 3 (SiLU, ReLU², Tanh) + special_pairs * 2 (Identity, Zero)
        self.num_routed_experts = (polyglu_expert_multiplier * 3) + (special_expert_pairs * 2)
        self.num_experts = (
            num_experts if num_experts is not None
            else (self.num_routed_experts + num_shared_experts)
        )

        # ── Shared expert ────────────────────────────────────────
        self.use_shared_expert = use_shared_expert
        self.num_shared_experts = num_shared_experts
        self.shared_expert_type = shared_expert_type

        # ── Router ───────────────────────────────────────────────
        self.router_type = router_type
        self.kernel_size = kernel_size
        self.gate_type = gate_type
        self.router_activation = router_activation
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.router_noise = router_noise
        self.load_balance_strategy = load_balance_strategy
        self.output_router_logits = output_router_logits

        # ============================================================
        # Auto-derived hyperparameters (when left as None)
        # ============================================================

        # rope_theta: default 10000 (matches Qwen3MoE / standard LLaMA)
        self.rope_theta = rope_theta if rope_theta is not None else 10000.0

        # moe_intermediate_size: compute parity with dense FFN.
        #   dense active/token = 2*hidden*intermediate ; MoE = 2*hidden*moe_intermediate*top_k
        #   parity → moe_intermediate = intermediate // top_k
        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size is not None
            else self.intermediate_size // self.num_experts_per_tok
        )

        # bias_update_factor: scales with num_experts (small n → small step; large n → strong
        # balancing for EP). Bounded [0, 0.35] via a Hill function A*n^α/(n^α+C), fit to
        # f(8)=0.07, f(16)=0.1417, f(∞)=0.35.
        if bias_update_factor is not None:
            self.bias_update_factor = bias_update_factor
        else:
            n_pow = self.num_routed_experts ** 1.445
            self.bias_update_factor = round(0.35 * n_pow / (n_pow + 81.0), 4)

        # bias_update_threshold: tokens to accumulate before applying a bias update
        self.bias_update_threshold = bias_update_threshold if bias_update_threshold is not None else 8000

        # rope_scaling: dynamic NTK-aware by default — identity within the trained window,
        # smooth base growth beyond it. type="none" for plain RoPE; factor=1.0 = pure dynamic.
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "dynamic", "factor": 1.0}

        # mlp_only_layers: first + last layers are dense MLP, rest are MoE
        self.mlp_only_layers = (
            mlp_only_layers if mlp_only_layers is not None
            else [0, num_hidden_layers - 1]
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # ============================================================
        # Validations
        # ============================================================
        if self.layer_norm_type != "rms":
            raise ValueError(f"Only 'rms' layer_norm_type is supported. Got: {self.layer_norm_type}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
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
            raise ValueError(
                "polyglu_expert_multiplier must be >= 1 (need at least one group of SiLU/ReLU²/Tanh GLU experts)"
            )
        if self.special_expert_pairs < 0:
            raise ValueError("special_expert_pairs must be >= 0")
        if self.num_routed_experts < 3:
            raise ValueError(
                f"num_routed_experts must be >= 3 (got {self.num_routed_experts}). Increase polyglu_expert_multiplier or special_expert_pairs."
            )
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed total number of experts")
        if self.load_balance_strategy not in ("none", "bias"):
            raise ValueError(
                f"load_balance_strategy must be 'none' or 'bias', got '{self.load_balance_strategy}'"
            )
        if self.router_activation not in ("none", "relu", "silu"):
            raise ValueError(
                f"router_activation must be 'none', 'relu', or 'silu', got '{self.router_activation}'"
            )
        if self.rope_scaling.get("type") not in ("none", "dynamic"):
            raise ValueError(
                f"rope_scaling['type'] must be 'none' or 'dynamic', got {self.rope_scaling.get('type')!r}"
            )
        if self.rope_scaling.get("factor", 1.0) <= 0:
            raise ValueError("rope_scaling['factor'] must be positive")
        if not (0.0 <= self.rope_nope_ratio < 1.0):
            raise ValueError(f"rope_nope_ratio must be in [0, 1), got {self.rope_nope_ratio}")
        _groups = self.num_attention_heads // self.num_key_value_heads
        _n_rope = self.num_attention_heads - round(self.num_attention_heads * self.rope_nope_ratio)
        if _n_rope % _groups != 0:
            raise ValueError(
                f"rope_nope_ratio={self.rope_nope_ratio} gives {_n_rope} RoPE heads, which is not divisible by "
                f"num_key_value_groups={_groups}. The RoPE/NoPE boundary must align with KV groups. "
                f"Try rope_nope_ratio=0.5 (half heads)."
            )
        for idx in self.mlp_only_layers:
            if not (0 <= idx < self.num_hidden_layers):
                raise ValueError(
                    f"mlp_only_layers index {idx} is out of range for {self.num_hidden_layers} layers"
                )
