from transformers import PretrainedConfig

BIBO_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
NORM_TOPK_MODES = ("sum", "softmax")


class BiBoConfig(PretrainedConfig):

    model_type = "bibo"

    def __init__(
        self,
        vocab_size=128000,
        hidden_size=1536,
        intermediate_size=4104,
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        layer_norm_type="rms",
        exp_post_embed_norm=False,
        use_xsa=True,
        xsa_alpha_init=0.0,
        attention_dropout=0.0,
        attention_bias=False,
        hybrid_layer_pattern=None,
        sliding_window=128,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        rope_theta=None,
        rope_scaling=None,
        partial_rotary_factor=0.334,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        use_cache=True,
        mlp_only_layers=None,
        moe_intermediate_size=None,
        num_experts_per_tok=6,
        num_routed_experts=6,
        special_expert_pairs=1,
        pos_identity_expert=True,
        neg_identity_expert=True,
        use_shared_expert=False,
        shared_expert_type="mlp",
        num_shared_experts=1,
        kernel_size=3,
        norm_topk_prob="sum",
        bias_update_factor=None,
        bias_update_threshold=8000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        self.rms_norm_eps = rms_norm_eps
        self.layer_norm_type = layer_norm_type
        self.exp_post_embed_norm = exp_post_embed_norm

        self.use_xsa = use_xsa
        self.xsa_alpha_init = xsa_alpha_init   # per-head logit; strength = tanh(init), 0 = XSA off
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.sliding_window = sliding_window
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias
        self.add_full_attention_sink_bias = add_full_attention_sink_bias

        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        self.num_experts_per_tok = num_experts_per_tok
        self.num_routed_experts = num_routed_experts
        self.special_expert_pairs = special_expert_pairs
        self.pos_identity_expert = pos_identity_expert
        self.neg_identity_expert = neg_identity_expert
        self.num_pos_identity_experts = special_expert_pairs if pos_identity_expert else 0
        self.num_neg_identity_experts = special_expert_pairs if neg_identity_expert else 0
        self.num_glu_experts = (num_routed_experts - self.num_pos_identity_experts
                                - self.num_neg_identity_experts)

        self.use_shared_expert = use_shared_expert
        self.shared_expert_type = shared_expert_type
        self.num_shared_experts = num_shared_experts

        self.kernel_size = kernel_size
        self.norm_topk_prob = "sum" if norm_topk_prob is True else norm_topk_prob


        self.rope_theta = rope_theta if rope_theta is not None else 1e7

        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size is not None
            else self.intermediate_size // self.num_experts_per_tok
        )

        self.bias_update_factor = 0.4 if bias_update_factor is None else bias_update_factor

        self.bias_update_threshold = bias_update_threshold if bias_update_threshold is not None else 8000

        if self.rope_scaling is None:
            self.rope_scaling = {"type": "dynamic", "factor": 1.0}

        self.mlp_only_layers = (
            mlp_only_layers if mlp_only_layers is not None
            else sorted({0, num_hidden_layers - 1})
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.head_dim = self.hidden_size // self.num_attention_heads
        _rope_dim = round(self.partial_rotary_factor * self.head_dim)
        self.rope_dim = _rope_dim - (_rope_dim % 2)

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads "
                f"({self.num_attention_heads})"
            )
        if not 1 <= self.num_experts_per_tok <= self.num_routed_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) must be in "
                f"[1, num_routed_experts={self.num_routed_experts}]"
            )
        if self.num_glu_experts < 1:
            raise ValueError(
                f"num_routed_experts={self.num_routed_experts} minus "
                f"{self.num_pos_identity_experts + self.num_neg_identity_experts} +/-Identity specials "
                f"leaves {self.num_glu_experts} GLU experts; need at least 1. Raise "
                f"num_routed_experts or lower special_expert_pairs."
            )
        if self.rope_dim < 2:
            raise ValueError(
                f"partial_rotary_factor={self.partial_rotary_factor} gives rope_dim={self.rope_dim} "
                f"(head_dim={self.head_dim}); need at least 2 rotary dims."
            )
        if self.norm_topk_prob and self.norm_topk_prob not in NORM_TOPK_MODES:
            raise ValueError(
                f"norm_topk_prob must be one of {NORM_TOPK_MODES} (or False for raw scores), "
                f"got {self.norm_topk_prob!r}"
            )
        if self.hybrid_layer_pattern is not None:
            if len(self.hybrid_layer_pattern) != self.num_hidden_layers:
                raise ValueError(
                    f"hybrid_layer_pattern length ({len(self.hybrid_layer_pattern)}) must equal "
                    f"num_hidden_layers ({self.num_hidden_layers})"
                )
            if any(self.hybrid_layer_pattern) and not (
                isinstance(self.sliding_window, int) and self.sliding_window > 0):
                raise ValueError(
                    "hybrid_layer_pattern marks SWA layers but sliding_window is not a positive int "
                    "-- those layers would silently run full attention."
                )
        _has_swa = self.hybrid_layer_pattern is not None and any(self.hybrid_layer_pattern)
        if not _has_swa:
            self.sliding_window = None
        self.layer_types = (
            ["sliding_attention" if v else "full_attention" for v in self.hybrid_layer_pattern]
            if _has_swa else ["full_attention"] * self.num_hidden_layers
        )
