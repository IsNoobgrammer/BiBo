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
        swa_qk_norm=True,
        rope_theta=None,                     # base for the WINDOWED layers; global layers are NoPE
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
        # ASYMMETRIC override. `special_expert_pairs` is a per-type count applied to BOTH signs,
        # so it can only express n(+Identity) == n(-Identity). These let the two differ -- e.g.
        # 6 positive and 2 negative, to test whether signed pass-through needs as much subtractive
        # capacity as additive. Both the eager MoE and the Triton act-code path already build
        # their expert ranges from the two counts independently, so nothing downstream changes.
        # None = derive from special_expert_pairs as before.
        num_pos_identity_experts=None,
        num_neg_identity_experts=None,
        use_shared_expert=False,
        shared_expert_type="mlp",
        num_shared_experts=1,
        kernel_size=3,
        norm_topk_prob="sum",
        bias_update_factor=None,
        bias_update_threshold=8000,
        bf16_residual_stream=False,
        bf16_moe_out=False,
        **kwargs,
    ):
        # Cast the residual stream to BF16 at the embedding, so every `residual + sublayer`
        # add downstream stays bf16 instead of promoting back to fp32. Master weights and
        # optimizer state are UNCHANGED (still fp32) -- this is the stream only, which is
        # what modded-nanogpt runs and what halves residual traffic.
        self.bf16_residual_stream = bool(bf16_residual_stream)
        self.bf16_moe_out = bool(bf16_moe_out)
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
        self.sliding_window_per_layer = None   # set below iff a per-layer list is given
        # QK-norm on WINDOWED layers only. Global layers always keep it. MiMo-V2.5-Pro ships no
        # QK-norm anywhere, Gemma 4 applies it on every layer; this flag tests the middle.
        self.swa_qk_norm = swa_qk_norm


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
        self.num_pos_identity_experts = (
            int(num_pos_identity_experts) if num_pos_identity_experts is not None
            else (special_expert_pairs if pos_identity_expert else 0))
        self.num_neg_identity_experts = (
            int(num_neg_identity_experts) if num_neg_identity_experts is not None
            else (special_expert_pairs if neg_identity_expert else 0))
        if self.num_pos_identity_experts < 0 or self.num_neg_identity_experts < 0:
            raise ValueError("identity expert counts must be >= 0, got "
                             f"+{self.num_pos_identity_experts} / -{self.num_neg_identity_experts}")
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

        # POSITIONAL ENCODING IS FIXED, NOT CONFIGURABLE (Aug 14 2026). Full-attention layers get
        # NoPE; sliding-window layers get full RoPE over the whole head_dim. There is no partial
        # rotary fraction, no per-layer-type width, no second base and no NTK scaling -- each of
        # those was a knob whose value the measurement settled, so it is spelled in the code
        # instead. See docs/ and the rope round: ctx4095 came out 3.3107 (global NoPE) / 3.3245
        # (partial) / 3.3805 (full), while local width moved it 0.0037, inside the noise floor.
        self.head_dim = self.hidden_size // self.num_attention_heads

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
            # sliding_window is either ONE int (uniform) or a per-layer list -- the hierarchical
            # form, e.g. 128 on the first windowed layer of a block and 512 on the second, so a
            # block refines locally then widens. A list must cover EVERY layer, not just the
            # windowed ones, so the index is the plain layer_idx and cannot drift out of step with
            # hybrid_layer_pattern. Entries at global layers are ignored (write 0 or None there).
            if isinstance(self.sliding_window, (list, tuple)):
                if len(self.sliding_window) != self.num_hidden_layers:
                    raise ValueError(
                        f"sliding_window is a list of {len(self.sliding_window)} but the model has "
                        f"{self.num_hidden_layers} layers; give one entry per layer (use 0/None on "
                        f"global layers) so the index is layer_idx."
                    )
                per_layer = list(self.sliding_window)
                bad = [(i, w) for i, (v, w) in enumerate(zip(self.hybrid_layer_pattern, per_layer))
                       if v and not (isinstance(w, int) and w > 0)]
                if bad:
                    raise ValueError(
                        f"hybrid_layer_pattern marks layer(s) {[i for i, _ in bad]} as windowed but "
                        f"their sliding_window entries are {[w for _, w in bad]} -- those layers "
                        f"would silently run FULL attention."
                    )
                # `sliding_window` itself must stay a SCALAR: transformers' DynamicCache reads it
                # off the config to crop sliding layers (`-self.sliding_window + 1`) and a list
                # raises TypeError there. Training never notices (use_cache=False); the eval and
                # sampling paths would die. Keep the MAX here so the cache retains at least what
                # the widest layer needs -- over-retaining is harmless because the band mask still
                # restricts attention -- and read the real per-layer value off the attribute below.
                self.sliding_window_per_layer = per_layer
                self.sliding_window = max(w for w in per_layer if isinstance(w, int) and w > 0)
            elif any(self.hybrid_layer_pattern) and not (
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
