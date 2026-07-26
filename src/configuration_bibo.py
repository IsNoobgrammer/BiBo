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
        use_ssmax=True,  # SSMax: scaling softmax for long context (forced OFF on SWA layers)
        attention_dropout=0.0,
        attention_bias=False,
        # ── Hybrid SWA / attention sink (docs/attention_layers.md) ──
        hybrid_layer_pattern=None,       # per-layer list: 1=sliding-window, 0=full. None => all-global (current)
        sliding_window=128,              # SWA window W (keys visible per query on windowed layers)
        add_swa_attention_sink_bias=True,   # learnable per-head sink on SWA layers (the norm; unscaled)
        add_full_attention_sink_bias=False,  # sink on global layers (False => G2, current behavior)
        # ── RoPE (dim-wise partial: first rope_dim of EVERY head rotate, rest NoPE) ──
        rope_theta=None,    # Auto-derived (1e7) if None — matched to partial rotary
        rope_scaling=None,  # Auto: {"type": "dynamic", "factor": 1.0} if None
        partial_rotary_factor=0.334,  # fraction of head_dim that gets RoPE (dim-wise, all heads). 1.0 = full RoPE.
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
        polyglu_expert_multiplier=2,  # Groups of 3 (SiLU, ReLU², NormSiLU) GLU experts
        special_expert_pairs=1,       # Count of special experts PER TYPE (see pos/neg_identity_expert)
        pos_identity_expert=True,     # include the +Identity special expert(s) (code 3, param-free  +w*x)
        neg_identity_expert=True,     # include the -Identity special expert(s) (code 4, param-free  -w*x)
        # ── Shared expert ────────────────────────────────────────
        use_shared_expert=False,    # Off by default (param-match Qwen3MoE — no shared expert)
        shared_expert_type="mlp",   # "mlp" (SwiGLU, like Qwen) or "conv" (CausalConv1D)
        # ── Router ───────────────────────────────────────────────
        router_type="mlp",    # "mlp" (Linear, default) | "conv" (causal Conv1d over kernel_size taps)
        kernel_size=3,        # kernel width for BOTH the conv router and the conv shared expert
        gate_type="sigmoid",       # "sigmoid" | "situ" (SIGNED, needs norm_topk_prob=True) | "softmax"
        router_activation="none",  # on the LOGITS, before gate_type: "none" | "relu" | "silu"
        norm_topk_prob=True,       # softmax the gathered top-k weights to sum to 1 (not MiMo's ÷sum)
        routed_scaling_factor=1.0,  # post-norm routed-weight scale; 1.0 = no-op
        load_balance_strategy="bias",  # "none" | "bias" (aux-loss-free bias updates)
        bias_update_factor=0.001,   # sign-update step; small on purpose, see __init__ note
        bias_update_threshold=8000,  # tokens between bias updates
        balance_exclude_specials=False,  # balance only the GLU block, freeze ±Identity biases at 0
        glu_token_budget=None,      # LongCat K_e/K: fraction of the k slots/token targeted at the GLU
                                    # block (e.g. 0.75 -> GLU 3/4, ±Identity 1/4). None = DeepSeek
                                    # mean-relative balancing. See BiBoMoELayer.update_bias.
        **kwargs,
    ):
        # Pop removed knobs: PretrainedConfig setattr()s unknown kwargs, so a stale value would
        # reappear as an attribute AND be re-serialized into config.json as if it still existed.
        for dead, why in (("router_noise", "noise injection removed Jul 26 2026; was never enabled"),
                          ("zero_expert", "Zero expert removed Jul 26 2026; use neg_identity_expert"),
                          ("identity_expert", "renamed Jul 26 2026; use pos_identity_expert")):
            if kwargs.pop(dead, None) is not None:
                logger.warning(f"`{dead}` was removed ({why}). Ignored, and not saved.")

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

        # ── Hybrid SWA / attention sink ──────────────────────────
        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.sliding_window = sliding_window
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias
        self.add_full_attention_sink_bias = add_full_attention_sink_bias

        # ── RoPE (dim-wise partial) ──────────────────────────────
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        # head_dim / rope_dim are DERIVED below (after super().__init__) so a stale value serialized
        # into config.json can't override the value implied by hidden_size / heads / partial factor.

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
        self.pos_identity_expert = pos_identity_expert
        self.neg_identity_expert = neg_identity_expert
        # The two specials are SIGNED param-free passthroughs: +w*x and -w*x. There is no Zero expert
        # (removed Jul 26 2026) — a Zero expert's output is 0, so its direct weight gradient is
        # identically 0 and the router can never learn that picking it was right; softmax coupling then
        # leaves only negative pressure on its score, so all Zero usage is forced by the load balancer
        # AGAINST the gradient. The ± pair still spans "skip this layer" (w+ ≈ w- cancels) but reaches
        # it with live gradients on both branches. LongCat-Flash (arXiv:2509.01322) likewise uses
        # identity, never zero, for its 256 zero-COMPUTATION experts.
        # Each sign is toggleable so either can be ablated alone; special_expert_pairs is the per-type
        # count and a disabled type is a zero-width block. Layout stays GLU-first (kernel weight slot ==
        # expert index for GLU), then the +Identity block, then the -Identity block.
        self.num_pos_identity_experts = special_expert_pairs if pos_identity_expert else 0
        self.num_neg_identity_experts = special_expert_pairs if neg_identity_expert else 0
        # experts = polyglu_multiplier * 3 (SiLU, ReLU², NormSiLU) + (+Identity) block + (-Identity) block
        self.num_routed_experts = ((polyglu_expert_multiplier * 3)
                                   + self.num_pos_identity_experts + self.num_neg_identity_experts)

        # ── Shared expert ────────────────────────────────────────
        self.use_shared_expert = use_shared_expert
        self.shared_expert_type = shared_expert_type

        # ── Router ───────────────────────────────────────────────
        self.router_type = router_type
        self.kernel_size = kernel_size
        self.gate_type = gate_type
        self.router_activation = router_activation
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.load_balance_strategy = load_balance_strategy

        # ── Auto-derived when left as None ────────────────────────

        # 1e7 matches dim-wise partial RoPE: fewer rotated dims want longer wavelengths (MiMo-V2.5
        # pairs partial_rotary_factor=0.334 with theta=1e7).
        self.rope_theta = rope_theta if rope_theta is not None else 1e7

        # Compute parity with the dense FFN: dense active/token = 2*hidden*intermediate,
        # MoE = 2*hidden*moe_intermediate*top_k, so moe_intermediate = intermediate // top_k.
        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size is not None
            else self.intermediate_size // self.num_experts_per_tok
        )

        # A FIXED small step, NOT a function of num_routed_experts (the old auto-Hill grew u from
        # 0.07 at n=8 to 0.35 asymptotically, which is backwards). With independent per-expert
        # scoring (sigmoid / situ) the score distribution does not move with n — only the order
        # statistics get denser — so the bias distance needed to flip a top-k selection SHRINKS as
        # experts are added: measured mean gap at the k|k+1 boundary is 0.041 (n=8) -> 0.0064
        # (n=128) -> 0.0045 (n=512) for sigmoid. A small fixed step is therefore *more* than
        # sufficient at scale, not less.
        # Why this small: sign() never returns 0, so the bias dithers +-u forever and never settles;
        # u IS the balancer's steady-state routing-noise floor. u must stay well under the boundary
        # gap or it reshuffles selections on its own even at perfect balance (u=0.03 is ~5x the
        # n=128 gap; u=0.001 is ~0.16x). 0.001 also matches DeepSeek-V3. Raise it for a faster
        # response at the cost of more routing jitter; 0 disables balancing entirely.
        self.bias_update_factor = 0.001 if bias_update_factor is None else bias_update_factor

        self.bias_update_threshold = bias_update_threshold if bias_update_threshold is not None else 8000
        self.balance_exclude_specials = balance_exclude_specials
        self.glu_token_budget = glu_token_budget

        # Dynamic NTK-aware: identity inside the trained window, smooth base growth beyond it.
        # type="none" for plain RoPE.
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "dynamic", "factor": 1.0}

        self.mlp_only_layers = (
            mlp_only_layers if mlp_only_layers is not None
            else sorted({0, num_hidden_layers - 1})   # dedupe: N==1 -> [0], not [0,0]
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # AFTER super().__init__(**kwargs): a stale serialized head_dim/rope_dim arrives via kwargs
        # on reload and must NOT win over the current derivation.
        self.head_dim = self.hidden_size // self.num_attention_heads
        _rope_dim = round(self.partial_rotary_factor * self.head_dim)
        self.rope_dim = _rope_dim - (_rope_dim % 2)   # rotate_half needs an even dim

        # ── Validations ───────────────────────────────────────────
        if self.layer_norm_type != "rms":
            raise ValueError(f"Only 'rms' layer_norm_type is supported. Got: {self.layer_norm_type}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
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
        if self.bias_update_threshold <= 0:
            raise ValueError("bias_update_threshold must be positive")
        if self.glu_token_budget is not None:
            if not 0.0 < self.glu_token_budget <= 1.0:
                raise ValueError(
                    f"glu_token_budget must be in (0, 1], got {self.glu_token_budget}")
            n_special = self.num_pos_identity_experts + self.num_neg_identity_experts
            if self.glu_token_budget < 1.0 and n_special == 0:
                raise ValueError(
                    f"glu_token_budget={self.glu_token_budget} reserves {1 - self.glu_token_budget:.2f} "
                    f"of every token's routing slots for special experts, but there are none. The "
                    f"balancer would push the GLU block below budget with nowhere for the traffic to "
                    f"go. Enable pos/neg_identity_expert with special_expert_pairs>0, or leave "
                    f"glu_token_budget=None.")
        if self.add_full_attention_sink_bias and self.use_ssmax:
            raise ValueError(
                "add_full_attention_sink_bias=True with use_ssmax=True (global sink + SSMax, 'G1') "
                "needs the sink scaled by the SSMax factor C=s·log(n) (docs/attention_layers.md §4); "
                "that coupling is not implemented yet. Disable one, or wire the C-scaled sink first."
            )
        if self.shared_expert_type not in ("mlp", "conv"):
            raise ValueError(f"shared_expert_type must be 'mlp' or 'conv', got '{self.shared_expert_type}'")
        if self.polyglu_expert_multiplier < 1:
            raise ValueError(
                "polyglu_expert_multiplier must be >= 1 (need at least one group of SiLU/ReLU²/NormSiLU GLU experts)"
            )
        if self.special_expert_pairs < 0:
            raise ValueError("special_expert_pairs must be >= 0")
        if self.num_routed_experts < 3:
            raise ValueError(
                f"num_routed_experts must be >= 3 (got {self.num_routed_experts}). Increase polyglu_expert_multiplier or special_expert_pairs."
            )
        if self.num_experts_per_tok < 1:
            raise ValueError("num_experts_per_tok must be >= 1")
        if self.num_experts_per_tok > self.num_routed_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) cannot exceed num_routed_experts "
                f"({self.num_routed_experts}) — the router only selects among routed experts."
            )
        if self.load_balance_strategy not in ("none", "bias"):
            raise ValueError(
                f"load_balance_strategy must be 'none' or 'bias', got '{self.load_balance_strategy}'"
            )
        if self.router_type not in ("mlp", "conv"):
            raise ValueError(f"router_type must be 'mlp' or 'conv', got '{self.router_type}'")
        if self.router_type == "conv" and self.kernel_size < 1:
            raise ValueError(f"router_type='conv' needs kernel_size >= 1, got {self.kernel_size}")
        if self.router_activation not in ("none", "relu", "silu"):
            raise ValueError(
                f"router_activation must be 'none', 'relu', or 'silu', got '{self.router_activation}'"
            )
        if self.gate_type not in ("sigmoid", "situ", "softmax"):
            raise ValueError(
                f"gate_type must be 'sigmoid', 'situ', or 'softmax', got '{self.gate_type}'"
            )
        # A signed gate with ÷sum-style normalization off means the combine weights can be negative
        # AND unnormalized. Softmax normalization (norm_topk_prob=True) is what makes "situ" safe.
        if self.gate_type == "situ" and not self.norm_topk_prob and self.num_experts_per_tok > 1:
            raise ValueError(
                "gate_type='situ' produces SIGNED scores; set norm_topk_prob=True so the top-k "
                "weights are softmax-normalized (positive, sum to 1). Pass norm_topk_prob=True, "
                "or use gate_type='sigmoid' if you need unnormalized weights."
            )
        if self.rope_scaling.get("type") not in ("none", "dynamic"):
            raise ValueError(
                f"rope_scaling['type'] must be 'none' or 'dynamic', got {self.rope_scaling.get('type')!r}"
            )
        if self.rope_scaling.get("factor", 1.0) <= 0:
            raise ValueError("rope_scaling['factor'] must be positive")
        if not (0.0 < self.partial_rotary_factor <= 1.0):
            raise ValueError(f"partial_rotary_factor must be in (0, 1], got {self.partial_rotary_factor}")
        if self.rope_dim < 2:
            raise ValueError(
                f"partial_rotary_factor={self.partial_rotary_factor} gives rope_dim={self.rope_dim} "
                f"(head_dim={self.head_dim}); need at least 2 rotary dims. Increase the factor."
            )
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if self.hybrid_layer_pattern is not None:
            if len(self.hybrid_layer_pattern) != self.num_hidden_layers:
                raise ValueError(
                    f"hybrid_layer_pattern length ({len(self.hybrid_layer_pattern)}) must equal "
                    f"num_hidden_layers ({self.num_hidden_layers})"
                )
            if any(v not in (0, 1) for v in self.hybrid_layer_pattern):
                raise ValueError("hybrid_layer_pattern entries must be 0 (full) or 1 (sliding-window)")
            if any(v == 1 for v in self.hybrid_layer_pattern) and not (
                isinstance(self.sliding_window, int) and self.sliding_window > 0):
                raise ValueError(
                    "hybrid_layer_pattern marks SWA layers (1) but sliding_window is not a positive "
                    "int — SWA layers require a window size (else they silently run full-attention)."
                )
        # No SWA layer anywhere -> serialize sliding_window=None. HF machinery (cache selection,
        # mask utils, third-party tooling) keys off config.sliding_window; a non-None value would
        # advertise windowed attention for a fully-global model.
        _has_swa = self.hybrid_layer_pattern is not None and any(self.hybrid_layer_pattern)
        if not _has_swa:
            self.sliding_window = None
        # Standard HF per-layer attention types — lets DynamicCache(config=...) build
        # window-evicting sliding layers for SWA and unbounded layers for global.
        self.layer_types = (
            ["sliding_attention" if v else "full_attention" for v in self.hybrid_layer_pattern]
            if _has_swa else ["full_attention"] * self.num_hidden_layers
        )
        for idx in self.mlp_only_layers:
            if not (0 <= idx < self.num_hidden_layers):
                raise ValueError(
                    f"mlp_only_layers index {idx} is out of range for {self.num_hidden_layers} layers"
                )
