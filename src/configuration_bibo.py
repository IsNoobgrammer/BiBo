from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

BIBO_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# Router gate activations. Defined HERE (a leaf module) so config validation, the router, the
# ablate patch and the CLI all read one list instead of four drifting copies.
#   sigmoid   baseline, span ~[0.007,0.993] over logits +-5, floor 0.007
#   situ      sigmoid(x)*tanh(x). SIGNED and NON-MONOTONIC -> needs norm_topk_prob=True (div-sum breaks)
#   softmax   normalizes across experts before top-k
#   tsig      tanh(sigmoid(x)) in (0,0.762). tanh is CONCAVE on [0,1] so it compresses the TOP of the
#             sigmoid harder than the bottom: ASYMMETRIC (0 sits 60.6% up the span), shrinking
#             differences among HIGH-scoring experts. Less gradient-uniform than sigmoid (0.256 vs
#             0.420 over +-2) -- it is NOT a temperature, despite looking like one when plotted.
#   sigtanh   sigmoid(tanh(x)) in (0.269,0.731). tanh saturates first => soft clamp. Highest FLOOR of
#             any bounded gate (min/max 0.368) but nearly flat past |x|~2.5, so confident experts stop
#             differing by WEIGHT (selection is unaffected -- still monotone).
#   sqsp      sqrt(softplus(x)). UNBOUNDED above (2.24 at x=5) and EXPANDS the top rather than
#             compressing it. Under div-sum a runaway expert has no ceiling: watch max expert load.
GATE_TYPES = ("sigmoid", "situ", "softmax", "tsig", "sigtanh", "sqsp")
ROUTER_INPUT_NORMS = ("none", "rms", "unit")
# Per-token norm on the MoE BLOCK OUTPUT, applied to the combined expert sum O just before it
# reaches the residual stream. "rms" = O/rms(O) with a learnable per-channel gain; "unit" = the
# same but GAIN-FREE, so only the DIRECTION of the expert mixture survives and its magnitude is
# pinned exactly. This is the scale-control knob that top-k weight normalization only approximates:
# sum-to-1 bounds the WEIGHTS but the experts' own output magnitudes still ride through.
MOE_OUT_NORMS = ("none", "rms", "unit")
# GLU experts = polyglu_expert_multiplier * POLYGLU_GROUP. Was 3 when the act menu was the fixed
# triple (SiLU, ReLU2, NormSiLU). The act axis has since settled on the NORMED pair
# (normsilu, normsitu) -- relu2/NormRelu measured far worse -- so the natural group is 2, and 2
# also makes even expert counts like 64 reachable (3 never could: 63 or 66 only).
# NOTE this CHANGES what a given multiplier means: mult 10 was 30 GLU experts, it is now 20.
POLYGLU_GROUP = 2
SIGNED_GATES = ("situ",)          # gates whose scores can go negative -> div-sum normalization is invalid


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
        polyglu_expert_multiplier=2,  # GLU experts = this * POLYGLU_GROUP (2, see top of file)
        special_expert_pairs=1,       # Count of special experts PER TYPE (see pos/neg_identity_expert)
        pos_identity_expert=True,     # include the +Identity special expert(s) (code 3, param-free  +w*x)
        neg_identity_expert=True,     # include the -Identity special expert(s) (code 4, param-free  -w*x)
        # ── Shared expert ────────────────────────────────────────
        use_shared_expert=False,    # Off by default (param-match Qwen3MoE — no shared expert)
        shared_expert_type="mlp",   # "mlp" (SwiGLU, like Qwen) or "conv" (CausalConv1D)
        num_shared_experts=1,       # WIDTH multiple for the shared expert: its intermediate size is
                                    # moe_intermediate_size * this. N parallel width-I GLUs summed ==
                                    # one width-N*I GLU, so this is exactly "N shared experts" at one
                                    # GEMM (Kimi K3 does the same). Only read when use_shared_expert.
                                    # Shared output is added UNSCALED (Kimi: y = y + shared(x); DeepSeek-V3
                                    # and Gemma likewise) -- routed_scaling_factor never touches it.
        # ── Router ───────────────────────────────────────────────
        router_type="mlp",    # "mlp" (Linear, default) | "conv" (causal Conv1d over kernel_size taps)
        kernel_size=3,        # kernel width for BOTH the conv router and the conv shared expert
        gate_type="sigmoid",       # one of GATE_TYPES (top of this file); "situ" is SIGNED -> needs norm_topk_prob=True
        moe_out_norm="none",       # per-token norm on the MoE BLOCK OUTPUT (see MOE_OUT_NORMS)
        moe_latent_dim=0,          # LatentMoE: expert in/out width d (0 = off, experts run at hidden_size)
        latent_moe_use_norm=True,  # RMSNorm on the latent before the up-projection (K3 ships this on)
        router_input_norm="none",  # per-token norm on the ROUTER INPUT ONLY: "none" | "rms" (own
                                   # learnable gain) | "unit" (x/rms(x), no gain -> logits depend on
                                   # DIRECTION only). The block is pre-norm so the router and the
                                   # experts share post_attention_layernorm(h); that tensor is RMS-
                                   # normed but then scaled by a learned gain, so ||h_t|| still
                                   # varies per token and the gate temperature is effectively
                                   # per-token. Normalizing here leaves the EXPERT input untouched.
        router_temperature=1.0,    # logits are divided by this BEFORE the gate. T>1 flattens the
                                   # score distribution (derivative scales exactly 1/T), T<1 sharpens.
                                   # NOT redundant with the router weight scale under Muon: Muon pins
                                   # the weight's spectral norm, so the model cannot rescale its own
                                   # logits to absorb T. (Listed as "removed, never implemented" in
                                   # AGENTS.md until Jul 27 2026 -- this is the real implementation.)
        router_activation="none",  # on the LOGITS, before gate_type: "none" | "relu" | "silu"
        norm_topk_prob=True,       # softmax the gathered top-k weights to sum to 1 (not MiMo's ÷sum)
        routed_scaling_factor=1.0,  # post-norm routed-weight scale; 1.0 = no-op
        load_balance_strategy="bias",  # "none" | "bias" (aux-loss-free bias updates)
        bias_update_factor=None,    # None -> MODE-DEPENDENT default (prop 0.4, sign 0.001); u means
                                    # different things in each mode, so it cannot be one literal
        bias_update_threshold=8000,  # tokens between bias updates
        bias_update_mode="prop",    # DEFAULT since Jul 26 2026. "prop" (LongCat: raw deviation,
                                    # no sign()) | "sign" (DeepSeek-V3 bang-bang).
                                    # Measured, 18 GLU experts, 500 steps, replicate sigma 0.013:
                                    #   sign over  5x of u (0.001->0.005): loss moved 0.0775 (6 sigma)
                                    #   prop over 100x of u (0.02 ->2.0 ): loss moved 0.0271 (~noise)
                                    # ~8x less sensitive PER DECADE, because proportional control
                                    # has a fixed point: the steady state is set by the TARGET and u
                                    # only sets convergence RATE. sign has none, so u sets the
                                    # operating point and the dither amplitude too -- and its optimum
                                    # MOVES with the expert layout, so it needs a per-config sweep.
                                    # Cost of prop: it does not beat a hand-tuned sign (2.7186 vs
                                    # 2.7030, 1.2 sigma). We trade <=1.2 sigma of peak for not tuning.
                                    # NOTE u IS NOT COMPARABLE ACROSS MODES: sign steps by u, prop by
                                    # u*deviation, and share deviations run ~1e-3..1e-1.
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
        # experts = polyglu_multiplier * POLYGLU_GROUP + (+Identity) block + (-Identity) block
        self.num_routed_experts = ((polyglu_expert_multiplier * POLYGLU_GROUP)
                                   + self.num_pos_identity_experts + self.num_neg_identity_experts)

        # ── Shared expert ────────────────────────────────────────
        self.use_shared_expert = use_shared_expert
        self.shared_expert_type = shared_expert_type
        self.num_shared_experts = num_shared_experts

        # ── Router ───────────────────────────────────────────────
        self.router_type = router_type
        self.kernel_size = kernel_size
        self.gate_type = gate_type
        self.router_input_norm = router_input_norm
        self.moe_out_norm = moe_out_norm
        self.moe_latent_dim = int(moe_latent_dim or 0)
        self.latent_moe_use_norm = bool(latent_moe_use_norm)
        self.router_temperature = router_temperature
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
        # The default depends on the MODE, because u means different things in each:
        #   prop (default): step = u * deviation, deviations ~1e-3..1e-1 -> u ~ 0.4.
        #   sign:           step = u exactly                             -> u ~ 0.001..0.005.
        # For sign, u is the balancer's steady-state routing-noise floor: any nonzero deviation
        # gives a full +-u step, so it dithers forever and u must stay well under the selection
        # boundary gap or it reshuffles routing even at perfect balance. prop has a fixed point and
        # no such floor. 0 disables balancing entirely in both modes.
        _u_default = 0.001 if bias_update_mode == "sign" else 0.4
        self.bias_update_factor = _u_default if bias_update_factor is None else bias_update_factor

        self.bias_update_threshold = bias_update_threshold if bias_update_threshold is not None else 8000
        self.balance_exclude_specials = balance_exclude_specials
        self.glu_token_budget = glu_token_budget
        self.bias_update_mode = bias_update_mode

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
        if self.bias_update_mode not in ("sign", "prop"):
            raise ValueError(f"bias_update_mode must be 'sign' or 'prop', got '{self.bias_update_mode}'")
        # u is NOT comparable across modes and the prop-scale default is catastrophic under sign:
        # sign steps by u EXACTLY, so u=0.4 moves the bias 0.4 per update against sigmoid scores that
        # live in (0,1) -- it would obliterate routing. Catch the mode flip that forgets to rescale u.
        if self.bias_update_mode == "sign" and self.bias_update_factor > 0.05:
            raise ValueError(
                f"bias_update_mode='sign' with bias_update_factor={self.bias_update_factor} is a "
                f"prop-scale step. sign moves the bias by u EXACTLY every update, against scores in "
                f"(0,1); sane values are ~0.001-0.005. Either set bias_update_mode='prop' (the "
                f"default, where u~0.4 is correct) or lower bias_update_factor.")
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
                f"polyglu_expert_multiplier must be >= 1 (one group = {POLYGLU_GROUP} GLU experts)"
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
        if self.use_shared_expert and self.num_shared_experts < 1:
            raise ValueError(f"num_shared_experts must be >= 1 when use_shared_expert, got {self.num_shared_experts}")
        if self.router_input_norm not in ROUTER_INPUT_NORMS:
            raise ValueError(f"router_input_norm must be one of {ROUTER_INPUT_NORMS}, got '{self.router_input_norm}'")
        if self.moe_out_norm not in MOE_OUT_NORMS:
            raise ValueError(f"moe_out_norm must be one of {MOE_OUT_NORMS}, got '{self.moe_out_norm}'")
        if self.router_temperature <= 0:
            raise ValueError(f"router_temperature must be > 0, got {self.router_temperature}")
        if self.router_activation not in ("none", "relu", "silu"):
            raise ValueError(
                f"router_activation must be 'none', 'relu', or 'silu', got '{self.router_activation}'"
            )
        if self.gate_type not in GATE_TYPES:
            raise ValueError(f"gate_type must be one of {GATE_TYPES}, got '{self.gate_type}'")
        # A signed gate with ÷sum-style normalization off means the combine weights can be negative
        # AND unnormalized. Softmax normalization (norm_topk_prob=True) is what makes "situ" safe.
        if self.gate_type in SIGNED_GATES and not self.norm_topk_prob and self.num_experts_per_tok > 1:
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
