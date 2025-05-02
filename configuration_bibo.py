from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

BIBO_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class BiBoConfig(PretrainedConfig):
    r"""
    Configuration class for the BiBo model inherited from PretrainedConfig.

    Args:
        vocab_size (int): Vocabulary size.
        hidden_size (int): Hidden state dimension.
        intermediate_size (int): MLP hidden state dimension.
        num_hidden_layers (int): Number of transformer layers.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of key/value heads for GQA/MQA.
        hidden_act (str): Activation function.
        max_position_embeddings (int): Max sequence length.
        initializer_range (float): Weight initialization range.
        rms_norm_eps (float): RMSNorm epsilon.
        rms_norm_type (str): Type of normalization to use ("rms", "dyt", "erf").
        use_cache (bool): Use cache for decoding.
        pad_token_id (int): Padding token id.
        bos_token_id (int): Beginning of sentence token id.
        eos_token_id (int): End of sentence token id.
        tie_word_embeddings (bool): Tie input/output embeddings.
        rope_theta (float): RoPE base period.
        rope_scaling (dict): RoPE scaling config.
        attention_dropout (float): Attention dropout.
        use_sliding_window (bool): Use sliding window attention.
        sliding_window (int): Sliding window size.
        max_window_layers (int): Number of layers using SWA.
        attention_bias (bool): Use bias in attention projections.
        mlp_only_layers (list): Indices of layers using dense MLP (first and last by default).
        decoder_sparse_step (int): Frequency of MoE layers.
        moe_intermediate_size (int): MoE expert hidden dim.
        num_routed_experts (int): Number of routed experts.
        num_shared_experts (int): Number of shared experts.
        num_experts_per_tok (int): Number of experts per token.
        num_experts (int): Total number of experts (routed + shared).
        router_temperature (float): Router temperature.
        bias_update_factor (float): Router bias update factor.
        router_noise (float): Router noise.
        kernel_size (int): MoE kernel size.
        norm_topk_prob (bool): Normalize topk probabilities.
        output_router_logits (bool): Output router logits and aux loss.
        router_aux_loss_coef (float): Router aux loss coefficient.
    """
    model_type = "bibo"

    def __init__(
        self,
        vocab_size=128000,
        hidden_size=1536,
        intermediate_size=4104, #2nd ramanujan hardy number 1729,4104 .. etc
        # """
        # 4104 = 16³ + 2³ (16 cubed + 2 cubed) = 4096 + 8
        # 4104 = 15³ + 9³ (15 cubed + 9 cubed) = 3375 + 729
        # """
        num_hidden_layers=8,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        layer_norm_type="rms", # options are "dyt","rms","erf"
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=None,
        attention_bias=False,
        mlp_only_layers=None,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        num_routed_experts=8,
        num_shared_experts=1,
        num_experts_per_tok=2,
        num_experts=None,
        router_temperature=1.3,
        bias_update_factor=1e-2,
        bias_update_threshold=100_000, # amount of tokens(bs*seq) to update bias 
        router_noise=0.5,
        router_type="mlp", # mlp or conv
        kernel_size=3,
        norm_topk_prob=False,
        output_router_logits=False,
        conv_router=False,  # router will also be a causal conv1d with same kernel as shared expert 
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
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers if max_window_layers is not None else num_hidden_layers
        self.attention_bias = attention_bias
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts if num_experts is not None else (num_routed_experts + num_shared_experts)
        self.router_temperature = router_temperature
        self.bias_update_factor = bias_update_factor
        self.bias_update_threshold=bias_update_threshold
        self.router_noise = router_noise
        self.router_type=router_type
        self.kernel_size = kernel_size
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
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
            # print(self.hidden_size % self.num_attention_heads)
        
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
        if self.layer_norm_type not in ("rms", "dyt", "erf"):
            raise ValueError(f"rms_norm_type must be one of 'rms', 'dyt', or 'erf', got '{self.layer_norm_type}'")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive if specified")
        if self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive")
        if self.bias_update_factor < 0.0:
            raise ValueError("bias_update_factor must be non-negative")
        if self.router_noise < 0.0:
            raise ValueError("router_noise must be non-negative")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed total number of experts")
        for idx in self.mlp_only_layers:
            if not (0 <= idx < self.num_hidden_layers):
                raise ValueError(f"mlp_only_layers index {idx} is out of range for {self.num_hidden_layers} layers")
        rope_config_validation(self)


if __name__ == "__main__":
    config = BiBoConfig()
    print(config)
