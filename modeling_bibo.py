"""

PyTorch BiBo model
Apache 2.0 License

- adi-kmt

"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from einops import rearrange

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeModelOutputWithPast,       # Using MoE specific outputs
    MoeCausalLMOutputWithPast,    # Using MoE specific outputs
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from configuration_bibo import BiBoConfig
from transformers.cache_utils import Cache,DynamicCache, SlidingWindowCache, StaticCache

logger = logging.get_logger(__name__)



# for norm we have 3 type of norm
# Rms-Norm(arxiv: https://arxiv.org/abs/1910.07467 ),
# Dynamic-tanh(DyT) (arxiv: https://arxiv.org/abs/2503.10622 ) 
# Dynamic-Erf (Proposed by us; since our little experiments it to show more closeness to RMS)

# BiBoRMSNorm adopted from Qwen3RMSNorm
class BiBoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        Args:
            hidden_size (int): Size of the last dimension of the input tensor.
            eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5.

        Input shape:
            hidden_states: (batch_size, ..., hidden_size)

        Output shape:
            Same as input: (batch_size, ..., hidden_size)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# BiBoDyTNorm adopted from https://github.com/jiachenzhu/DyT/blob/main/dynamic_tanh.py
class BiBoDyTNorm(nn.Module):
    """
    A PyTorch module implementing a dynamic Tanh activation with learnable scaling (alpha), weight, and bias parameters.

    Args:
        hidden_size (int): Size of the hidden dimension to normalize.
        alpha_init_value (float, optional): Initial value for the learnable alpha parameter. Default is 0.69.

    The forward pass applies a scaled Tanh activation followed by an affine transformation.
    """
    def __init__(self, hidden_size, alpha_init_value=0.69):
        super().__init__()
        self.normalized_shape = hidden_size
        self.alpha_init_value = alpha_init_value
        # self.channels_last = channels_last # position of hidden_states ## refer to dyt they dont use bais for llm (llama)

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        # self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


# proposed better alternative to dyt 
class BiBoErfNorm(nn.Module):
    """
    Applies a parameterized Error Function (erf) element-wise:
        f(x) = weight * erf(alpha * x)
    """
    def __init__(self, hidden_size, alpha_init_value=1.0):
        """
        Args:
            hidden_size (int): The size of the last dimension of the input tensor.
            alpha_init_value (float): Initial value for the scalar 'alpha' parameter.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.tensor(alpha_init_value))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.bias = nn.Parameter(torch.zeros(hidden_size)) # Optional bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the parameterized erf function."""

        # if hasattr(self, 'bias'):
        #     output = output + self.bias
        return self.weight * torch.erf(self.alpha * x)

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, alpha_init_value={self.alpha_init_value}"



class BiBoRotaryEmbedding(nn.Module):

    """
    Implements rotary positional embeddings for the BiBo model.

    Initializes and caches cosine and sine embedding values up to a maximum sequence length.
    During the forward pass, returns the cached cosine and sine embeddings for the given sequence length,
    updating the cache if a longer sequence is encountered.

    Args:
        dim (int): Embedding dimension.
        max_position_embeddings (int, optional): Maximum sequence length to cache. Default is 2048.
        base (int, optional): Base for computing inverse frequencies. Default is 10000.
        device (torch.device, optional): Device for tensor allocation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine positional embeddings for the input sequence length.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def _rotate_half(x):

    """
    Splits the last dimension of the input tensor in half and rotates the halves by concatenating the negated second half with the first half.

    Args:
        x (torch.Tensor): Input tensor of shape (..., hidden_dim).

    Returns:
        torch.Tensor: Tensor with rotated halves along the last dimension.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class BiBoMLP(nn.Module):

    """
    MLP module for the BiBo model.

    This implements a standard MLP with SwiGLU-style gating:
    - gate_proj and up_proj process the input in parallel
    - gate output is activated and multiplied with up_proj output
    - down_proj projects back to hidden dimension

    Args:
        config (BiBoConfig): Model configuration object
        is_expert (bool, optional): Whether this MLP is used as an expert in MoE. 
    Determines which intermediate size to use. Defaults to False.
    """
    def __init__(self, config: BiBoConfig, is_expert=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if is_expert else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BiBoIdentityExpert(nn.Module):
    """Identity/Residual Expert to increase combinatrics (hack) """
    def __init__(self, config: BiBoConfig, *args, **kwargs): 
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BiBoCausalConv1D(nn.Module):
    """
    Implements a 1D causal convolutional expert block for the BiBo-MoE architecture.

    This module applies a causal (left-padded) 1D convolution to the input sequence, followed by a gated activation and linear projections.
    The convolution is used as a gating mechanism over the hidden states, enabling the model to capture local sequential dependencies
    while preserving causality (i.e., no information leakage from future tokens).

    Args:
        config (BiBoConfig): Configuration object containing model hyperparameters such as hidden size, intermediate size,
                             kernel size, and activation function.

    Attributes:
        hidden_size (int): model dim
        intermediate_size (int): shared expert dim
        kernel_size_gate (int): kernel window
        causal_padding_gate (int): kernel - 1
        gate_conv (nn.Conv1d): 1D convolutional layer for gating.
        up_proj (nn.Linear): Linear projection from hidden_size to intermediate_size.
        down_proj (nn.Linear): Linear projection from intermediate_size back to hidden_size.
        act_fn (Callable): Activation function applied after the convolution.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).

    Raises:
        RuntimeError: If the output sequence length does not match the input sequence length (should not occur if padding is correct).

    Example:
        >>> config = BiBoConfig()
        >>> layer = CausalConv1D(config)
        >>> x = torch.randn(2, 16, config.hidden_size)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 16, config.hidden_size])
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.kernel_size_gate = config.kernel_size
        self.causal_padding_gate = self.kernel_size_gate - 1

        # only keep gating of hidden states convulation
        self.gate_conv = nn.Conv1d(self.hidden_size, self.intermediate_size, self.kernel_size_gate, padding=0, bias=False)

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = x.shape
        x_perm = rearrange(x, 'b s h -> b h s')  # [bsz, seq_len, hidden_dim] -> [bsz, hidden_dim, seq_len]
        
        # Apply causal padding manually on the left
        # k-1 "Pad tensors in seq_len"
        x_padded = F.pad(x_perm, (self.causal_padding_gate, 0))
        gate_conv_out = self.gate_conv(x_padded)  # [bsz, intermediate_size, seq_len]
        gate_output = rearrange(gate_conv_out, 'b i s -> b s i')
        output = self.down_proj(self.act_fn(gate_output) * self.up_proj(x))

        if output.shape[1] != seq_len:
             raise RuntimeError(f"ModifiedConvExpert length mismatch. Expected {seq_len}, got {output.shape[1]}")
        return output



class BiBoMoERouter(nn.Module):
    """
    Mixture of Experts router for the BiBo model.
    
    This router determines which experts should process each token. It supports two routing mechanisms:
    1. MLP-based routing: Uses a linear projection to compute routing logits
    2. Convolution-based routing: Uses a causal 1D convolution to compute routing logits with local context
    
    Args:
        config (BiBoConfig): Configuration object containing router parameters
        
    Attributes:
        num_routed_experts (int): Number of experts to route between
        top_k (int): Number of experts to route each token to
        temperature (float): Temperature for softmax scaling
        router_noise (float): Standard deviation of noise added during training
        bias (nn.Parameter): Learnable bias for routing logits
        router_type (str): Type of router ("mlp" or "conv")
        kernel_size (int): Kernel size for convolutional router
        causal_padding (int): Padding size for causal convolution
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.temperature = config.router_temperature
        self.router_noise = config.router_noise
        self.router_type = config.router_type
        self.kernel_size = config.kernel_size
        self.causal_padding = self.kernel_size - 1  # For causal convolution

        self.bias = nn.Parameter(torch.zeros(self.num_routed_experts))        
        if self.router_type == "mlp":
            self.gate_proj = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
        elif self.router_type == "conv":
            self.gate_conv = nn.Conv1d(config.hidden_size, self.num_routed_experts, self.kernel_size, padding=0, bias=False)
        else:
            raise ValueError(f"Unknown router type: {self.router_type}. Expected 'mlp' or 'conv'.")

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass of the router.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - top_k_indices: Indices of selected experts for each token (batch_size, seq_len, top_k)
                - norm_weights: Normalized weights for selected experts (batch_size, seq_len, top_k)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if self.router_type == "mlp":
            flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
            router_logits = self.gate_proj(flat_hidden).float()
        else:  
            x_perm = rearrange(hidden_states, 'b s h -> b h s')
            x_padded = F.pad(x_perm, (self.causal_padding, 0))
            conv_out = self.gate_conv(x_padded)  # [batch_size, num_experts, seq_len]
            
            # [batch_size*seq_len, num_experts]
            router_logits = rearrange(conv_out, 'b e s -> (b s) e').float()

        if self.training and self.router_noise > 0:
            noise_stddev = math.sqrt(self.router_noise)
            noise = torch.randn_like(router_logits) * noise_stddev
            router_logits = router_logits + noise.detach()  

        router_logits = router_logits + self.bias
        if self.temperature != 1.0:
            router_logits = router_logits / self.temperature

        routing_weights = F.softmax(router_logits, dim=1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        norm_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-6)
        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.to(hidden_states.dtype)




class BiBoMoELayer(nn.Module):
    """
    Mixture of Experts (MoE) layer for the BiBo model.
    Args:
        config (BiBoConfig): Configuration object containing MoE parameters including:
            - hidden_size: Dimension of hidden representations
            - num_routed_experts: Number of experts to route between
            - num_experts_per_tok: Number of experts each token is routed to (top-k)
            - bias_update_factor: Learning rate for router bias updates
            - bias_update_threshold: Threshold for triggering bias updates
            - num_shared_experts: Number of shared experts (typically 1 Conv expert)
    
    Attributes:
        hidden_size (int): Dimension of hidden representations
        num_routed_experts (int): Number of experts to route between
        num_experts_per_tok (int): Number of experts each token is routed to (top-k)
        bias_update_factor (float): Learning rate for router bias updates
        bias_update_threshold (float): Threshold for triggering bias updates
        routed_experts (nn.ModuleList): List of routed expert modules (MLPs + Identity)
        shared_experts_list (nn.ModuleList): List of shared expert modules (typically Conv)
        gate (BiBoMoERouter): Router module that determines token-to-expert assignment
    """
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.bias_update_factor = config.bias_update_factor 
        self.bias_update_threshold = config.bias_update_threshold

        # Initialize token counter and accumulated TPE for threshold-based bias updates
        self.register_buffer("tokens_processed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("accumulated_tpe", torch.zeros(config.num_routed_experts, dtype=torch.float))
        
        self.routed_experts = nn.ModuleList()
        num_mlp_routed = config.num_routed_experts - 1 # - 1 for identity expert 
                                                       # (we can more likely add min(1,top_k%4) as identity expert to increase combinatrics ) 
        if num_mlp_routed < 0:
            raise ValueError("num_routed_experts must be at least 1 (for IdentityExpert)")
        for _ in range(num_mlp_routed):
            self.routed_experts.append(BiBoMLP(config,is_expert=True))
        self.routed_experts.append(BiBoIdentityExpert(config))
        if len(self.routed_experts) != config.num_routed_experts:
            raise ValueError(f"Mismatch: Created {len(self.routed_experts)} routed experts, expected {config.num_routed_experts}")


        self.shared_experts_list = nn.ModuleList()
        if config.num_shared_experts is not None and config.num_shared_experts > 0:
             if config.num_shared_experts != 1:
                 warnings.warn(f"BiBoMoELayer configured for 1 shared Conv expert, but got num_shared_experts={config.num_shared_experts}. Using 1 Conv expert.", UserWarning)
             self.shared_experts_list.append(BiBoCausalConv1D(config))
        elif config.num_shared_experts is None:
             warnings.warn("num_shared_experts is None, assuming 0 shared experts.", UserWarning)
        self.gate = BiBoMoERouter(config)


    @torch.no_grad() # bias update based on heuristic ; no grads needed 
                     # can also be used like async(i.e even when its inside a step and can skip a step if we want)     
    def update_bias(self, tokens_per_expert: torch.Tensor):
        """
        Updates the router's learnable bias based on token distribution.
        Aims to balance load by increasing bias for under-utilized experts
        and decreasing bias for over-utilized experts.
        
        Args:
            tokens_per_expert (torch.Tensor): Count of tokens routed to each expert
                in the current batch
                
        Notes:
            - Increases bias for under-utilized experts (below mean token count)
            - Decreases bias for over-utilized experts (above mean token count)
            - Uses sign of deviation from mean to determine update direction
            - Update magnitude controlled by bias_update_factor
            - No gradients are computed for this update (torch.no_grad)
        """
        if not hasattr(self.gate, 'bias') or self.bias_update_factor <= 0:
            return

        """
        Heuristic based load balancing
        """

        tpe = tokens_per_expert.detach().float() 
        if self.num_routed_experts > 0:
             mean_tpe = tpe.mean()
             deviation = mean_tpe - tpe 
        else:
             deviation = torch.zeros_like(tpe) 

        # Update bias: add_(factor * sign(deviation))
        # bias increases if deviation > 0 (expert under-utilized)
        # bias decreases if deviation < 0 (expert over-utilized)
        self.gate.bias.add_(self.bias_update_factor * deviation.sign())


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing decisions from the router
        top_k_indices, top_k_weights = self.gate(hidden_states) # noqa: The router already handles the proper reshaping internally
        # top_k_indices: [bsz, seq_len, top_k], top_k_weights: [bsz, seq_len, top_k]


        """
        global context to update heuristic
        """

        tokens_per_expert = None
        if self.training and hasattr(self.gate, 'bias') and self.bias_update_factor > 0:
            current_tpe = torch.bincount(
                rearrange(top_k_indices, 'b s k -> (b s k)'), 
                minlength=self.num_routed_experts
            )
            
            batch_tokens = bsz * seq_len
            self.tokens_processed += batch_tokens
            self.accumulated_tpe += current_tpe.float()
            
            if self.tokens_processed >= self.bias_update_threshold:
                # Use accumulated TPE for bias update
                tokens_per_expert = self.accumulated_tpe.clone()
                
                # Reset buffer
                self.tokens_processed.zero_()
                self.accumulated_tpe.zero_()


        flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h') 
        final_routed = torch.zeros_like(flat_hidden) 
        flat_expert_indices = rearrange(top_k_indices, 'b s k -> (b s k)')
        flat_weights = rearrange(top_k_weights, 'b s k -> (b s k)')

        flat_token_indices = torch.arange(
            bsz * seq_len, device=hidden_states.device
        ).repeat_interleave(self.num_experts_per_tok)

        for i, expert in enumerate(self.routed_experts):
            # Find which entries in flat_expert_indices correspond to the current expert (i)
            mask = (flat_expert_indices == i)

            if mask.any(): # if any token
                # Get the indices 
                tokens_idx_for_expert = flat_token_indices[mask]
                weights_for_expert = flat_weights[mask].unsqueeze(1)

                # Optimization: Process unique tokens
                unique_tokens, inverse_indices = torch.unique(tokens_idx_for_expert, return_inverse=True)
                inputs_for_expert = flat_hidden[unique_tokens]
                outputs_for_expert_unique = expert(inputs_for_expert)
                outputs_for_expert = outputs_for_expert_unique[inverse_indices]
                weighted_output = outputs_for_expert * weights_for_expert

                final_routed.scatter_add_(
                    0, 
                    tokens_idx_for_expert.unsqueeze(1).expand(-1, hidden_dim), 
                    weighted_output
                )

        final_routed = rearrange(final_routed, '(b s) h -> b s h', b=bsz)
        shared_combined = torch.zeros_like(hidden_states) 
        if self.shared_experts_list:
            shared_combined = self.shared_experts_list[0](hidden_states)
        final_output = final_routed + shared_combined


        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for GQA.
    Args:
        hidden_states: (batch, num_key_value_heads, seq_len, head_dim)
        n_rep: number of repetitions (num_query_heads // num_key_value_heads)
    Returns:
        (batch, num_query_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)



class BiBoAttention(nn.Module):
    """
    Multi-Layer KV Sharing + Grouped Query Attention (MLKV-GQA) for BiBo model.

    This module supports:
    1. Grouped Query Attention (GQA): Multiple Q heads, fewer K/V heads per layer
    2. Multi-layer KV sharing (MLKV): Only one set of K/V projections for every group of layers, controlled by config.num_layer_kv_sharing

    Args:
        config (BiBoConfig): Model configuration
        layer_idx (int): Index of this layer in the stack
        use_sliding_window (bool, optional): Whether this layer should use sliding window attention.
    """
    def __init__(self, config: BiBoConfig, layer_idx: int, use_sliding_window: Optional[bool] = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_idx
        self.use_ssmax=config.use_ssmax
        # self.num_layer_kv_sharing = config.num_layer_kv_sharing


        if self.use_ssmax:
            self.ssmax_scale = nn.Parameter(torch.full((1, self.num_heads, 1, 1), 1.0) , requires_grad=True)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            )

        # Configure sliding window attention if enabled
        self.sliding_window = config.sliding_window if use_sliding_window else None
        # sliding_window_stride can be set in config for strided window attention
        self.attention_dropout = config.attention_dropout



        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        # rope
        self.rotary_emb = BiBoRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta,
        )

        # Norms
        if config.layer_norm_type == "rms":
            self.q_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = BiBoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        elif config.layer_norm_type == "dyt":
            self.q_norm = BiBoDyTNorm(self.head_dim)
            self.k_norm = BiBoDyTNorm(self.head_dim)
        elif config.layer_norm_type == "erf":
            self.q_norm = BiBoErfNorm(self.head_dim)
            self.k_norm = BiBoErfNorm(self.head_dim)
        else:
            raise ValueError(f"Unknown layer_norm_type: {config.layer_norm_type}")


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        batch_size, q_len, _, _ = query_states.shape
        
        if self.sliding_window is not None and False:
            stride = getattr(self.config, 'sliding_window_stride', None)
            attn_output = self.eager_sliding_window_attention(
                query_states, 
                key_states, 
                value_states, 
                attention_mask,
                self.sliding_window,
                stride
            )
        else:
            attn_output = self.eager_standard_attention(
                query_states, 
                key_states, 
                value_states, 
                attention_mask
            )
        
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    
    def eager_sliding_window_attention(self):
        pass
    
    
    def eager_standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,

    ) -> torch.Tensor:
        """
        Compute standard attention.
        
        Args:
            query_states: (batch_size, num_heads, q_len, head_dim)
            key_states: (batch_size, num_heads, kv_len, head_dim)
            value_states: (batch_size, num_heads, kv_len, head_dim)
            attention_mask: Optional attention mask
            
        Returns:
            torch.Tensor: Output tensor after applying standard attention
        """


        kv_len = key_states.shape[-2]
        if self.use_ssmax:
            log_n = torch.log(torch.clamp(torch.tensor(kv_len, device=query_states.device, dtype=self.ssmax_scale.dtype), min=2.0))
            # min=2.0 since log(1) = 0 and negative for <1
            query_states = query_states * self.ssmax_scale * log_n
            
            # Standard Softmax Ratio: exp(z_i) / exp(z_k) = exp(z_i - z_k)
            # SSMax Ratio: exp(C * z_i) / exp(C * z_k) = exp(C * z_i - C * z_k) = exp(C * (z_i - z_k)) = (exp(z_i - z_k))^C
            # C is scaling factor i.e s*log(seq_len) ; 
            # in a gist: a learnable, seq-len adaptive temperature applied per head to control attention sharpness, preventing fading in long contexts.



        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights * s_scaled

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        return attn_output




        
