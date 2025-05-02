"""

PyTorch BiBo model

- adi-kmt

"""

import math
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

logger = logging.get_logger(__name__)



# for norm we have 3 type of norm
# Rms-Norm(arxiv: https://arxiv.org/abs/1910.07467 ),
# Dynamic-tanh(DyT) (arxiv: https://arxiv.org/abs/2503.10622 ) 
# Dynamic-Erf (Proposed by us; since our little experiments it to show more closeness to RMS)

# BiBoRMSNorm adopted from Qwen3RMSNorm
class BiBoRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Args:
            hidden_size (int): Size of the last dimension of the input tensor.
            eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-6.

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

class CausalConv1D(nn.Module):
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
    """BiBo Mixture of Experts Layer with Convolutional Shared Expert"""
    def __init__(self, config: BiBoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.bias_update_factor = config.bias_update_factor # For bias update logic

        # --- Routed Experts ---
        self.routed_experts = nn.ModuleList()
        num_mlp_routed = config.num_routed_experts - 1 # Account for Identity expert
        if num_mlp_routed < 0:
            raise ValueError("num_routed_experts must be at least 1 (for IdentityExpert)")
        for _ in range(num_mlp_routed):
            self.routed_experts.append(MLPExpert(config))
        # Add the Identity expert at the end
        self.routed_experts.append(IdentityExpert(config))
        if len(self.routed_experts) != config.num_routed_experts:
            raise ValueError(f"Mismatch: Created {len(self.routed_experts)} routed experts, expected {config.num_routed_experts}")

        # --- Shared Experts ---
        self.shared_experts_list = nn.ModuleList()
        if config.num_shared_experts is not None and config.num_shared_experts > 0:
             if config.num_shared_experts != 1:
                 warnings.warn(f"BiBoMoELayer configured for 1 shared Conv expert, but got num_shared_experts={config.num_shared_experts}. Using 1 Conv expert.", UserWarning)
             # Instantiate the single convolutional shared expert
             self.shared_experts_list.append(ModifiedConvolutionalExpert(config))
        elif config.num_shared_experts is None:
             warnings.warn("num_shared_experts is None, assuming 0 shared experts.", UserWarning)
        # No warning if num_shared_experts is explicitly 0

        # --- Router ---
        self.gate = BiBoMoERouter(config)


    @torch.no_grad() # Bias update should not require gradients
    def update_bias(self, tokens_per_expert: torch.Tensor):
        """
        Updates the router's learnable bias based on token distribution.
        Aims to balance load by increasing bias for under-utilized experts
        and decreasing bias for over-utilized experts.
        """
        # Check if bias exists and update factor is positive
        if not hasattr(self.gate, 'bias') or self.bias_update_factor <= 0:
            return

        tpe = tokens_per_expert.detach().float() # Ensure float and detach

        # Calculate deviation from the mean number of tokens per expert
        # Avoid division by zero if num_routed_experts is 0 (shouldn't happen with checks)
        if self.num_routed_experts > 0:
             mean_tpe = tpe.mean()
             deviation = mean_tpe - tpe # Positive if below mean, negative if above
        else:
             deviation = torch.zeros_like(tpe) # No deviation if no experts

        # Update bias: add_(factor * sign(deviation))
        # bias increases if deviation > 0 (expert under-utilized)
        # bias decreases if deviation < 0 (expert over-utilized)
        update_amount = self.bias_update_factor * deviation.sign()
        self.gate.bias.add_(update_amount)
        # Optional: Clamp bias to prevent extreme values?
        # self.gate.bias.data.clamp_(-max_bias_value, max_bias_value)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ MoE forward pass including routing, expert computation, and combination. """
        bsz, seq_len, hidden_dim = hidden_states.shape
        num_tokens = bsz * seq_len
        flat_hidden = hidden_states.view(num_tokens, -1) # [num_tokens, hidden_dim]

        # 1. Get routing decisions (indices and weights)
        top_k_indices, top_k_weights = self.gate(hidden_states)
        # top_k_indices: [num_tokens, top_k], top_k_weights: [num_tokens, top_k]

        # 2. Calculate Tokens Per Expert (TPE) for bias update (training only)
        tokens_per_expert = None
        if self.training and hasattr(self.gate, 'bias') and self.bias_update_factor > 0:
            # Bincount needs non-negative indices
            tpe = torch.bincount(top_k_indices.view(-1), minlength=self.num_routed_experts)
            tokens_per_expert = tpe # Store for bias update later

        # 3. Dispatch tokens to experts and compute routed output
        final_routed = torch.zeros_like(flat_hidden) # Initialize output buffer

        # Flatten indices and weights for easier processing
        flat_expert_indices = top_k_indices.view(-1) # [num_tokens * top_k]
        flat_weights = top_k_weights.view(-1)       # [num_tokens * top_k]

        # Create token indices corresponding to the flattened expert indices
        # Each token appears top_k times in this list
        flat_token_indices = torch.arange(
            num_tokens, device=hidden_states.device
        ).repeat_interleave(self.num_experts_per_tok) # [num_tokens * top_k]

        # Iterate through each expert to process its assigned tokens
        for i, expert in enumerate(self.routed_experts):
            # Find which entries in flat_expert_indices correspond to the current expert (i)
            mask = (flat_expert_indices == i)

            if mask.any(): # If any tokens are routed to this expert
                # Get the indices of the tokens routed to this expert
                tokens_idx_for_expert = flat_token_indices[mask] # Indices into the original num_tokens dimension

                # Get the corresponding weights
                weights_for_expert = flat_weights[mask].unsqueeze(1) # [num_tokens_for_this_expert, 1]

                # Optimization: Process unique tokens only once if top_k > 1
                unique_tokens, inverse_indices = torch.unique(tokens_idx_for_expert, return_inverse=True)

                # Select the unique input hidden states for these tokens
                inputs_for_expert = flat_hidden[unique_tokens] # [num_unique_tokens, hidden_dim]

                # Compute expert output for unique tokens
                outputs_for_expert_unique = expert(inputs_for_expert) # [num_unique_tokens, hidden_dim]

                # Expand the unique outputs back to match the original number of routed tokens
                # using the inverse_indices from torch.unique
                outputs_for_expert = outputs_for_expert_unique[inverse_indices] # [num_tokens_for_this_expert, hidden_dim]

                # Weight the expert output
                weighted_output = outputs_for_expert * weights_for_expert

                # Add the weighted output to the final result using scatter_add_
                # This efficiently adds contributions for the same token routed to multiple experts
                final_routed.scatter_add_(0, tokens_idx_for_expert.unsqueeze(1).expand(-1, hidden_dim), weighted_output)

        # Reshape routed output back to original shape
        final_routed = final_routed.view(bsz, seq_len, hidden_dim)

        # 4. Compute shared expert output (if configured)
        shared_combined = torch.zeros_like(hidden_states) # Default if no shared experts
        if self.shared_experts_list:
            # Assuming only one shared expert (the Conv one) as per design
            shared_combined = self.shared_experts_list[0](hidden_states)

        # 5. Combine routed and shared outputs
        final_output = final_routed + shared_combined

        # 6. Update router bias (after forward pass, using TPE calculated earlier)
        if tokens_per_expert is not None:
            self.update_bias(tokens_per_expert)

        return final_output
