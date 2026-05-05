"""Decoder layer"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.configuration_bibo import BiBoConfig
from .norm import BiBoRMSNorm
from .attn import BiBoAttention
from .ffn import BiBoMLP, BiBoMoELayer

__all__ = [
    'BiBoResidualGate',
    'BiBoCausalResidualConv',
    'BiBoMultiStreamResidual',
    'BiBoDecoderLayer',
]


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _stream_probs(num_streams: int, main_mass: float) -> torch.Tensor:
    probs = torch.full((num_streams,), (1.0 - main_mass) / (num_streams - 1), dtype=torch.float32)
    probs[0] = main_mass
    return probs


class BiBoResidualGate(nn.Module):
    """
    Optional residual write gate.

    This keeps the identity path intact and only scales the branch update:
    `x = residual + gate * branch_output`. Initializing the gate near 1.0
    preserves baseline Transformer behavior while still letting training learn
    token-wise or channel-wise residual flow control.
    """
    def __init__(self, config: BiBoConfig, branch_name: str):
        super().__init__()
        self.gate_type = config.residual_gate_type
        self.hidden_size = config.hidden_size
        self.branch_name = branch_name

        if self.gate_type == "none":
            self.weight = None
            self.bias = None
        elif self.gate_type == "scalar":
            init_logit = math.log(config.residual_gate_init / (1.0 - config.residual_gate_init))
            self.weight = None
            self.bias = nn.Parameter(torch.full((1,), init_logit))
        else:
            out_features = {
                "token": 1,
                "channel": self.hidden_size,
            }[self.gate_type]
            init_logit = math.log(config.residual_gate_init / (1.0 - config.residual_gate_init))
            # Use raw parameters instead of nn.Linear so PreTrainedModel.post_init
            # does not overwrite the near-identity gate initialization.
            self.weight = nn.Parameter(torch.zeros(out_features, self.hidden_size))
            self.bias = nn.Parameter(torch.full((out_features,), init_logit))

        self.register_buffer("last_gate_mean", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_gate_open_frac", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_gate_closed_frac", torch.tensor(float("nan")), persistent=False)

    def forward(self, gate_input: torch.Tensor, branch_output: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "none":
            return branch_output

        if self.gate_type == "scalar":
            logits = self.bias.view(1, 1, 1).expand(*branch_output.shape[:-1], 1)
        else:
            logits = F.linear(gate_input.to(self.weight.dtype), self.weight, self.bias)

        gate = torch.sigmoid(logits).to(branch_output.dtype)
        gated_output = branch_output * gate

        if not torch.jit.is_scripting():
            with torch.no_grad():
                gate_float = gate.detach().float()
                self.last_gate_mean.copy_(gate_float.mean())
                self.last_gate_open_frac.copy_((gate_float > 0.9).float().mean())
                self.last_gate_closed_frac.copy_((gate_float < 0.1).float().mean())

        return gated_output

    def stats(self) -> dict:
        if self.gate_type == "none":
            return {}
        return {
            f"{self.branch_name}/mean": float(self.last_gate_mean.item()),
            f"{self.branch_name}/open_frac": float(self.last_gate_open_frac.item()),
            f"{self.branch_name}/closed_frac": float(self.last_gate_closed_frac.item()),
        }


class BiBoCausalResidualConv(nn.Module):
    """
    Causal convolution over residual states along model depth.

    This is an attention-residual-style read mixer with a fixed causal window:
    the current layer can read only previous depth states plus its own output.
    It does not convolve over sequence tokens, so token-level causality is
    preserved as long as the underlying decoder layers are causal.
    """
    def __init__(self, config: BiBoConfig, layer_idx: int):
        super().__init__()
        self.mixer_type = config.residual_mixer_type
        self.kernel_size = config.residual_conv_kernel_size
        self.layer_idx = layer_idx

        if self.mixer_type == "none":
            self.kernel_logits = None
            self.dynamic_weight = None
            self.dynamic_bias = None
        else:
            previous_mass = 1.0 - config.residual_conv_init
            previous_prob = previous_mass / (self.kernel_size - 1)
            probs = torch.full((self.kernel_size,), previous_prob, dtype=torch.float32)
            probs[-1] = config.residual_conv_init
            if self.mixer_type == "causal_conv":
                self.kernel_logits = nn.Parameter(torch.log(probs))
                self.dynamic_weight = None
                self.dynamic_bias = None
            else:
                self.kernel_logits = None
                self.dynamic_weight = nn.Parameter(torch.zeros(self.kernel_size, config.hidden_size))
                self.dynamic_bias = nn.Parameter(torch.log(probs))

        self.register_buffer("last_current_weight", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_previous_mass", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_num_states", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(
        self,
        current_state: torch.Tensor,
        residual_history: Optional[Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        if self.mixer_type == "none":
            return current_state

        history = tuple(residual_history or ())
        states = history[-(self.kernel_size - 1):] + (current_state,)
        num_states = len(states)
        if self.mixer_type == "causal_conv":
            logits = self.kernel_logits[-num_states:]
            weights = torch.softmax(logits.float(), dim=0).to(current_state.dtype)
            stacked = torch.stack(states, dim=0)
            mixed = torch.sum(stacked * weights.view(num_states, 1, 1, 1), dim=0)
        else:
            logits = F.linear(
                current_state.to(self.dynamic_weight.dtype),
                self.dynamic_weight[-num_states:],
                self.dynamic_bias[-num_states:],
            )
            weights = torch.softmax(logits.float(), dim=-1).to(current_state.dtype)
            stacked = torch.stack(states, dim=2)
            mixed = torch.sum(stacked * weights.unsqueeze(-1), dim=2)

        if not torch.jit.is_scripting():
            with torch.no_grad():
                weights_float = weights.detach().float()
                self.last_current_weight.copy_(weights_float[..., -1].mean())
                self.last_previous_mass.copy_(weights_float[..., :-1].sum(dim=-1).mean())
                self.last_num_states.copy_(torch.tensor(num_states, device=self.last_num_states.device))

        return mixed

    def stats(self) -> dict:
        if self.mixer_type == "none":
            return {}
        return {
            f"layer_{self.layer_idx}/residual_conv/current_weight": float(self.last_current_weight.item()),
            f"layer_{self.layer_idx}/residual_conv/previous_mass": float(self.last_previous_mass.item()),
            f"layer_{self.layer_idx}/residual_conv/num_states": int(self.last_num_states.item()),
        }


class BiBoMultiStreamResidual(nn.Module):
    """
    mHC-style parallel residual streams with learned read/write gates.

    The model reads a gated mixture of streams before a layer, runs the normal
    decoder block, then writes the layer delta back into the streams. This keeps
    attention/MoE modules unchanged while giving each layer token-wise control
    over which residual lanes are useful.
    """
    def __init__(self, config: BiBoConfig, layer_idx: int):
        super().__init__()
        self.num_streams = config.residual_num_streams
        self.gate_type = config.residual_stream_gate_type
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if self.num_streams <= 1:
            self.read_weight = None
            self.read_bias = None
            self.write_weight = None
            self.write_bias = None
        else:
            read_probs = _stream_probs(self.num_streams, config.residual_stream_read_init)
            write_probs = _stream_probs(self.num_streams, config.residual_stream_write_init)
            if self.gate_type == "scalar":
                self.read_weight = None
                self.write_weight = None
            else:
                self.read_weight = nn.Parameter(torch.zeros(self.num_streams, self.hidden_size))
                self.write_weight = nn.Parameter(torch.zeros(self.num_streams, self.hidden_size))
            self.read_bias = nn.Parameter(torch.log(read_probs))
            self.write_bias = nn.Parameter(torch.tensor([_logit(float(p)) for p in write_probs]))

        self.register_buffer("last_read_main", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_read_entropy", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_write_main", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("last_write_aux", torch.tensor(float("nan")), persistent=False)

    def read(self, streams: torch.Tensor) -> torch.Tensor:
        if self.num_streams <= 1:
            return streams[:, :, 0, :]

        controller = streams.mean(dim=2)
        if self.gate_type == "scalar":
            logits = self.read_bias.view(1, 1, self.num_streams)
        else:
            logits = F.linear(controller.to(self.read_weight.dtype), self.read_weight, self.read_bias)
        weights = torch.softmax(logits.float(), dim=-1).to(streams.dtype)
        mixed = torch.sum(streams * weights.unsqueeze(-1), dim=2)

        if not torch.jit.is_scripting():
            with torch.no_grad():
                weights_float = weights.detach().float()
                entropy = -(weights_float * torch.log(weights_float.clamp_min(1e-8))).sum(dim=-1)
                self.last_read_main.copy_(weights_float[..., 0].mean())
                self.last_read_entropy.copy_(entropy.mean())

        return mixed

    def write(self, streams: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if self.num_streams <= 1:
            return streams + update.unsqueeze(2)

        if self.gate_type == "scalar":
            logits = self.write_bias.view(1, 1, self.num_streams)
        else:
            logits = F.linear(update.to(self.write_weight.dtype), self.write_weight, self.write_bias)
        gates = torch.sigmoid(logits).to(update.dtype)
        streams = streams + gates.unsqueeze(-1) * update.unsqueeze(2)

        if not torch.jit.is_scripting():
            with torch.no_grad():
                gates_float = gates.detach().float()
                self.last_write_main.copy_(gates_float[..., 0].mean())
                if self.num_streams > 1:
                    self.last_write_aux.copy_(gates_float[..., 1:].mean())
                else:
                    self.last_write_aux.copy_(torch.tensor(0.0, device=self.last_write_aux.device))

        return streams

    def stats(self) -> dict:
        if self.num_streams <= 1:
            return {}
        return {
            f"layer_{self.layer_idx}/streams/read_main": float(self.last_read_main.item()),
            f"layer_{self.layer_idx}/streams/read_entropy": float(self.last_read_entropy.item()),
            f"layer_{self.layer_idx}/streams/write_main": float(self.last_write_main.item()),
            f"layer_{self.layer_idx}/streams/write_aux": float(self.last_write_aux.item()),
        }


class BiBoDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    
    Args:
        config: Model config
        layer_idx: Layer index
    """
    def __init__(self, config: BiBoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Determine sliding window for this layer
        use_sliding_window = False
        if config.use_sliding_window and config.max_window_layers is not None:
            use_sliding_window = layer_idx < config.max_window_layers
        
        self.self_attn = BiBoAttention(
            config=config, 
            layer_idx=layer_idx,
            use_sliding_window=use_sliding_window
        )

        # MoE or dense MLP
        self.is_moe_layer = layer_idx not in config.mlp_only_layers
        if self.is_moe_layer:
            self.mlp = BiBoMoELayer(config)
        else:
            self.mlp = BiBoMLP(config, is_expert=False)

        self.input_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_residual_gate = BiBoResidualGate(config, "attn")
        self.mlp_residual_gate = BiBoResidualGate(config, "mlp")
        self.residual_mixer = BiBoCausalResidualConv(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        residual_history: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            position_embeddings: (cos, sin) from rotary embedding
            attention_mask: Optional mask
            past_key_value: KV cache
            cache_position: Cache position indices
            output_attentions: Return attn weights
            use_cache: Use KV cache
        
        Returns:
            (hidden_states, attn_weights, present_key_value)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        gate_input = hidden_states

        # Self attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self.attn_residual_gate(gate_input, attn_output)

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate_input = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_residual_gate(gate_input, hidden_states)
        hidden_states = self.residual_mixer(hidden_states, residual_history)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def residual_gate_stats(self) -> dict:
        if self.attn_residual_gate.gate_type == "none":
            return {}
        stats = {}
        for name, value in self.attn_residual_gate.stats().items():
            stats[f"layer_{self.layer_idx}/{name}"] = value
        for name, value in self.mlp_residual_gate.stats().items():
            stats[f"layer_{self.layer_idx}/{name}"] = value
        return stats

    def residual_mixer_stats(self) -> dict:
        return self.residual_mixer.stats()
