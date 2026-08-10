"""BiBo model classes"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers import initialization as hf_init
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

from src.configuration_bibo import BiBoConfig
from .norm import BiBoRMSNorm
from .embed import BiBoRotaryEmbedding, DualRotaryEmbedding
from .layers import BiBoDecoderLayer

logger = logging.get_logger(__name__)

__all__ = ['BiBoPreTrainedModel', 'BiBoModel', 'BiBoForCausalLM']


class BiBoPreTrainedModel(PreTrainedModel):
    config_class = BiBoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiBoDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        """MUST write through `transformers.initialization` (`hf_init.*`), never `param.data.foo_()`.

        Those helpers no-op on any tensor already flagged `_is_hf_initialized`, which is how
        transformers >=5 protects checkpoint weights: `from_pretrained` loads, marks the loaded
        params, then walks every module calling `_init_weights`. Raw in-place `.data` writes ignore
        the flag and silently re-randomize everything that was just loaded (was doing exactly that
        to all 24 nn.Linear + the Embedding — see the Jul 26 2026 AGENTS.md entry).
        """
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):   # Conv1d = conv shared expert
            hf_init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                hf_init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            hf_init.normal_(module.weight, mean=0.0, std=std)
            # slicing drops the flag, so check it explicitly before zeroing the pad row
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                hf_init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, BiBoRMSNorm):
            hf_init.ones_(module.weight)                 # Qwen3 parity; needed on meta-init
        elif isinstance(module, BiBoRotaryEmbedding):
            # inv_freq/original_inv_freq are persistent=False, so they are NOT in the checkpoint and
            # from_pretrained's lazy/meta path never runs __init__'s computation for them. Without
            # this branch they come back as uninitialized memory — and since dynamic-NTK returns
            # original_inv_freq in-window, a zeroed buffer means cos=1/sin=0, i.e. RoPE silently
            # becomes the identity on every loaded checkpoint. (Stock Llama does the same thing.)
            freqs = module._compute_inv_freq(module.base, module.inv_freq.device)
            hf_init.copy_(module.inv_freq, freqs)
            hf_init.copy_(module.original_inv_freq, freqs)


class BiBoModel(BiBoPreTrainedModel):
    """BiBo transformer trunk."""
    def __init__(self, config: BiBoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BiBoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # EXPERIMENTAL: BLOOM-style embedding norm. Built only when enabled, so the param count is
        # unchanged otherwise. self.norm (pre-LM-head) is always present either way.
        self.embed_norm = (BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                           if getattr(config, "exp_post_embed_norm", False) else None)
        # rope_dim, NOT head_dim — cos/sin cover only the rotated slice of each head.
        # Dual: full-attention layers and sliding-window layers get their own base and rotary
        # width. Collapses to a single module when the two configs match, so the default costs
        # nothing extra. Returns (global_pair, local_pair); attention indexes it with is_swa.
        self.rotary_emb = DualRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,  # tolerate extra model inputs injected by GenerationMixin across HF 5.x versions
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` incompatible with gradient checkpointing. Setting `use_cache=False`")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # config-aware so SWA layers (config.layer_types) get window-evicting sliding layers and
        # hold O(sliding_window) KV during decode instead of O(total_len).
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # host int, no GPU sync — also feeds the rotary extent below
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )

        # A 2D (B, K_total) mask (1=real, 0=pad) is threaded to every layer, which folds it into its
        # own causal/band mask. None keeps the SDPA is_causal fast path (backend does the skip).
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"attention_mask must be a 2D (batch, seq) padding mask or None, "
                    f"got {attention_mask.dim()}D"
                )
            if bool(attention_mask.all()):
                attention_mask = None   # all-ones (generate's default) -> keep the fast path

        if position_ids is None:
            if attention_mask is not None:
                # Mask-aware positions: pads don't advance the position counter (left-pad safe).
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_(min=0)
                position_ids = position_ids[:, -seq_length:]
            else:
                position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        if self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)
        # BF16 RESIDUAL STREAM. Without this the stream is fp32 for a subtle reason: weights
        # are fp32 master and nn.Embedding is NOT autocast, so inputs_embeds arrives fp32 and
        # every `residual + attn_out` promotes the bf16 sublayer output back up. One cast here
        # keeps the whole stream bf16, which is what modded-nanogpt does. Master weights and
        # optimizer state are untouched.
        if getattr(self.config, 'bf16_residual_stream', False):
            hidden_states = hidden_states.to(torch.bfloat16)

        # seq_len as a host int keeps the dynamic-NTK path free of a per-step GPU sync / graph break.
        position_embeddings = self.rotary_emb(
            hidden_states, position_ids, seq_len=past_seen_tokens + seq_length)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,             # 2D padding mask or None (None -> is_causal path)
                    None,
                    cache_position,
                    output_attentions,
                    False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,   # 2D padding mask or None (None -> is_causal path)
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # layer_outputs: (hidden_states, [attn_weights]) — the cache is mutated in place
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None   # Cache object, returned as-is

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class BiBoForCausalLM(BiBoPreTrainedModel, GenerationMixin):
    """BiBo causal LM."""
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BiBoConfig):
        super().__init__(config)
        self.model = BiBoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def enable_selective_gradient_checkpointing(self):
        """Enable selective gradient checkpointing (checkpoint MoE/MLP only, not attention)"""
        for layer in self.model.layers:
            layer.use_selective_checkpointing = True

    def disable_selective_gradient_checkpointing(self):
        """Disable selective gradient checkpointing"""
        for layer in self.model.layers:
            layer.use_selective_checkpointing = False

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,  # tolerate extra model inputs injected by GenerationMixin across HF 5.x versions
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else True

        # Decoder output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        loss = None
        # Standard CE — one big F.linear + F.cross_entropy.
        if labels is not None:
            logits = self.lm_head(hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
