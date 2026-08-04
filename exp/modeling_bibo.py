"""Experimental BiBo model with Kimi K3 Block Attention Residuals.

The residual topology follows Moonshot AI's official Kimi K3 Hugging Face code:

* completed block representatives and the current within-block prefix sum are
  mixed with a learned, per-sublayer pseudo-query;
* attention and MLP reads have independent projection/RMSNorm parameters;
* block boundaries occur every ``attn_res_block_size`` decoder layers, including
  layer zero, where the token embedding becomes the first block representative;
* a final depth-wise mix is applied before the trunk's output RMSNorm; and
* RMS normalization, scores, softmax, and aggregation are computed in fp32.

Only the experimental package imports stable BiBo components. Nothing under
``src`` imports or probes ``exp``.
"""

import contextlib
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from src.modeling.attn import BiBoAttention
from src.modeling.embed import BiBoRotaryEmbedding
from src.modeling.ffn import BiBoMLP, BiBoMoELayer
from src.modeling.models import BiBoPreTrainedModel as _StableBiBoPreTrainedModel
from src.modeling.norm import BiBoRMSNorm

from .configuration_bibo import BiBoConfig

try:
    # Fused AR: one read of V, fp32 accumulation, and a custom backward that saves only the
    # INPUTS. Measured fwd+bwd at T=16384/H=512: 16-20x faster than this file's torch path and
    # 4.7x less peak memory at N=11 (2450 -> 520 MB per site). Parity and gradcheck live in
    # triton-kernel-fused/parity_check/parity_attn_res.py, graded against FP32 eager at every
    # dtype -- the bf16 kernel matches bf16 eager's error to the digit on all three gradients.
    from kernels.sm120.attn_res import attn_res as _fused_ar
    _HAS_FUSED_AR = True
except Exception:                                   # no triton / no kernels checkout
    _HAS_FUSED_AR = False

try:
    # Fused carry write: h = attn_read + c*attn_out + d*emb in ONE pass instead of one elementwise
    # pass per stream. Benched at the board shape (65536 x 512): 2.52x forward / 1.90x fwd+bwd at
    # two streams, and the gap widens with stream count. It is also MORE ACCURATE than this file's
    # torch path, not just faster: eager evaluates `_c.to(attn_output.dtype) * attn_output`, which
    # rounds the learned fp32 scalar to BF16 and multiplies in bf16, while the kernel promotes
    # every operand to fp32 and accumulates there -- measured 30,000-47,000x closer to fp64 truth
    # on the real dtype layout. So enabling it SHIFTS TRAINING NUMERICS (toward correct); an arm
    # run on it is not bit-comparable to one run without it.
    from kernels.sm120.residual_add import make_mlp_input as _fused_res_add
    _HAS_FUSED_RES_ADD = True
except Exception:
    _HAS_FUSED_RES_ADD = False

# attn_res_carry_scale -> the kernel's transform code for the attn_out multiplier. "none" has no
# theta at all (c is a hard 1.0) and is handled by the caller, not here.
_CARRY_MODE = {"unbounded": "none", "sigmoid": "2sigmoid", "tanh": "2tanh"}

logger = logging.get_logger(__name__)

__all__ = [
    "apply_attention_residual",
    "BiBoDecoderLayer",
    "BiBoPreTrainedModel",
    "BiBoModel",
    "BiBoForCausalLM",
]


def apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: BiBoRMSNorm,
) -> torch.Tensor:
    """Mix completed blocks and the current prefix across model depth.

    Args:
        prefix_sum: Current within-block state, shaped ``(tokens, hidden)``.
        block_residual: Completed block states, shaped
            ``(tokens, completed_blocks, hidden)``.
        projection: Learned pseudo-query represented as a bias-free ``hidden -> 1``
            projection.
        norm: Gain-bearing RMSNorm applied independently to every candidate key.

    Returns:
        The depth-attended state with shape ``(tokens, hidden)`` and the same
        dtype as ``prefix_sum``.

    This deliberately spells out RMSNorm and folds its gain into the projection,
    exactly as Kimi K3 does. The fp32 path is important for stable scores and
    aggregation under bf16 training.
    """
    if prefix_sum.ndim != 2:
        raise ValueError(
            f"prefix_sum must have shape (tokens, hidden), got {tuple(prefix_sum.shape)}"
        )
    if block_residual.ndim != 3:
        raise ValueError(
            "block_residual must have shape (tokens, blocks, hidden), got "
            f"{tuple(block_residual.shape)}"
        )
    if (
        block_residual.shape[0] != prefix_sum.shape[0]
        or block_residual.shape[2] != prefix_sum.shape[1]
    ):
        raise ValueError(
            "prefix_sum and block_residual disagree on token/hidden dimensions: "
            f"{tuple(prefix_sum.shape)} vs {tuple(block_residual.shape)}"
        )

    # norm(values) @ projection.weight, with both parameter vectors multiplied first so the
    # implementation matches Kimi K3's checkpoint semantics. Autograd splits the gradient back
    # to norm.weight and projection.weight through this product, so the kernel only needs the
    # folded vector.
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()

    if _HAS_FUSED_AR and prefix_sum.is_cuda:
        return _fused_ar(block_residual, prefix_sum, score_weight, norm.variance_epsilon)

    # AUTOCAST MUST BE OFF HERE. Under torch.autocast(bf16) `torch.matmul` is autocast-eligible,
    # so the two matmuls below get their fp32 inputs silently cast back to bf16 and the whole
    # "fp32 scores and aggregation" this function documents does not happen. K3's reference has
    # the same hole. The Triton kernel accumulates in true fp32, so without this the two paths
    # disagree by bf16-level error compounded over every layer -- 3.9e-01 in the hidden state,
    # measured by test_attn_res_gpu.
    _ac = (torch.autocast("cuda", enabled=False) if prefix_sum.is_cuda
           else contextlib.nullcontext())

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()

    # ALGEBRAICALLY IDENTICAL to `(rms_norm(values) * score_weight).sum(-1)`, but without ever
    # building a normalized copy. The RMS factor is per (token, block) and broadcasts over hidden,
    # so it pulls straight out of the contraction:
    #     sum_d (v_d * inv_rms) * w_d  ==  inv_rms * sum_d v_d * w_d
    # The naive form materialized TWO extra (tokens, blocks+1, hidden) fp32 tensors -- the
    # normalized keys and the keys*weight product -- and both were kept alive for backward, at
    # every residual site of every layer. This form contracts with a single GEMV straight to
    # (tokens, blocks+1) and allocates nothing of hidden size.
    with _ac:
        sq_sum = torch.linalg.vector_norm(values_float, dim=-1).square()
        inv_rms = torch.rsqrt(sq_sum / values_float.shape[-1] + norm.variance_epsilon)
        scores = torch.matmul(values_float, score_weight) * inv_rms
        probabilities = scores.softmax(dim=-1).unsqueeze(1)
        mixed = torch.matmul(probabilities, values_float).squeeze(1)
    return mixed.to(values.dtype)


class BiBoDecoderLayer(nn.Module):
    """One BiBo decoder layer with optional Kimi K3 Block AttnRes."""

    def __init__(self, config: BiBoConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.use_attn_residuals = config.attn_res_block_size is not None
        self.attn_res_block_size = config.attn_res_block_size

        self.self_attn = BiBoAttention(config=config, layer_idx=layer_idx)
        self.is_moe_layer = layer_idx not in config.mlp_only_layers
        self.mlp = BiBoMoELayer(config) if self.is_moe_layer else BiBoMLP(config, is_expert=False)

        self.input_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 2 = K3 faithful: an independent depth-mix before the attention sublayer AND before the
        # MLP sublayer. 1 = ONE mix per layer, at the layer input only; the MLP then takes an
        # ordinary PreNorm residual. Halves the depth-attention work and the AttnRes parameters.
        self.attn_res_sites = getattr(config, "attn_res_sites", 2)
        # What the MLP reads when sites==1. False = the raw within-block prefix sum (MEASURED,
        # LOST: +0.00815 bpb -- it removes depth-mixing from the MLP entirely rather than halving
        # it). True = Ht + A, the layer's own site-1 mix plus this layer's attention output, so the
        # MLP keeps a depth-mixed base one sublayer stale AND still sees attention. Default False
        # so a config written before this flag existed rebuilds to what it actually ran.
        self.attn_res_carry = getattr(config, "attn_res_carry", False)
        # The block-boundary reset (prefix_sum = None -> prefix_sum = attn_output) drops the fp32
        # embedding out of the residual stream, so under bf16 autocast an AttnRes model runs a
        # BF16 stream while the standard-residual control runs FP32 (the embedding is fp32 and
        # `fp32 + bf16` promotes, forever). That is worth ~2-3% throughput and 0.7 GB and it made
        # every AttnRes arm non-comparable to the baseline. Setting this keeps the stream in the
        # layer-input dtype so the ONLY difference from the control is AttnRes itself.
        self.attn_res_fp32_stream = getattr(config, "attn_res_fp32_stream", False)
        # CARRY SCALE. In carry the MLP reads Ht + A, so A enters at coefficient exactly 1,
        # deliberately OUTSIDE the softmax simplex -- that is why carry beats sites=2, where every
        # unit of probability spent on a previous block comes straight out of this layer's
        # attention, and where simply having more blocks flattens the distribution and shrinks
        # p_last for reasons that have nothing to do with what the MLP wants.
        #
        # 1 is still an arbitrary coefficient. Modes, ALL initialised to exactly s = 1.0 so each
        # is a strict generalization of plain carry and the three are mutually comparable:
        #
        #   "unbounded"  s = theta,           init 1.0     range R    <- DIAGNOSTIC, run first
        #   "sigmoid"    s = 2*sigmoid(theta) init th=0    range (0,2)
        #   "tanh"       s = 2*tanh(theta)    init th=atanh(.5)  range (-2,2)
        #
        # Unbounded goes first ON PURPOSE and is not meant to ship: it is the only variant that
        # can tell us WHERE the model wants s to sit, which is what picks the bound. If min/max
        # settle inside (0,2) then sigmoid is the right cage; if they run past it, a cage at 2
        # would be clipping the answer and the question changes. Every unbounded scale we have
        # tried has eventually run away, so this is a measurement, not a candidate.
        _cs_mode = getattr(config, "attn_res_carry_scale", "none")
        _cs_mode = "none" if _cs_mode in (False, None) else str(_cs_mode)
        self.attn_res_carry_scale = _cs_mode
        _init = {"unbounded": 1.0, "sigmoid": 0.0, "tanh": math.atanh(0.5)}.get(_cs_mode)
        self.attn_res_carry_theta = (
            nn.Parameter(torch.full((1,), float(_init)))
            if (_init is not None and self.use_attn_residuals and self.attn_res_carry) else None)
        # d: off-simplex embedding gain on the carry write. Init EXACTLY 0 so the arm is a strict
        # generalization -- at step 0 it is bit-identical to plain carry, and any nonzero final
        # value is the model asking for it. Not created at layer 0: there attn_read IS the
        # embedding (block_residual is empty, so the depth read is skipped), so d would be a
        # duplicate of the identity path. Unnormed on purpose -- see attn_res_emb_term in the
        # config, and the MoE-output-norm round, which measured "norm the addend" at -0.010 bpb.
        # attn_res_emb_scale picks d = f(theta). "none" keeps the raw scalar, which is what the
        # first emb arm ran and what produced the measured profile d = [1.32, 0.55, 0.40, 0.33,
        # 0.33, 0.49, 0.28, 0.42, 0.30]. NOTE the range: plain "sigmoid" caps d at 1.0 and would
        # CLIP layer 1, which asked for 1.32 -- the single largest and most informative value on
        # that axis. "2sigmoid" spans (0, 2) and covers it.
        _es = str(getattr(config, "attn_res_emb_scale", "none"))
        if _es not in ("none", "sigmoid", "2sigmoid", "tanh", "2tanh"):
            raise ValueError(f"attn_res_emb_scale must be none/sigmoid/2sigmoid/tanh/2tanh, got {_es!r}")
        self.attn_res_emb_scale = _es
        # site="ht": d is a RADIAL gain on the depth-mix output, emb * r^(p-1) with r = rms(emb)
        # per token and p = sigmoid(theta) in (0,1). p=1 is the raw embedding (rms 0.04), p=0 is
        # unit-norm (rms 1). Same trick as radial_theta on the activation, and it exists because the
        # raw embedding is 500-10000x smaller than everything it is added to, so an unnormalised
        # skip cannot matter. Init theta=-4 -> p=0.018 -> rms ~1, i.e. NEAR UNIT NORM.
        # site="mlp" is the retired path: d*emb added to the carry write, theta init 0, no norm.
        _es = str(getattr(config, "attn_res_emb_site", "mlp"))
        if _es not in ("mlp", "ht"):
            raise ValueError(f"attn_res_emb_site must be mlp or ht, got {_es!r}")
        self.attn_res_emb_site = _es
        self.attn_res_emb_eps = config.rms_norm_eps
        _need_carry = self.attn_res_carry or _es == "ht"
        _emb_on = (getattr(config, "attn_res_emb_term", False) and self.use_attn_residuals
                   and _need_carry and layer_idx > 0)
        # attn_res_emb_gain REPLACES the radial exponent with a flat gain on the RAW embedding:
        # i * emb, ONE scalar, no norm, and theta is not created at all. Two reasons the norm is
        # gone. (1) The radial version measured itself out of existence -- every layer drove theta
        # from -4 down to [-4.55, -5.55] over 1250 steps, and all that travel moved the real
        # magnitude r^p from 0.944 to 0.98, a 4% range; p was asymptoting to r^p == 1, i.e. to a
        # plain unit-norm skip, so the radial machinery had already collapsed to a norm. (2) That
        # arm LOST (0.6672 vs 0.6579 c-only) while the one emb arm that ever won, mlp-site d*emb
        # at 0.6572, was unnormed -- and the MoE-output-norm round independently measured
        # "RMS-norm the addend" at -0.010 bpb in all 3 variants. So the norm is the suspect.
        # Init EXACTLY 0 so the arm is bit-identical to plain carry at step 0 and is a strict
        # generalization -- same discipline as c and d. dL/di is nonzero there (gate 8 asserts it),
        # so it can leave.
        _gain_on = bool(getattr(config, "attn_res_emb_gain", False)) and _es == "ht"
        self.attn_res_emb_theta = (
            nn.Parameter(torch.full((1,), -4.0 if _es == "ht" else 0.0))
            if (_emb_on and not _gain_on) else None)
        self.attn_res_emb_gain = nn.Parameter(torch.zeros(1)) if (_emb_on and _gain_on) else None
        # Non-persistent so it never enters the state_dict: an extra key would break every existing
        # checkpoint load and the exp(control) == src equality that gates this whole family.
        if self.use_attn_residuals and self.attn_res_carry:
            self.register_buffer("attn_res_carry_one", torch.ones(1), persistent=False)

        if self.use_attn_residuals:
            self.self_attention_res_norm = BiBoRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            if self.attn_res_sites == 2:
                self.mlp_res_norm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

        self.use_selective_checkpointing = False

    def _standard_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def _attn_res_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp(hidden_states)

    def _forward_standard_residuals(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        output_attentions: bool,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = residual + attn_output

        if self.use_selective_checkpointing and self.training:
            hidden_states = checkpoint(
                self._standard_ffn_forward, hidden_states, use_reentrant=False
            )
        else:
            hidden_states = self._standard_ffn_forward(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    def _forward_attention_residuals(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        output_attentions: bool,
        block_residual: Optional[torch.Tensor],
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape
        _stream_dtype = hidden_states.dtype      # what the control's stream would be
        prefix_sum = hidden_states

        if block_residual is None:
            block_residual = hidden_states.new_zeros(
                batch_size * seq_len, 0, hidden_size
            )
        if block_residual.shape[1] > 0:
            hidden_states = apply_attention_residual(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).reshape(batch_size, seq_len, hidden_size)
            if self.attn_res_emb_gain is not None:
                # HT = AR(...) + i * emb, the RAW embedding. No norm of any kind.
                # Both normed ht arms lost (radial 0.6672, and the radial is what unit-norm
                # collapses to) while the only emb arm that ever won -- mlp-site d*emb at
                # 0.6572 -- was unnormed, so the NORM is the suspect, not the site. The
                # MoE-output-norm round says the same thing from the other direction: RMS-norming
                # an addend into the residual stream cost -0.010 bpb in all 3 variants.
                # Scale note: that winning arm peaked at d = 1.32, i.e. it injected rms ~0.05,
                # not ~1. Loud was plausibly the bug, so raw is the regime that already works.
                # Same fused kernel as the carry -- this is exactly a second stream on the add.
                _emb = block_residual[:, 0].reshape(batch_size, seq_len, hidden_size)
                if _HAS_FUSED_RES_ADD and hidden_states.is_cuda:
                    hidden_states = _fused_res_add(
                        hidden_states, self.attn_res_emb_gain, _emb, modes=("none",))
                else:
                    # must round exactly where the kernel does: scalar to the stream dtype,
                    # product in the stream dtype, accumulate in the output dtype
                    _g = self.attn_res_emb_gain.float().to(_emb.dtype)
                    hidden_states = hidden_states + (_g * _emb).to(hidden_states.dtype)
            elif self.attn_res_emb_theta is not None and self.attn_res_emb_site == "ht":
                # retired: radial, emb * r^(p-1) with p = sigmoid(theta). Kept so the arms
                # already on disk still load. It measured itself down to r^p -> 1, i.e. to a
                # plain unit-norm skip, and lost.
                _emb = block_residual[:, 0].reshape(batch_size, seq_len, hidden_size).float()
                _ms = _emb.square().mean(-1, keepdim=True).add(self.attn_res_emb_eps)
                _p = torch.sigmoid(self.attn_res_emb_theta.float())
                hidden_states = hidden_states + (
                    _emb * _ms.sqrt().pow(_p - 1.0)).to(hidden_states.dtype)
        # K3's site-1 read, unchanged. Kept for the carry variant: the mix computed at the END of
        # layer l-1 from (S, B) is the SAME tensor as this one -- S and B do not change across the
        # layer boundary -- so "defer the mix to the end of the layer" and "keep K3's site-1 mix"
        # are the same computation, differing only in which layer owns the parameters.
        attn_read = hidden_states

        # K3 measures block size in decoder layers. Layer zero is a boundary:
        # the embedding is stored, and the new partial block starts at attn_out.
        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.reshape(-1, hidden_size).unsqueeze(1)),
                dim=1,
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        attn_output, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        if prefix_sum is None:
            # boundary layer: the stream restarts from attn_output alone
            prefix_sum = (attn_output.to(_stream_dtype) if self.attn_res_fp32_stream
                          else attn_output)
        else:
            prefix_sum = prefix_sum + attn_output

        if self.attn_res_sites == 2:
            hidden_states = apply_attention_residual(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
            ).reshape(batch_size, seq_len, hidden_size)
        elif self.attn_res_carry:
            # ONE mix per layer, CARRY: the MLP reads the site-1 mix plus this layer's attention
            # output. Depth-mixed (one sublayer stale) and attention IS visible, which the plain
            # prefix-sum variant below is not -- that one deleted depth-mixing from the MLP rather
            # than halving it, and cost +0.00815 bpb for it.
            # block_residual[:, 0] IS the embedding, always: layer 0 is a block boundary for every
            # block size (0 % n == 0), so the untransformed embedding is what gets archived there.
            # Reusing it costs no new plumbing through the layer signature.
            _has_emb = (self.attn_res_emb_theta is not None and block_residual.shape[1] > 0
                    and self.attn_res_emb_site == "mlp")
            _emb = (block_residual[:, 0].reshape(batch_size, seq_len, hidden_size)
                    if _has_emb else None)
            if _HAS_FUSED_RES_ADD and attn_read.is_cuda:
                # attn_res_carry_one is a non-persistent ones buffer, so carry_scale="none"
                # (a hard c = 1.0) goes down the same fused path instead of forking the formula.
                _m = self.attn_res_carry_theta
                _pairs = [_m if _m is not None else self.attn_res_carry_one, attn_output]
                _modes = [_CARRY_MODE[self.attn_res_carry_scale] if _m is not None else "none"]
                if _has_emb:
                    _pairs += [self.attn_res_emb_theta, _emb]
                    _modes.append(self.attn_res_emb_scale)
                hidden_states = _fused_res_add(attn_read, *_pairs, modes=tuple(_modes))
            else:
                if self.attn_res_carry_theta is not None:
                    _t = self.attn_res_carry_theta.float()
                    _c = (_t if self.attn_res_carry_scale == "unbounded"
                          else 2.0 * torch.sigmoid(_t) if self.attn_res_carry_scale == "sigmoid"
                          else 2.0 * torch.tanh(_t))
                    hidden_states = attn_read + _c.to(attn_output.dtype) * attn_output
                else:
                    hidden_states = attn_read + attn_output
                if _has_emb:
                    _d = self.attn_res_emb_theta.float()
                    _d = ({"none": lambda x: x, "sigmoid": torch.sigmoid,
                           "2sigmoid": lambda x: 2.0 * torch.sigmoid(x),
                           "tanh": torch.tanh, "2tanh": lambda x: 2.0 * torch.tanh(x)}
                          [self.attn_res_emb_scale](_d))
                    hidden_states = hidden_states + (
                        _d.to(hidden_states.dtype) * _emb.to(hidden_states.dtype))
        else:
            # ONE mix per layer, PREFIX: the MLP reads the raw within-block sum. Measured and lost.
            hidden_states = prefix_sum

        if self.use_selective_checkpointing and self.training:
            mlp_output = checkpoint(
                self._attn_res_mlp_forward, hidden_states, use_reentrant=False
            )
        else:
            mlp_output = self._attn_res_mlp_forward(hidden_states)
        prefix_sum = prefix_sum + mlp_output

        outputs = (prefix_sum, block_residual)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        block_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del use_cache, kwargs
        if not self.use_attn_residuals:
            return self._forward_standard_residuals(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value,
                cache_position,
                bool(output_attentions),
            )
        return self._forward_attention_residuals(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
            bool(output_attentions),
            block_residual,
        )


class BiBoPreTrainedModel(_StableBiBoPreTrainedModel):
    config_class = BiBoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BiBoDecoderLayer"]


class BiBoModel(BiBoPreTrainedModel):
    """Experimental BiBo transformer trunk."""

    def __init__(self, config: BiBoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_attn_residuals = config.attn_res_block_size is not None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BiBoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = (
            BiBoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, "exp_post_embed_norm", False)
            else None
        )
        self.rotary_emb = BiBoRotaryEmbedding(
            config.rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_type=config.rope_scaling["type"],
            scaling_factor=config.rope_scaling.get("factor", 1.0),
        )
        if self.use_attn_residuals:
            self.output_attn_res_norm = BiBoRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _apply_output_attention_residual(
        self, hidden_states: torch.Tensor, block_residual: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        return apply_attention_residual(
            hidden_states.reshape(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).reshape(batch_size, seq_len, hidden_size)

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
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        del kwargs
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` incompatible with gradient checkpointing. "
                "Setting `use_cache=False`"
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_length,
                device=inputs_embeds.device,
            )

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 2D (batch, seq) padding mask or None, "
                    f"got {attention_mask.dim()}D"
                )
            if bool(attention_mask.all()):
                attention_mask = None

        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_(min=0)
                position_ids = position_ids[:, -seq_length:]
            else:
                position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        if self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)

        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
            seq_len=past_seen_tokens + seq_length,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        block_residual = None
        if self.use_attn_residuals:
            block_residual = hidden_states.new_zeros(
                batch_size * seq_length, 0, self.config.hidden_size
            )

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if self.use_attn_residuals:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        None,
                        cache_position,
                        output_attentions,
                        False,
                        block_residual,
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        None,
                        cache_position,
                        output_attentions,
                        False,
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    block_residual=block_residual,
                )

            hidden_states = layer_outputs[0]
            if self.use_attn_residuals:
                block_residual = layer_outputs[1]
                if output_attentions:
                    all_self_attns += (layer_outputs[2],)
            elif output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.use_attn_residuals:
            hidden_states = self._apply_output_attention_residual(
                hidden_states, block_residual
            )
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                value
                for value in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if value is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BiBoForCausalLM(BiBoPreTrainedModel, GenerationMixin):
    """Experimental BiBo causal LM with Kimi K3 Block AttnRes."""

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

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def enable_selective_gradient_checkpointing(self):
        for layer in self.model.layers:
            layer.use_selective_checkpointing = True

    def disable_selective_gradient_checkpointing(self):
        for layer in self.model.layers:
            layer.use_selective_checkpointing = False

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
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else True

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
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

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
