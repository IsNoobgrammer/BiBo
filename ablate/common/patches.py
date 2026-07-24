"""Swappable kernel patches for BOTH arms (BiBo-min and Qwen3MoE), applied in place.

Components (each patches the corresponding class/function on both models):
  'liger_norm' : LigerRMSNorm  -> BiBoRMSNorm + Qwen3MoeRMSNorm  (llama casting, offset 0 == eager)
  'liger_rope' : Liger RoPE    -> apply_rotary_pos_emb in bibo attn.base + qwen modeling
  'moe'        : tkf fused moe  -> BiBoFusedExperts + Qwen3MoeExperts (act codes differ per arm)

Cross-entropy is NOT patched here: the training loop calls fused_linear_cross_entropy on the base
model's (hidden, lm_head.weight) directly, so CE is a swappable training-loop component, not a
model monkeypatch. Call apply(components) once before training.
"""
from . import _paths  # noqa: F401
import torch
import torch.nn.functional as F

try:
    _nc = torch.compiler.disable
except AttributeError:
    _nc = torch._dynamo.disable

# PolyGLU act-cycle override (codes: 0=silu, 1=relu2, 2=normsilu, 5=situ, 6=normrelu2, 7=normsitu). None -> default (0,1,2) cycle.
# train.py sets this from --silu/--relu2/--normsilu/--situ, e.g. [0] = all-SiLU experts, [0,1] = silu/relu2 alternating.
ACT_CYCLE = None

# Router GATE override (the fn turning router logits into per-expert scores). 'sigmoid' = shipped
# DeepSeek-V3 behavior; 'situ' = tanh(g)*sigmoid(g). train.py sets this from --router_gate.
# CAUTION (situ): matches sigmoid's shape for g>0 but is NEGATIVE and NON-MONOTONIC for g<0
# (min ~-0.205 at g~-1.2). Two consequences vs sigmoid: (a) top-k selection stops being monotonic
# in the logit -- a very negative logit scores HIGHER (closer to 0) than a mildly negative one;
# (b) the shipped sum(w) weight norm no longer sums to 1 (see _norm_topk) -- pair a signed gate
# with ROUTER_NORM='softmax'. Ablation-only knob; keep 'sigmoid' until an A/B says otherwise.
ROUTER_GATE = "sigmoid"


ROUTER_NORM = "sum"


def _gate_scores(router_logits, gate_type):
    """Router logits -> per-expert scores. Mirrors router.py Step 4 + the 'situ' ablation arm."""
    if ROUTER_GATE == "situ":
        return torch.tanh(router_logits) * torch.sigmoid(router_logits)
    if gate_type == "sigmoid":
        return torch.sigmoid(router_logits)
    return torch.softmax(router_logits, dim=1)


def _norm_topk(top_k_weights):
    """Top-k SCORES -> combine WEIGHTS. The whole point of this step is sum(weights) == 1.

      'sum'     shipped: w / sum(w). Sums to 1 ONLY if every w >= 0 (true for sigmoid). For a
                SIGNED gate it breaks: sum(w) can cancel toward 0 (weights explode) or go
                negative (every weight flips sign).
      'softmax' softmax over the selected top-k scores: all-positive and sums to EXACTLY 1 for
                ANY real scores, so it is the correct partner for a signed gate (situ). Cost:
                it compresses dynamic range -- situ scores live in [-0.21, 1), so a top-2 gap of
                at most ~1.2 logits maps to weights within ~[0.23, 0.77] (flatter routing).
      'l1'      w / sum|w|. Bounds each weight to [-1,1] and cannot explode, but does NOT sum to
                1 for signed w (e.g. [0.8,-0.2] -> sums to 0.6). Kept only as an ablation arm.
    """
    if ROUTER_NORM == "softmax":
        return torch.softmax(top_k_weights, dim=-1)
    if ROUTER_NORM == "l1":
        return top_k_weights / (top_k_weights.abs().sum(-1, keepdim=True) + 1e-20)
    return top_k_weights / (top_k_weights.sum(-1, keepdim=True) + 1e-20)


def add_situ_params(model):
    """Learnable SiTU: register per-expert (alpha, gamma) so code-5 experts compute
    gamma*tanh(alpha*g)*sigmoid(g) instead of the parameter-free tanh(g)*sigmoid(g).
    Two 1D (E,) params (not one (E,2)) so build_optimizers' ndim>=2 rule sends them to AdamW,
    and the bf16-ckpt rule keeps them fp32. Call AFTER build_arm, BEFORE build_optimizers."""
    import torch.nn as nn
    from src.modeling.ffn.moe import BiBoFusedExperts
    n = 0
    for m in model.modules():
        if isinstance(m, BiBoFusedExperts):
            E = m.zero_end   # rows must match the codes tensor length (polyglu + specials)
            dev = m.gate_up_proj.device
            m.situ_alpha = nn.Parameter(torch.ones(E, device=dev))
            m.situ_gamma = nn.Parameter(torch.ones(E, device=dev))
            n += 1
    return n


# ───────────────────────── liger norm ─────────────────────────
def patch_liger_norm():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction

    def _liger_rms(self, hidden_states):
        return LigerRMSNormFunction.apply(hidden_states, self.weight, self.variance_epsilon,
                                          0.0, "llama", False)
    from src.modeling.norm import BiBoRMSNorm
    from baseline.qwen3moe.modeling import Qwen3MoeRMSNorm
    BiBoRMSNorm.forward = _nc(_liger_rms)
    Qwen3MoeRMSNorm.forward = _nc(_liger_rms)


# ───────────────────────── liger rope ─────────────────────────
def patch_liger_rope():
    from liger_kernel.ops.rope import LigerRopeFunction

    def _liger_rope(q, k, cos, sin, unsqueeze_dim=1):
        # cos[:1]: Liger indexes by sequence position (batch-shared); valid for unpadded packed data.
        return LigerRopeFunction.apply(q, k, cos[:1], sin[:1], None, unsqueeze_dim)
    import src.modeling.attn.base as bibo_attn_base
    import baseline.qwen3moe.modeling as qwen_mod
    bibo_attn_base.apply_rotary_pos_emb = _nc(_liger_rope)   # bibo calls it on the rope_dim slice (partial)
    qwen_mod.apply_rotary_pos_emb = _nc(_liger_rope)          # qwen calls it on full head_dim


# ───────────────────────── fused MoE ─────────────────────────
def patch_fused_moe():
    # FORCE per-expert: measured 2.31x faster than grouped on Blackwell at our expert size (H=512, I=768)
    # -- grouped's tl.dot only wins for large experts -- AND per-expert is the only path that handles the
    # Identity/Zero special experts correctly. (moe() auto-dispatch would wrongly pick grouped at >=4096 tok.)
    from kernels.sm120.moe import moe_per_expert as moe_fused

    # BiBo: diverse PolyGLU activations (silu/relu2/normsilu cycled) + optional Identity/Zero specials
    def _bibo_moe(self, hidden_states, top_k_indices, top_k_weights):
        codes = getattr(self, "_act_codes", None)
        if codes is None or codes.device != hidden_states.device:
            cyc = ACT_CYCLE or (0, 1, 2)
            lst = ([cyc[e % len(cyc)] for e in range(self.num_polyglu_experts)]
                   + [3] * (self.identity_end - self.identity_start)
                   + [4] * (self.zero_end - self.zero_start))
            codes = torch.tensor(lst, dtype=torch.int32, device=hidden_states.device)
            self._act_codes = codes
        ap = (torch.stack([self.situ_alpha, self.situ_gamma], dim=1)
              if getattr(self, "situ_alpha", None) is not None else None)
        return moe_fused(hidden_states, top_k_indices, top_k_weights,
                         self.gate_up_proj, self.down_proj, codes, act_params=ap)

    # Qwen: homogeneous SiLU (act code 0) for every expert
    def _qwen_moe(self, hidden_states, top_k_index, top_k_weights):
        codes = getattr(self, "_act_codes", None)
        if codes is None or codes.device != hidden_states.device:
            codes = torch.zeros(self.num_experts, dtype=torch.int32, device=hidden_states.device)
            self._act_codes = codes
        return moe_fused(hidden_states, top_k_index, top_k_weights,
                         self.gate_up_proj, self.down_proj, codes)

    from src.modeling.ffn.moe import BiBoFusedExperts
    from baseline.qwen3moe.modeling import Qwen3MoeExperts
    BiBoFusedExperts.forward = _nc(_bibo_moe)
    Qwen3MoeExperts.forward = _nc(_qwen_moe)


# ───────────────────────── router gate ablation (BiBo only) ─────────────────────────
def patch_router_gate():
    """Swap the router's score fn (Step 4) for ROUTER_GATE, leaving every other step verbatim.
    Body mirrors src/modeling/ffn/router.py forward EXACTLY except the one gating line, so the
    bias/selection/unbiased-gather/norm/scaling semantics stay bit-identical on the sigmoid path."""
    from einops import rearrange
    import src.modeling.ffn.router as rmod
    R = rmod.BiBoMoERouter

    def _fwd(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        if self.router_type == "mlp":
            flat_hidden = rearrange(hidden_states, 'b s h -> (b s) h')
            router_logits = self.gate_proj(flat_hidden).float()
        else:
            x_perm = rearrange(hidden_states, 'b s h -> b h s')
            x_padded = F.pad(x_perm, (self.causal_padding, 0))
            router_logits = rearrange(self.gate_conv(x_padded), 'b e s -> (b s) e').float()
        router_logits = self._apply_router_activation(router_logits)
        scores = _gate_scores(router_logits, self.gate_type)          # <<< the ONLY change
        selection_scores = scores + self.bias                          # bias: SELECTION only
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False)
        top_k_weights = scores.gather(-1, top_k_indices)               # UNBIASED weights
        if self.top_k > 1 and self.norm_topk_prob:
            norm_weights = _norm_topk(top_k_weights)   # ROUTER_NORM: sum (shipped) | softmax | l1
        else:
            norm_weights = top_k_weights
        norm_weights = norm_weights * self.routed_scaling_factor
        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.float()

    R.forward = _nc(_fwd)


# ───────────────────────── fused conv router (BiBo only) ─────────────────────────
def patch_conv_router():
    """Route BiBo's CONV router through the sm120 fused-Triton conv kernel (no cuDNN).
    Only fires for router_type='conv' + gate_type='sigmoid' + router_activation='none' (the kernel's
    hardcoded pipeline: causal conv -> sigmoid -> +bias select -> top-k -> unbiased gather -> norm/scale).
    Any other config (mlp router, softmax gate, relu/silu activation) falls through to the eager forward,
    so this is safe to apply unconditionally (no-op for the qwen arm / mlp router)."""
    from kernels.sm120.router import fused_router
    import src.modeling.ffn.router as rmod
    R = rmod.BiBoMoERouter
    _orig = getattr(R, "_orig_forward", None) or R.forward
    R._orig_forward = _orig

    def _fwd(self, hidden_states):
        # ROUTER_GATE guard: the kernel hardcodes sigmoid, so a non-sigmoid gate MUST fall through
        # to the eager forward (else it would silently compute the wrong gate).
        if (self.router_type == "conv" and self.gate_type == "sigmoid"
                and self.router_activation == "none" and ROUTER_GATE == "sigmoid"):
            # cast fp32 master conv weight to the (bf16 under autocast) input dtype so the kernel's
            # tl.dot sees matching operands; the .to() is differentiable -> grad flows back to fp32 weight
            w_conv = self.gate_conv.weight.to(hidden_states.dtype)
            idx, w = fused_router(hidden_states, w_conv, self.bias,
                                  self.top_k, self.num_routed_experts,
                                  norm_topk_prob=self.norm_topk_prob,
                                  routed_scaling_factor=self.routed_scaling_factor)
            return idx.long(), w.float()
        return _orig(self, hidden_states)
    R.forward = _nc(_fwd)


# ───────────────────────── FlashAttention (both arms) ─────────────────────────
def flash_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def resolve_attn(impl):
    """Downgrade a flash impl to 'sdpa' when flash_attn isn't importable (local / T4). Returns effective impl."""
    if impl and impl.startswith("flash") and not flash_available():
        print(f"[attn] {impl} requested but flash_attn unavailable -> falling back to sdpa", flush=True)
        return "sdpa"
    return impl or "sdpa"


def patch_bibo_flash():
    """Route BiBo's global-attention HOT PATH (training: no mask/sink/padding) through flash_attn_func.
    Any failure (flash missing, fp32, mask needed) falls back to the original SDPA path -> safe everywhere.
    Qwen gets flash via config._attn_implementation instead (native HF dispatch)."""
    import src.modeling.attn.base as base
    _orig = getattr(base, "_orig_full_attention", None) or base.full_attention
    base._orig_full_attention = _orig

    def _wrapped(query, key, value, sinks, *, num_key_value_groups, scaling,
                 padding_mask=None, dropout=0.0, training=False, output_attentions=False):
        q_len, kv_len = query.shape[-2], key.shape[-2]
        need_mask = (output_attentions or sinks is not None or padding_mask is not None
                     or (q_len > 1 and kv_len > q_len))
        if not need_mask:
            try:
                from flash_attn import flash_attn_func
                q = query.transpose(1, 2).contiguous()      # (B,S,H,d) — flash layout; GQA broadcast in-kernel
                k = key.transpose(1, 2).contiguous()
                v = value.transpose(1, 2).contiguous()
                o = flash_attn_func(q, k, v, dropout_p=(dropout if training else 0.0),
                                    softmax_scale=scaling, causal=(q_len > 1))
                return o.transpose(1, 2), None
            except Exception:
                pass                                          # fall through to SDPA
        return _orig(query, key, value, sinks, num_key_value_groups=num_key_value_groups,
                     scaling=scaling, padding_mask=padding_mask, dropout=dropout,
                     training=training, output_attentions=output_attentions)
    base.full_attention = _wrapped


_APPLY = {"liger_norm": patch_liger_norm, "liger_rope": patch_liger_rope, "moe": patch_fused_moe,
          "router": patch_conv_router, "router_gate": patch_router_gate}


def apply(components):
    """components: iterable subset of {'liger_norm','liger_rope','moe'}. Returns the list applied."""
    done = []
    # router_gate first (stable sort keeps the rest in order): patch_conv_router wraps whatever
    # forward is current, so the gate swap must already be installed for conv's fallback to use it.
    for c in sorted(components, key=lambda c: c != "router_gate"):
        if c == "ce":
            continue  # CE lives in the training loop
        if c not in _APPLY:
            raise ValueError(f"unknown patch {c!r}; valid: {list(_APPLY) + ['ce']}")
        _APPLY[c]()
        done.append(c)
    return done
