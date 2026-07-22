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

try:
    _nc = torch.compiler.disable
except AttributeError:
    _nc = torch._dynamo.disable

# PolyGLU act-cycle override (codes: 0=silu, 1=relu2, 2=normsilu). None -> default (0,1,2) cycle.
# train.py sets this from --silu/--relu2/--normsilu, e.g. [0] = all-SiLU experts, [0,1] = silu/relu2 alternating.
ACT_CYCLE = None


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
        return moe_fused(hidden_states, top_k_indices, top_k_weights,
                         self.gate_up_proj, self.down_proj, codes)

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
        if (self.router_type == "conv" and self.gate_type == "sigmoid"
                and self.router_activation == "none"):
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
          "router": patch_conv_router}


def apply(components):
    """components: iterable subset of {'liger_norm','liger_rope','moe'}. Returns the list applied."""
    done = []
    for c in components:
        if c == "ce":
            continue  # CE lives in the training loop
        if c not in _APPLY:
            raise ValueError(f"unknown patch {c!r}; valid: {list(_APPLY) + ['ce']}")
        _APPLY[c]()
        done.append(c)
    return done
