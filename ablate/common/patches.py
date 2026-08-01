"""Swappable kernel patches for BOTH arms (BiBo-min and Qwen3MoE), applied in place.

  'liger_norm' : LigerRMSNorm  -> BiBoRMSNorm + Qwen3MoeRMSNorm  (llama casting, offset 0 == eager)
  'liger_rope' : Liger RoPE    -> apply_rotary_pos_emb in bibo attn.base + qwen modeling
  'moe'        : tkf fused moe -> BiBoFusedExperts + Qwen3MoeExperts (act codes differ per arm)

Cross-entropy is NOT patched here: the training loop calls fused_linear_cross_entropy on the base
model's (hidden, lm_head.weight) directly, so CE is a swappable training-loop component, not a
model monkeypatch. Call apply(components) once before training.

CUT Aug 1 2026 to follow src: the router ablation surface (gate types, router_activation,
router_temperature, router_input_norm, routed_scaling_factor, the learnable layer/expert scales) and
add_situ_params. src's router is now MLP+sigmoid with nothing to swap, and radial's exponent theta is
a real nn.Parameter (`radial_theta`) on BiBoFusedExperts, so no injection step is needed at all.
"""
from . import _paths  # noqa: F401
import os
import torch

try:
    _nc = torch.compiler.disable
except AttributeError:
    _nc = torch._dynamo.disable


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
    bibo_attn_base.apply_rotary_pos_emb = _nc(_liger_rope)   # bibo calls it on the rope_dim slice
    qwen_mod.apply_rotary_pos_emb = _nc(_liger_rope)          # qwen calls it on full head_dim


# ───────────────────────── fused MoE ─────────────────────────
RADIAL_CODE = 8       # kernels: 8 = radial NormSiLU (r^p * SiLU(g/r), p = sigmoid(theta))
POS_IDENTITY_CODE = 3
NEG_IDENTITY_CODE = 4


def patch_fused_moe():
    from kernels.sm120.moe import moe_per_expert as moe_fused
    from kernels.sm75.moe import _code_max

    # BIBO_MOE_DISPATCH=per_expert|grouped|auto (default per_expert). Measured head-to-head at 30 GLU
    # on real training steps: per-expert 205.1k vs grouped 205.6k tok/s (0 specials) and 204.9k vs
    # 204.6k (10 specials) -- a WASH. The microbenchmark that predicted grouped 1.24-1.58x faster was
    # measuring an isolated call in a tight loop, where the per-layer host sync is free because the
    # CPU has nothing else queued. It does not survive contact with a real step.
    # Radial needs act_params, which the grouped path does not accept, so grouped only ever engages
    # on a stack with no GLU experts -- kept for the A/B, not expected to fire.
    _DISPATCH = os.environ.get("BIBO_MOE_DISPATCH", "per_expert")

    _neg_identity_checked = []

    def _assert_kernel_does_neg_identity(device, dtype):
        """One-time probe: does THIS kernel build implement act code 4 as -Identity (-w*x), or is it
        still the old ZERO expert (contribute nothing)?

        triton-kernel-fused is a separate repo on a separate checkout. A box that cloned it before
        Jul 26 2026 answers 'zero', and then every token routed to a -Identity expert is silently
        DROPPED instead of subtracted. No exception, no shape error, no warning: just a quietly wrong
        model and a wasted run. Version strings can't catch it -- only asking the kernel does."""
        H, I = 8, 2
        x = torch.ones(1, H, device=device, dtype=dtype)
        gu = torch.zeros(1, 2 * I, H, device=device, dtype=dtype)   # 1 dummy GLU slot, never routed
        dn = torch.zeros(1, H, I, device=device, dtype=dtype)
        codes = torch.tensor([0, NEG_IDENTITY_CODE], dtype=torch.int32, device=device)
        out = moe_fused(x, torch.tensor([[1]], device=device),
                        torch.ones(1, 1, device=device, dtype=dtype), gu, dn, codes)
        if torch.allclose(out.float(), -x.float(), atol=1e-3):
            return
        import kernels
        raise RuntimeError(
            f"kernel act code 4 is NOT -Identity: routing w=1 through it gave {out.flatten()[:4].tolist()} "
            f"(expected all -1). This kernel still implements code 4 as the ZERO expert, so every "
            f"-Identity token would be silently dropped instead of subtracted.\n"
            f"  kernels package: {getattr(kernels, '__file__', '?')}\n"
            f"  Fix: update the triton-kernel-fused checkout (needs commit 46727c8 or later, on "
            f"master), or run with --no_neg_identity / --special_pairs 0.")

    def _bibo_moe(self, hidden_states, top_k_indices, top_k_weights):
        codes = getattr(self, "_act_codes", None)
        if codes is None or codes.device != hidden_states.device:
            n_neg = self.neg_end - self.neg_start
            if n_neg and not _neg_identity_checked:
                _assert_kernel_does_neg_identity(hidden_states.device, hidden_states.dtype)
                _neg_identity_checked.append(True)
                print("[moe] kernel probe: act code 4 == -Identity (-w*x) OK", flush=True)
            lst = ([RADIAL_CODE] * self.num_glu_experts
                   + [POS_IDENTITY_CODE] * (self.pos_end - self.pos_start)
                   + [NEG_IDENTITY_CODE] * n_neg)
            codes = torch.tensor(lst, dtype=torch.int32, device=hidden_states.device)
            self._act_codes = codes
        # radial_theta is the exponent LOGIT, and the kernel reads it from act_params column 0.
        # It is not optional -- code 8 raises without it (see parity_check/parity_radial.py).
        ap = self.radial_theta.unsqueeze(1)
        if _DISPATCH != "per_expert" and _code_max(codes) <= 4:
            from kernels.sm120.moe_grouped import moe_grouped_cublas_polyglu, grouped_supported
            if grouped_supported(hidden_states, self.gate_up_proj, self.down_proj):
                return moe_grouped_cublas_polyglu(hidden_states, top_k_indices, top_k_weights,
                                                  self.gate_up_proj, self.down_proj, codes)
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


# ───────────────────────── FlashAttention (both arms) ─────────────────────────
def flash_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def resolve_attn(impl):
    """Downgrade a flash impl to 'sdpa' when flash_attn isn't importable (local / T4)."""
    if impl and impl.startswith("flash") and not flash_available():
        print(f"[attn] {impl} requested but flash_attn unavailable -> falling back to sdpa", flush=True)
        return "sdpa"
    return impl or "sdpa"


def patch_bibo_flash():
    """Route BiBo's global-attention HOT PATH (training: no mask/sink/padding) through flash_attn_func.
    Any failure (flash missing, fp32, mask needed) falls back to the original SDPA path.
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
                q = query.transpose(1, 2).contiguous()      # (B,S,H,d) -- flash layout
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


_APPLY = {"liger_norm": patch_liger_norm, "liger_rope": patch_liger_rope, "moe": patch_fused_moe}


def apply(components):
    """components: iterable subset of {'liger_norm','liger_rope','moe'} ('ce' lives in the loop)."""
    done = []
    for c in components:
        if c == "ce":
            continue
        if c not in _APPLY:
            raise ValueError(f"unknown patch {c!r}; valid: {list(_APPLY) + ['ce']}")
        _APPLY[c]()
        done.append(c)
    return done
