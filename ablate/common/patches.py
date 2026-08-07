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
# 8  = radial NormSiLU, p = sigmoid(theta) in (0,1): gain r^p bounded by [1, r], p->0 IS normsilu.
# 10 = radial NormSiLU, p = tanh(theta) in (-1,1): additionally admits gain < 1 (r^-1 shrinks
#      high-rms rows). Motivated by measured dL/dp > 0 in 6/6 layers at low p on a trained k=8
#      checkpoint -- the model asking for a steeper ramp than sigmoid's floor allows.
# train.py sets RADIAL_P from --radial_p. NOTE this steers the PATCHED (Triton) forward only;
# src's eager BiBoFusedExperts hardcodes sigmoid, so a code-10 run without the 'moe' patch would
# silently compute code 8. train.py asserts the patch is present.
RADIAL_P = "sigmoid"
RADIAL_CODES = {"sigmoid": 8, "tanh": 10}
# train.py sets this from --act. "silu" swaps the GLU experts to plain SwiGLU (kernel act code 0,
# the same code the Qwen arm uses) so the activation itself becomes an ablation axis again inside
# the otherwise-current stack. src's eager path has been radial-only since the Aug 1 debloat and is
# NOT switched by this -- training runs the patched path, and train.py asserts the patch is on.
EXPERT_ACT = "radial"
SILU_CODE = 0
POS_IDENTITY_CODE = 3
NEG_IDENTITY_CODE = 4


_neg_identity_checked = []


def _assert_kernel_does_neg_identity(device, dtype):
    """One-time probe: does THIS kernel build implement act code 4 as -Identity (-w*x), or is it
    still the old ZERO expert (contribute nothing)?

    triton-kernel-fused is a separate repo on a separate checkout. A box that cloned it before
    Jul 26 2026 answers 'zero', and then every token routed to a -Identity expert is silently
    DROPPED instead of subtracted. No exception, no shape error, no warning: just a quietly wrong
    model and a wasted run. Version strings can't catch it -- only asking the kernel does."""
    from kernels.sm120.moe import moe_per_expert as moe_fused
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


def _expert_codes(experts, device, dtype):
    """Per-expert-id act codes, cached on the module. SHARED by the 'moe' and 'megakernel' patches
    on purpose: two copies of this mapping would be free to drift, and a stack where the two paths
    disagree about what an expert computes fails as a quality result rather than an error."""
    codes = getattr(experts, "_act_codes", None)
    if codes is not None and codes.device == device:
        return codes
    n_neg = experts.neg_end - experts.neg_start
    if n_neg and not _neg_identity_checked:
        _assert_kernel_does_neg_identity(device, dtype)
        _neg_identity_checked.append(True)
        print("[moe] kernel probe: act code 4 == -Identity (-w*x) OK", flush=True)
    glu_code = SILU_CODE if EXPERT_ACT == "silu" else RADIAL_CODES[RADIAL_P]
    lst = ([glu_code] * experts.num_glu_experts
           + [POS_IDENTITY_CODE] * (experts.pos_end - experts.pos_start)
           + [NEG_IDENTITY_CODE] * n_neg)
    codes = torch.tensor(lst, dtype=torch.int32, device=device)
    experts._act_codes = codes
    return codes


def _act_params(experts):
    """radial_theta is the exponent LOGIT, and the kernel reads it from act_params column 0.
    It is not optional -- code 8 raises without it (see parity_check/parity_radial.py).
    act_params rows are indexed by EXPERT ID, so the ±Identity specials need rows too even
    though nothing reads them: radial_theta is only (num_glu,). Passing the short tensor makes
    backward hand autograd a gradient of the wrong shape, several frames from the cause.

    act code 0 does not read act_params. Returning None rather than the tensor makes that
    explicit: radial_theta then provably cannot reach the kernel, so it takes no gradient and
    `p` stays pinned at its init in the log -- a live check that the --act silu arm really is
    SiLU and not radial wearing a different tag."""
    if EXPERT_ACT == "silu":
        return None
    ap = experts.radial_theta
    n_pad = experts.neg_end - experts.num_glu_experts
    if n_pad:
        ap = torch.cat([ap, ap.new_zeros(n_pad)])
    return ap.unsqueeze(1)


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

    def _bibo_moe(self, hidden_states, top_k_indices, top_k_weights):
        codes = _expert_codes(self, hidden_states.device, hidden_states.dtype)
        ap = _act_params(self)
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


# ───────────────────────── megakernel: norm + router + experts ─────────────────────────
def patch_megakernel():
    """Replace post_attention_layernorm + router + experts with ONE fused block on MoE layers.

    The seam is BiBoDecoderLayer._ffn_forward, NOT BiBoFusedExperts.forward, because the megakernel
    eats the NORM and the norm lives on the decoder layer one level above the MoE module. Both
    branches of BiBoDecoderLayer.forward (checkpointed and not) route through _ffn_forward, so this
    is the only place that needs patching.

    Dense layers (config.mlp_only_layers, 0 and 9 on the board config) fall through untouched.

    This SUPERSEDES 'moe' on MoE layers: the fused block calls moe_per_expert itself, so a run with
    both patches simply leaves BiBoFusedExperts.forward patched and never calls it.

    All router diagnostics are preserved (top1w / rent / bal / gap) by replaying the hooks of the
    two modules this path bypasses -- see the comment at the call site.
    """
    from kernels.sm120.megakernel.moe.block import megakernel_block
    from src.modeling.layers import BiBoDecoderLayer

    _orig = getattr(BiBoDecoderLayer, "_orig_ffn_forward", None) or BiBoDecoderLayer._ffn_forward
    BiBoDecoderLayer._orig_ffn_forward = _orig

    def _ffn(self, hidden_states):
        if not self.is_moe_layer:
            return _orig(self, hidden_states)
        moe = self.mlp
        # A shared expert consumes the NORMED hidden, which the fused block computes internally and
        # never returns. Failing loudly beats silently dropping it and reading the result as
        # "the megakernel costs quality".
        assert not moe.use_shared_expert, "megakernel patch does not support shared experts"
        # the fused router hardcodes sum-norm; a softmax arm would be silently renormalised
        assert moe.gate.norm_topk_prob and moe.gate.norm_topk_prob != "softmax", \
            f"megakernel router is sum-norm only, got norm_topk_prob={moe.gate.norm_topk_prob!r}"
        b, s, h = hidden_states.shape
        residual = hidden_states
        # gate_proj is nn.Linear(H, E) so .weight is [E,H]; the kernel takes [H,E] and reads it with
        # no stride arguments, so a transposed VIEW would be silently misread -- .contiguous() is
        # load-bearing, not tidiness. 512x64 elements, so the copy is free.
        w = {"nw": self.post_attention_layernorm.weight,
             "rw": moe.gate.gate_proj.weight.t().contiguous(),
             "bias": moe.gate.bias,
             "gu": moe.experts.gate_up_proj, "dn": moe.experts.down_proj}
        flat = hidden_states.reshape(b * s, h)
        # RouterTrace sets _probe_gap on the router when it wants the boundary gap; the fused
        # kernel only pays for it when asked
        want_gap = bool(getattr(moe.gate, "_probe_gap", False))
        out, idx, wgt, gap = megakernel_block(
            flat, w, _expert_codes(moe.experts, hidden_states.device, hidden_states.dtype),
            top_k=moe.gate.top_k, eps=self.post_attention_layernorm.variance_epsilon,
            act_params=_act_params(moe.experts), return_routing=True, want_gap=want_gap)
        # The diagnostics hang off TWO modules this path no longer calls, so both sets of hooks
        # have to be replayed or the log line silently goes blank while the model stays correct:
        #   BiBoFusedExperts forward PRE-hook -> top1w, rent, bal   (MoEStats + RouterTrace)
        #   BiBoMoERouter    forward hook     -> gap                (RouterTrace, reads .boundary_gap)
        # `bal` is how expert collapse gets noticed, so losing it quietly is the expensive failure.
        for _h in moe.experts._forward_pre_hooks.values():
            _h(moe.experts, (flat, idx.long(), wgt))
        if gap is not None:
            moe.gate.boundary_gap = gap.mean()
        for _h in moe.gate._forward_hooks.values():
            _h(moe.gate, (hidden_states,), (idx.view(b, s, -1).long(), wgt.view(b, s, -1).float()))
        # the balancing bias is driven from these indices and mutated by .add_() outside the
        # optimizer; it must fire exactly once per forward, exactly as BiBoMoELayer.forward does it
        if moe.training and moe.bias_update_factor > 0:
            tpe = moe._balance_step(idx.view(b, s, -1).long(), b * s)
            if tpe is not None:
                moe.update_bias(tpe)
        out = out.view(b, s, h)
        if self._bf16_moe_out:
            out = out.to(torch.bfloat16)
        hidden_states = residual + out
        if self._bf16_stream:
            hidden_states = hidden_states.to(torch.bfloat16)
        return hidden_states

    BiBoDecoderLayer._ffn_forward = _nc(_ffn)


# ───────────────────────── fused XSA (BiBo only) ─────────────────────────
def xsa_alpha_stats(model):
    """tanh(alpha) summary for the log line: 0 = XSA off on that head, 1 = full rejection."""
    vals = [m.xsa_alpha for m in model.modules() if getattr(m, "xsa_alpha", None) is not None]
    if not vals:
        return {}
    a = torch.tanh(torch.cat([v.detach().float().flatten() for v in vals]))
    return {"train/xsa_a_mean": a.mean().item(), "train/xsa_a_min": a.min().item(),
            "train/xsa_a_max": a.max().item()}


def patch_fused_xsa():
    """Route src's eager apply_xsa through the tkf Triton kernel.

    Without this, --use_xsa runs the EAGER rejection: src reads V twice and materializes the
    normalized V and the projection intermediate. The kernel fuses it into one pass. Both compute
    the same thing (Y - (Y.Vn)Vn), so this is a throughput patch, not a behaviour change -- which
    is exactly why it needs a numeric gate rather than a shrug: a silently-wrong fused XSA would
    look like 'XSA hurts quality' in an A/B and send the whole round the wrong way.

    Slices V to the query positions before the call, mirroring eager. For packed training
    q_len == kv_len so it is a no-op; it matters only for cached decode.

    `alpha` is src's per-head logit, forwarded verbatim: the kernel applies tanh(alpha) and
    returns the gradient already chained through the tanh, so both paths take the same argument
    and mean the same thing by it (parity_check/parity_xsa_alpha.py pins that)."""
    from kernels.sm120.xsa import fused_xsa
    import src.modeling.attn.base as base

    def _xsa(attn_output, value_states, enable_gqa=True, alpha=None):
        return fused_xsa(attn_output, value_states[:, :, -attn_output.shape[2]:, :], alpha)
    base.apply_xsa = _nc(_xsa)


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
    """Route BiBo's global-attention HOT PATH (training: no mask/padding) through flash_attn_func.
    Any failure (flash missing, fp32, mask needed) falls back to the original SDPA path.
    Qwen gets flash via config._attn_implementation instead (native HF dispatch)."""
    import src.modeling.attn.base as base
    _orig = getattr(base, "_orig_full_attention", None) or base.full_attention
    base._orig_full_attention = _orig

    def _wrapped(query, key, value, *, num_key_value_groups, scaling,
                 padding_mask=None, dropout=0.0, training=False, output_attentions=False):
        q_len, kv_len = query.shape[-2], key.shape[-2]
        need_mask = (output_attentions or padding_mask is not None
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
        return _orig(query, key, value, num_key_value_groups=num_key_value_groups,
                     scaling=scaling, padding_mask=padding_mask, dropout=dropout,
                     training=training, output_attentions=output_attentions)
    base.full_attention = _wrapped


_APPLY = {"liger_norm": patch_liger_norm, "liger_rope": patch_liger_rope, "moe": patch_fused_moe,
          "xsa": patch_fused_xsa, "megakernel": patch_megakernel}


def apply(components):
    """components: subset of {'liger_norm','liger_rope','moe','xsa','megakernel'} ('ce' lives in
    the loop)."""
    done = []
    for c in components:
        if c == "ce":
            continue
        if c not in _APPLY:
            raise ValueError(f"unknown patch {c!r}; valid: {list(_APPLY) + ['ce']}")
        _APPLY[c]()
        done.append(c)
    return done
