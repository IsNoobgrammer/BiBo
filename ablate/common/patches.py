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
import os
import torch
import torch.nn.functional as F

try:
    _nc = torch.compiler.disable
except AttributeError:
    _nc = torch._dynamo.disable

# PolyGLU act-cycle override (codes: 0=silu, 1=relu2, 2=normsilu, 5=situ, 6=normrelu2, 7=normsitu). None -> default (0,1,2) cycle.
# train.py sets this from --act, e.g. [0] = all-SiLU experts (the default), [0,5] = silu/situ alternating.
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

# MoE-branch magnitude control (all applied AFTER the top-k norm, so they deliberately break the
# sum-to-1 property -- that is the point: normalization fixes the SPLIT, these set the MAGNITUDE).
#   ROUTER_SCALE      fixed global scalar (--routed_scaling_factor), 1.0 = no-op.
#   router_scale      learnable per-LAYER scalar (--routed_scaling_learnable), init 1.0.
#   expert_scale      learnable per-EXPERT vector (--expert_scale_learnable), init 1.0.
# The per-expert vector is out_gain's mechanism, but out_gain was tested WITHOUT normalization, where
# the router's un-normalized weight sum was already a free per-token magnitude channel -- so it was
# redundant and did nothing. With sum-to-1 normalization that channel is gone, so the scale is now the
# ONLY magnitude control: router picks + splits, scale sets loudness. Unbounded on purpose (init 1.0)
# so the learned value REPORTS how much magnitude the model actually wants; bound it only if it drifts.
ROUTER_SCALE = 1.0


def add_router_scales(model, learn_layer=False, learn_expert=False):
    """Register the learnable magnitude params on every BiBoMoERouter. 1D so build_optimizers' ndim>=2
    rule routes them to AdamW (and the bf16-ckpt rule keeps them fp32). Call AFTER build_arm, BEFORE
    build_optimizers. Returns the number of routers touched."""
    import torch.nn as nn
    import src.modeling.ffn.router as rmod
    n = 0
    for m in model.modules():
        if isinstance(m, rmod.BiBoMoERouter):
            dev = m.bias.device
            if learn_layer:
                m.router_scale = nn.Parameter(torch.ones(1, device=dev))
            if learn_expert:
                m.expert_scale = nn.Parameter(torch.ones(m.num_routed_experts, device=dev))
            n += 1
    return n


def router_scale_stats(model):
    """(mean, min, max) of the learned scales, for logging. Empty dict when none are enabled."""
    import src.modeling.ffn.router as rmod
    out = {}
    for tag in ("router_scale", "expert_scale"):
        vals = [getattr(m, tag) for m in model.modules()
                if isinstance(m, rmod.BiBoMoERouter) and getattr(m, tag, None) is not None]
        if vals:
            v = torch.cat([p.detach().reshape(-1) for p in vals]).float()
            out[f"train/{tag}_mean"] = v.mean().item()
            out[f"train/{tag}_min"] = v.min().item()
            out[f"train/{tag}_max"] = v.max().item()
    return out


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
            E = m.neg_end   # rows must match the codes tensor length (polyglu + specials)
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
    # special experts correctly. (moe() auto-dispatch would wrongly pick grouped at >=4096 tok.)
    from kernels.sm120.moe import moe_per_expert as moe_fused
    from kernels.sm120.moe_grouped import moe_grouped_cublas_polyglu, grouped_supported
    from kernels.sm75.moe import _code_max
    # BIBO_MOE_DISPATCH=per_expert|grouped|auto (default auto). A microbenchmark said grouped
    # wins, but an isolated bench CANNOT see the cost that matters here: the grouped path does
    # `int(offs[-1].item())`, a HOST SYNC, once per MoE layer per micro-batch. In a tight bench
    # loop the CPU has nothing else to do so the sync is free; in real training it blocks the CPU
    # from running ahead to queue the next layer. Keep this switch so the two can be A/B'd on the
    # real step time instead of on a microbenchmark.
    # DEFAULT per_expert. Measured head-to-head at 30 GLU on real training steps: per-expert
    # 205.1k vs grouped 205.6k tok/s (0 specials) and 204.9k vs 204.6k (10 specials) -- a WASH.
    # The microbenchmark that predicted grouped 1.24-1.58x faster was measuring an isolated call
    # in a tight loop, where the per-layer host sync is free because the CPU has nothing else
    # queued. It does not survive contact with a real step. Grouped IS ~3x more accurate vs the
    # eager reference (1.0e-3 vs 2.8e-3), so revisit if numerics ever matter more than matching
    # history -- but every 18-GLU arm we compare against ran per-expert, and at equal speed
    # comparability wins. Override with BIBO_MOE_DISPATCH=grouped|auto.
    _DISPATCH = os.environ.get("BIBO_MOE_DISPATCH", "per_expert")

    _neg_identity_checked = []

    def _assert_kernel_does_neg_identity(device, dtype):
        """One-time probe: does THIS kernel build implement act code 4 as -Identity (-w*x), or is it
        still the old ZERO expert (contribute nothing)?

        triton-kernel-fused is a separate repo on a separate checkout. A box that cloned it before
        Jul 26 2026 -- or cloned a branch without the change -- answers 'zero', and then every token
        routed to a -Identity expert is silently DROPPED instead of subtracted. No exception, no
        shape error, no warning: just a quietly wrong model and a wasted run. (This is not
        hypothetical; a fresh box cloned the stale default branch and would have run two arms that
        way.) Version strings can't catch it -- only asking the kernel does.

        Cost: one 1-token MoE call, once per process, and only when a -Identity block exists."""
        H, I = 8, 2
        x = torch.ones(1, H, device=device, dtype=dtype)
        gu = torch.zeros(1, 2 * I, H, device=device, dtype=dtype)   # 1 dummy GLU slot, never routed
        dn = torch.zeros(1, H, I, device=device, dtype=dtype)
        codes = torch.tensor([0, 4], dtype=torch.int32, device=device)
        out = moe_fused(x, torch.tensor([[1]], device=device),      # route the single token to code 4
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

    # BiBo: diverse PolyGLU activations (silu/relu2/normsilu cycled) + optional ±Identity specials
    def _bibo_moe(self, hidden_states, top_k_indices, top_k_weights):
        codes = getattr(self, "_act_codes", None)
        if codes is None or codes.device != hidden_states.device:
            cyc = ACT_CYCLE or (0, 1, 2)
            # Kernel codes 3/4 = +Identity / -Identity (code 4 meant ZERO until Jul 26 2026; the
            # kernel now emits -w*x for it -- gated by parity_specials.py in triton-kernel-fused).
            n_neg = self.neg_end - self.neg_start
            if n_neg and not _neg_identity_checked:
                _assert_kernel_does_neg_identity(hidden_states.device, hidden_states.dtype)
                _neg_identity_checked.append(True)
                print("[moe] kernel probe: act code 4 == -Identity (-w*x) OK", flush=True)
            lst = ([cyc[e % len(cyc)] for e in range(self.num_polyglu_experts)]
                   + [3] * (self.pos_end - self.pos_start)
                   + [4] * n_neg)
            codes = torch.tensor(lst, dtype=torch.int32, device=hidden_states.device)
            self._act_codes = codes
        ap = (torch.stack([self.situ_alpha, self.situ_gamma], dim=1)
              if getattr(self, "situ_alpha", None) is not None else None)
        # Prefer the sm120 cuBLAS GROUPED path. NOT moe(): its prefer_grouped() heuristic caps at
        # GROUPED_TOKENS_PER_EXPERT_MAX=2048 tokens/expert, but our shape (H=512, I=768) runs ~2979
        # and grouped still wins there -- measured fwd+bwd at N=32768: 4.90ms vs 3.96ms at 4
        # specials (1.24x), widening to 1.58x at 12. The gap GROWS with special count because
        # per-expert pays a launch per special while doing no GEMM (it gets 1.16x SLOWER from 0->12
        # specials; grouped gets 0.82x faster). Grouped is also ~3x more accurate vs the eager
        # reference (1.0e-3 vs 2.8e-3) -- one grouped GEMM instead of many small accumulations.
        # Falls back to per-expert when grouped cannot run: fp32 (torch._grouped_mm is bf16/fp16
        # only), act codes >4 (situ/normrelu2/normsitu), or learnable act_params.
        if (_DISPATCH != "per_expert" and ap is None and _code_max(codes) <= 4
                and grouped_supported(hidden_states, self.gate_up_proj, self.down_proj)):
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
        # router_logits() -- NOT self.gate_proj(). This body was written when the MLP router was the
        # only one, and hardcoding gate_proj made every router_gate!=sigmoid arm crash on a conv
        # router with AttributeError. router_logits() is the shared entry point and dispatches on
        # router_type, so this patch now composes with mlp AND conv.
        router_logits = self.router_logits(hidden_states)
        router_logits = self._apply_router_activation(router_logits)
        if getattr(self, 'router_temperature', 1.0) != 1.0:           # mirror the native forward
            router_logits = router_logits / self.router_temperature
        scores = _gate_scores(router_logits, self.gate_type)          # <<< the ONLY change
        # Mirror the native forward's boundary-gap probe. Without this the metric silently reads
        # 0.0000 on every router_gate!=sigmoid arm (this patch REPLACES forward, so the native
        # probe never runs and the buffer keeps its init) -- which is exactly what the situ arms
        # reported before this was fixed.
        if self._probe_gap and self.top_k < self.num_routed_experts:
            with torch.no_grad():
                _tk = scores.topk(self.top_k + 1, dim=-1).values
                self.boundary_gap = (_tk[..., self.top_k - 1] - _tk[..., self.top_k]).mean()
        selection_scores = scores + self.bias                          # bias: SELECTION only
        _, top_k_indices = torch.topk(selection_scores, self.top_k, dim=-1, sorted=False)
        top_k_weights = scores.gather(-1, top_k_indices)               # UNBIASED weights
        if self.top_k > 1 and self.norm_topk_prob:
            norm_weights = _norm_topk(top_k_weights)   # ROUTER_NORM: sum (shipped) | softmax | l1
        else:
            norm_weights = top_k_weights
        norm_weights = norm_weights * self.routed_scaling_factor
        # magnitude control (deliberately after the norm -> sum is no longer 1; see ROUTER_SCALE notes)
        if ROUTER_SCALE != 1.0:
            norm_weights = norm_weights * ROUTER_SCALE
        _rs = getattr(self, "router_scale", None)
        if _rs is not None:
            norm_weights = norm_weights * _rs                       # per-layer learnable scalar
        _es = getattr(self, "expert_scale", None)
        if _es is not None:
            norm_weights = norm_weights * _es[top_k_indices]        # per-expert learnable vector
        top_k_indices = rearrange(top_k_indices, '(b s) k -> b s k', b=batch_size)
        norm_weights = rearrange(norm_weights, '(b s) k -> b s k', b=batch_size)
        return top_k_indices.long(), norm_weights.float()

    R.forward = _nc(_fwd)


# NOTE: patch_conv_router() lived here and routed the CONV router through the sm120 fused-Triton
# conv kernel. Removed Jul 26 2026 along with the conv router itself — MLP router only.


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
          "router_gate": patch_router_gate}


def apply(components):
    """components: iterable subset of {'liger_norm','liger_rope','moe'}. Returns the list applied."""
    done = []
    # router_gate first (stable sort keeps the rest in order) — kept for ordering stability now that
    # the conv-router patch that depended on it is gone.
    for c in sorted(components, key=lambda c: c != "router_gate"):
        if c == "ce":
            continue  # CE lives in the training loop
        if c not in _APPLY:
            raise ValueError(f"unknown patch {c!r}; valid: {list(_APPLY) + ['ce']}")
        _APPLY[c]()
        done.append(c)
    return done
