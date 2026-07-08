"""Manas-vs-Muon A/B harness — runs bench/train.py with ALL triton-kernel-fused kernels patched
in, NO BiBo src/ edits. The ONLY thing that differs between the two arms is the optimizer:

  arm = "muon"  (bibo_muon.yaml)  -> FusedMuon      ns8 aurora-K1                 (baseline)
  arm = "manas" (bibo_manas.yaml) -> ManasOptimizer ns8 aurora-K1 + rolling probe (test)

Manas == the Muon baseline PLUS the Nexus rolling-probe gradient-alignment force, so a loss/OOD
delta isolates exactly the probe. The probe needs fwd/bwd at theta+d: bench/train.py brackets each
micro fwd/bwd with optimizer.apply_probe()/remove_probe() (guarded — inert for plain Muon).

Fused kernels patched (identical on BOTH arms):
  1. CE      -> fused_linear_cross_entropy   (never materializes the (N, vocab) logits)
  2. MoE     -> kernels.sm75.moe             (sorted dispatch + fused PolyGLU, Identity/Zero codes)
  3. XSA     -> kernels.sm75.xsa.fused_xsa   (one triton kernel fwd+bwd)
  4. router  -> fused_router / fused_mlp_router (norm_topk + routed_scaling folded in the epilogue)
  5. SDPA    -> flash + mem-efficient pinned, math off

Arm is read from the config: training.muon_optimizer ("muon" | "manas") + training.probe {...}.

Two-GPU Kaggle run (GPU0 = Muon, GPU1 = Manas, in parallel):
  CUDA_VISIBLE_DEVICES=0 python bench/exp_manas.py --config bench/configs/bibo_muon.yaml  &
  CUDA_VISIBLE_DEVICES=1 python bench/exp_manas.py --config bench/configs/bibo_manas.yaml &
  wait

Local smoke (RTX 3050, compile broken locally so pass --no_compile):
  python bench/exp_manas.py --config bench/configs/bibo_manas.yaml --no_wandb --no_compile --total_steps 5
"""
import os
import sys

import yaml

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)
TKF = os.path.abspath(os.path.join(REPO, "..", "triton-kernel-fused"))
for p in (REPO, BENCH, TKF):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from kernels.sm75.muon import FusedMuon
from kernels.sm75.manas import ManasOptimizer, NS8_COEFFS
from kernels.sm75.cross_entropy import fused_linear_cross_entropy
from kernels.sm75.moe import moe as moe_fused
from kernels.sm75.xsa import fused_xsa
from kernels.sm75.router import fused_router, fused_mlp_router

# ── arm state (populated from the config in __main__ before train() builds the optimizer) ──
_ARM = "muon"
_PROBE = {}


# ── 1. optimizer: FusedMuon (baseline) or ManasOptimizer (probe) — swaps in via _build_muon ──
import optim as bibo_optim  # noqa: E402  (bench/optim.py)


def _build_muon(muon_params, muon_lr, weight_decay, use_modded, compile_opt, muon_extra):
    """Ignores config's modded_muon/ns_steps — both arms share the ns8 aurora-K1 base so the ONLY
    difference is Manas's probe. Returns (opt, impl_name)."""
    common = dict(lr=muon_lr, momentum=0.95, weight_decay=weight_decay,
                  coeffs=NS8_COEFFS, scale_mode="aurora", aurora_k=1, ns_dtype=torch.float16)
    if _ARM == "manas":
        opt = ManasOptimizer(
            muon_params,
            probe_gamma=_PROBE.get("gamma", 0.08),
            probe_rho=_PROBE.get("rho", 0.98),
            probe_rank=_PROBE.get("rank", 8),
            probe_refresh=_PROBE.get("refresh", 200),
            comp=_PROBE.get("comp", None),
            **common,
        )
        return opt, (f"Manas[g{_PROBE.get('gamma',0.08)},rho{_PROBE.get('rho',0.98)},"
                     f"r{_PROBE.get('rank',8)}]")
    return FusedMuon(muon_params, **common), "FusedMuon[ns8,aurora-K1]"


bibo_optim._build_muon = _build_muon

# ── 2. fused-linear CE (logits never materialized when labels are given) ────
import src.modeling.models as bibo_models  # noqa: E402

_orig_forward = bibo_models.BiBoForCausalLM.forward


def _ce_forward(self, input_ids=None, labels=None, **kw):
    if labels is None:
        return _orig_forward(self, input_ids=input_ids, labels=None, **kw)
    out = self.model(
        input_ids=input_ids, use_cache=False, return_dict=True,
        **{k: v for k, v in kw.items() if k in ("attention_mask", "position_ids", "inputs_embeds")},
    )
    h = out.last_hidden_state
    hs = h[:, :-1, :].reshape(-1, h.shape[-1])
    ls = labels[:, 1:].reshape(-1).to(h.device)
    loss = fused_linear_cross_entropy(hs, self.lm_head.weight, ls)
    return bibo_models.CausalLMOutputWithPast(loss=loss, logits=None,
                                              past_key_values=out.past_key_values)


bibo_models.BiBoForCausalLM.forward = _ce_forward

# ── 3. fused MoE experts (per-expert path; Identity/Zero act codes) ─────────
from src.modeling.ffn.moe import BiBoFusedExperts  # noqa: E402


def _moe_forward(self, hidden_states, top_k_indices, top_k_weights):
    codes = getattr(self, "_act_codes", None)
    if codes is None or codes.device != hidden_states.device:
        lst = ([e % 3 for e in range(self.num_polyglu_experts)]
               + [3] * (self.identity_end - self.identity_start)
               + [4] * (self.zero_end - self.zero_start))
        codes = torch.tensor(lst, dtype=torch.int32, device=hidden_states.device)
        self._act_codes = codes
    return moe_fused(hidden_states, top_k_indices, top_k_weights,
                     self.gate_up_proj, self.down_proj, codes)


BiBoFusedExperts.forward = _moe_forward

# ── 4. fused XSA (one triton kernel fwd+bwd, GQA broadcast in-kernel) ───────
import src.modeling.attn.base as bibo_attn_base  # noqa: E402


def _xsa_patch(attn_output, value_states, enable_gqa=True):
    return fused_xsa(attn_output, value_states)


bibo_attn_base.apply_xsa = _xsa_patch

# ── 5. fused router — conv (cuDNN) AND mlp (cuBLAS), norm_topk/scaling folded in ──
from src.modeling.ffn.router import BiBoMoERouter  # noqa: E402

_orig_router_forward = BiBoMoERouter.forward


def _router_forward(self, hidden_states):
    if self.gate_type != "sigmoid" or self.router_activation != "none":
        return _orig_router_forward(self, hidden_states)
    if self.router_type == "conv":
        return fused_router(hidden_states, self.gate_conv.weight, self.bias, self.top_k,
                            self.num_routed_experts, self.norm_topk_prob, self.routed_scaling_factor)
    return fused_mlp_router(hidden_states, self.gate_proj.weight, self.bias, self.top_k,
                            self.num_routed_experts, self.norm_topk_prob, self.routed_scaling_factor)


BiBoMoERouter.forward = _router_forward

# ── 6. attention: pin fast SDPA backends (mask-free is_causal -> flash; mem-eff fallback; no math) ──
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

import train as bibo_train  # noqa: E402


if __name__ == "__main__":
    args = bibo_train.parse_args()
    with open(args.config) as f:
        _cfg = yaml.safe_load(f)
    _ARM = _cfg["training"].get("muon_optimizer", "muon").lower()
    _PROBE = _cfg["training"].get("probe", {}) or {}
    assert _ARM in ("muon", "manas"), f"training.muon_optimizer must be muon|manas, got {_ARM}"
    print(f"[exp_manas] arm={_ARM} | fused CE + MoE + XSA + router active"
          + (f" | probe={_PROBE}" if _ARM == "manas" else ""))
    bibo_train.train(args)
