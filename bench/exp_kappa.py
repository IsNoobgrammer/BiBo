"""Kappa A/B training harness — runs bench/train.py with three patches, NO BiBo source edits:

  1. Muon    -> triton-kernel-fused FusedMuon, arm selected by KAPPA_ARM env:
                  default : _DSV4_COEFFS (10 it) + aurora_k1     (repo default; r=1 kappa ~40-450 lottery)
                  b12     : KJ x10 + pinned x2 (12 it) + aurora_k1 (r=1 kappa ~1.3-11, no dither)
                  champ   : b12 + signed-perm dither eps=0.05 on SQUARE slices (r=1 kappa 1.00 always)
  2. CE loss -> fused_linear_cross_entropy (never materializes the (N, 81000) logits)
  3. MoE     -> kernels.sm75.moe (sorted dispatch + fused PolyGLU; handles Identity/Zero specials)

Usage (from BiBo repo root, its .venv):
  $env:KAPPA_ARM="champ"; python bench/exp_kappa.py --config bench/configs/exp_kappa.yaml --no_wandb --no_compile
"""
import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH)
TKF = os.path.abspath(os.path.join(REPO, "..", "triton-kernel-fused"))
for p in (REPO, BENCH, TKF):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from kernels.sm75.muon import FusedMuon, newton_schulz, _DSV4_COEFFS
from kernels.sm75.cross_entropy import fused_linear_cross_entropy
from kernels.sm75.moe import moe as moe_fused

ARM = os.environ.get("KAPPA_ARM", "default")
assert ARM in ("default", "b12", "champ"), ARM
KJ = (3.4445, -4.775, 2.0315)
PIN = (2.0, -1.5, 0.5)
B12 = (KJ,) * 10 + (PIN,) * 2


# ── 1. Muon arm ──────────────────────────────────────────────────────────────
class KappaMuon(FusedMuon):
    def __init__(self, params, lr=3e-4, momentum=0.95, weight_decay=0.0, **_ignored):
        coeffs = _DSV4_COEFFS if ARM == "default" else B12
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                         coeffs=coeffs, ns_dtype=torch.float16)

    def _polar(self, u):
        if ARM == "champ" and u.shape[-2] == u.shape[-1]:
            n = u.shape[-1]
            g = torch.Generator(device=u.device).manual_seed(999)
            r = torch.randperm(n, device=u.device, generator=g)
            c = torch.randperm(n, device=u.device, generator=g)
            s = (torch.randint(0, 2, (n,), device=u.device, generator=g) * 2 - 1).to(u.dtype)
            E = torch.zeros(n, n, device=u.device, dtype=u.dtype)
            E[r, c] = s
            smax = 2.0 * u.float().flatten(-2).norm(dim=-1) / (n ** 0.5)      # per-slice ~spectral norm
            u = u + 0.05 * smax.view(-1, 1, 1).to(u.dtype) * E
        return newton_schulz(u, self.coeffs, self.ns_dtype)


import optim as bibo_optim  # noqa: E402  (bench/optim.py)


def _build_muon(muon_params, muon_lr, weight_decay, use_modded, compile_opt, muon_extra):
    return KappaMuon(muon_params, lr=muon_lr, weight_decay=weight_decay), f"KappaMuon[{ARM}]"


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
    loss = fused_linear_cross_entropy(hs, self.lm_head.weight.to(hs.dtype), ls)
    return bibo_models.CausalLMOutputWithPast(loss=loss, logits=None,
                                              past_key_values=out.past_key_values)


bibo_models.BiBoForCausalLM.forward = _ce_forward

# ── 3. fused MoE experts (auto per-expert path: handles Identity/Zero codes) ─
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
                     self.gate_up_proj.to(hidden_states.dtype),
                     self.down_proj.to(hidden_states.dtype), codes)


BiBoFusedExperts.forward = _moe_forward

# ── run the unmodified trainer ───────────────────────────────────────────────
import train as bibo_train  # noqa: E402

if __name__ == "__main__":
    print(f"[exp_kappa] arm={ARM} | fused CE + fused MoE + KappaMuon active")
    args = bibo_train.parse_args()
    bibo_train.train(args)
