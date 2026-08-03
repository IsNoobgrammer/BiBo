"""Optimizer builder: bf16-safe FusedMuon (NS8, aurora-K1) for 2D/3D matrices + AdamW for the rest.
Identical for both arms. NEVER fp16 (see the fp16-divergence finding); ns_dtype defaults bf16.

═══════════════════════════════════════════════════════════════════════════════════════════════
 MUON + THE ROUTER: ORTHOGONALIZE PER EXPERT, NEVER PER KERNEL TAP OR PER HIDDEN DIM
═══════════════════════════════════════════════════════════════════════════════════════════════
Muon's Newton-Schulz treats the LEADING dim of a param as a BATCH, orthogonalizes each trailing
2D slice, and iterates on the SMALLER Gram. So the axis that gets decorrelated is decided purely
by the parameter's STORED SHAPE — there is no flag for it:

  param shape        NS batch   Gram     decorrelates          verdict for a ROUTER
  ---------------    --------   ------   -------------------    --------------------------------
  (E, H)             1          (E,E)    the E EXPERTS          CORRECT  <- MLP router
  (E, H*K)           1          (E,E)    the E EXPERTS          CORRECT  <- conv router (we store this)
  (E, H, K)          E          (K,K)    the K KERNEL TAPS      WRONG    <- an nn.Conv1d weight
  (E, K, H) etc.     E          (K,K)    whichever trailing
                                         dim is smaller         WRONG

WHY (E,H,K) IS WRONG, measured, not argued:
  * Expert de-collapse is the ONE thing Muon gives a router. Feed it experts collapsed onto a
    single direction (|cos| = 0.9999): the 2D path returns 0.9999 -> 0.0000 (de-collapsed), the
    3D path returns 0.9999 -> 0.9999. The 3D path CANNOT de-collapse experts, ever.
    (Nuance: NS is a polynomial in X(X^T X), so it maps EXACTLY-zero singular values to zero and
    cannot manufacture rank. At |cos| == 1.0000 exactly, neither path recovers -- but that is a
    measure-zero case; any real collapse carries noise and the 2D path fixes it.)
  * Over 300 real training steps the 3D layout drove router expert-correlation up 13x faster than
    the MLP router (d xcos +0.0460 vs +0.0035); reshaping to 2D cut that ~3x (+0.0157).
  * What it decorrelates INSTEAD is the temporal tap axis, which nobody asked for: tap |cos|
    0.487 -> 0.0002 on a single update.
  * Bonus hazard: a 3D param also falls into the `stacks` bucket below and becomes an xorth
    (cross-expert whitening) target, which the 2D MLP router can never be -> asymmetric arms.

HOW WE FIX IT: at the SOURCE, not here. `BiBoMoERouter` stores the conv router weight as a 2D
(E, H*K) nn.Parameter and `.view(E, H, K)` inside forward for the F.conv1d call. Consequences:
  - correct-by-construction: no param-group flag, nothing to remember, and it stays correct with
    the fused Muon UNMODIFIED (we cannot patch kernels.sm120.muon from this repo);
  - it lands in `mats` below (2D, never whitened) — exactly where the MLP router lands, so the
    mlp-vs-conv ablation differs ONLY in architecture;
  - `--router_optim adamw` catches it too (see `is_router`).
If you ever reintroduce a routing param whose LEADING dim is the expert axis and whose trailing
dims are a FAN-IN (taps x hidden, groups x hidden, ...), flatten the fan-in before it reaches Muon.

⚠️ THE OPPOSITE RULE HOLDS FOR MoE EXPERT STACKS. `...experts.gate_up_proj` / `...down_proj` are
(E, out, in) — each slice is a GENUINE weight matrix, so batched per-expert NS is exactly right
and they MUST stay 3D. Never apply the flatten to them; it would orthogonalize across experts
instead of within each expert's matrix. Router == flatten. Expert stack == keep batched.

Related, checked and closed: `normuon` (per-row post-NS normalize) is a NO-OP for a router. NS
returns U V^T and (U V^T)(U V^T)^T = U V^T V U^T = I when E <= fan_in, so every expert row already
has exactly unit norm (measured spread max/min = 1.0005; normuon-vs-polar rel diff 2.2e-4). It
only bites on the batched 3D path or when E > fan_in. Verifier: `src/.autoresearch/probe_router_muon.py --selfcheck`.
═══════════════════════════════════════════════════════════════════════════════════════════════
"""
from . import _paths  # noqa: F401
import torch

_KJ, _PIN = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)
NS8 = (_KJ,) * 6 + (_PIN,) * 2


def build_optimizers(model, muon_lr=3e-4, adam_lr=3e-4, wd=0.1, momentum=0.95, ns_dtype=torch.bfloat16,
                     scale_mode="aurora", xorth_post=0.0, xorth_gate_ref=0.3, xorth_ema=0.95,
                     xorth_warmup_steps=0, xorth_where="post", router_adamw=False,
                     act_scale_lr=None, cautious_decay=False, vec_matrices_adamw=False,
                     optim="muon", probe_gamma=0.0, probe_rho_step=0.96, probe_rank=0):
    from kernels.sm120.muon import FusedMuon   # Blackwell: gram-NS (self-gates to symmul/cuBLAS on small mats) + 8M knee
    stacks, mats, other = [], [], []
    n_router = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Both routers are 2D so the ndim rule ALREADY sends them to Muon -> `mats` (never whitened);
        # that is the default and every result to date was produced that way. router_adamw=True is the
        # ablation arm that moves the router to AdamW instead.
        #   .gate.gate_proj  -> MLP router, (E, H). The conv router was deleted from src Aug 1 2026.
        # The `.gate.` prefix is load-bearing: a conv SHARED EXPERT is at
        # shared_experts_list.*.gate_conv (an actual nn.Conv1d, 3D) and must NOT match here.
        is_router = ".gate.gate_proj." in n
        # Guard the axis rule: a 3D routing param means someone reintroduced an nn.Conv1d router and
        # NS would silently whiten kernel taps instead of experts.
        if is_router and p.ndim != 2:
            raise ValueError(
                f"router param '{n}' has ndim={p.ndim}, expected 2. Muon orthogonalizes the "
                f"trailing-2D slices batched over the leading dim, so a 3D router weight gets its "
                f"KERNEL TAPS decorrelated instead of its EXPERTS (and joins the xorth stack "
                f"bucket). Store it as (num_routed_experts, fan_in) and view() it in forward — see "
                f"this module's docstring."
            )
        if is_router:
            n_router += 1
        if router_adamw and is_router:
            other.append((n, p))            # -> AdamW (ablation: router off Muon)
        elif "embed" in n or p.ndim not in (2, 3):
            other.append((n, p))            # -> AdamW (1D norms/biases, embeddings)
        elif vec_matrices_adamw and p.ndim == 2 and 1 in p.shape:
            # A (1, H) parameter is a VECTOR wearing a matrix shape -- AttnRes's pseudo-queries
            # are nn.Linear(hidden, 1). The ndim rule catches them by accident: Muon's convention
            # is 2D HIDDEN-LAYER MATRICES, with embeddings/heads/norms/biases on AdamW. For a
            # (1,H) the spectral norm IS the L2 norm, so Newton-Schulz returns the unit-normalised
            # row -- a fixed-L2-norm update every step regardless of gradient magnitude. Not
            # degenerate, but a very different dynamic from AdamW, and not one we chose.
            other.append((n, p))
        elif p.ndim == 3:
            stacks.append(p)                # 3D MoE expert stacks -> the xorth (cross-expert whitening) target
        else:
            mats.append(p)                  # plain 2D weight matrices -> never whitened
    print(f"[optim] router projections: {n_router} -> {'AdamW' if router_adamw else 'Muon'}", flush=True)
    # gram_restarts=[4,5] = the NS8-schedule fp16 autotune winner (gram only activates for dim>=2048; harmless below)
    # scale_mode = post-NS row scaling (ABLATION AXIS): aurora (default, no EMA) | normuon | aurora_ema |
    # aurora_ema_v2 (the EMA variants keep a persistent per-row 2nd-moment buffer) | polar.
    # xorth_post = cross-expert whitening MAX strength (0=off); SCOPED to the 3D expert stacks only (2D=0), so
    # whitening acts exactly on the MoE experts. xorth_gate_ref = correlation gate (full whitening at off-diag
    # RMS >= this; below it ramps to ~0 so decorrelated experts are left alone; <=0 disables gate). xorth_ema =
    # EMA decay of the persistent per-stack (E,E) gram (identity-init).
    groups = []
    if stacks:
        groups.append({"params": stacks, "xorth_post": xorth_post})
    if mats:
        groups.append({"params": mats, "xorth_post": 0.0})
    _xo = dict(xorth_post=xorth_post, xorth_gate_ref=xorth_gate_ref, xorth_ema=xorth_ema,
               xorth_warmup_steps=xorth_warmup_steps, xorth_where=xorth_where)
    # cautious_decay: decay a coordinate only where the update is already growing |W| (see
    # kernels/sm75/muon.py). MUON ONLY -- AdamW keeps standard decay, unlike modded-nanogpt which
    # applies it to both. Passed explicitly so the run record always states which mode was used:
    # every baseline on the board predates this and is NON-cautious.
    print(f"[optim] cautious weight decay: {bool(cautious_decay)} (Muon only; AdamW standard)", flush=True)
    _mk = dict(lr=muon_lr, momentum=momentum, weight_decay=wd, coeffs=NS8, ns_dtype=ns_dtype,
               aurora_k=1, gram_restarts=[4, 5], scale_mode=scale_mode,
               cautious_decay=bool(cautious_decay), **_xo)
    # optim=manas: the SAME sm120 gram-NS Muon step (kernels.sm120.manas subclasses it cooperatively --
    # never import kernels.sm75.manas here, that would swap the NS backend along with the optimizer and
    # confound the A/B) plus the rolling probe. Every argument above is shared verbatim with the muon
    # arm, so the arms differ by the probe and nothing else.
    #   probe_rank=0 -> None = FULL-RANK micro-vote: state is manas_d + manas_prev_g, both bf16
    #   (~2.5 GB here), no Q/omega/QR/sketch. Chosen because the 137M rank ladder was monotone
    #   (r8 < r32 <= r64 < r512 on train AND bpb) -- the low-rank sketch was the bottleneck.
    #   probe_gamma: dose, per the measured law gamma = 0.08*sqrt(lr/3e-4)*k/sqrt(m) (train.py computes
    #   it). 0 = probe never engages and this is EXACTLY FusedMuon -- the plumbing control arm.
    if optim == "manas":
        from kernels.sm120.manas import ManasOptimizer
        muon = ManasOptimizer(groups, micro_vote=True, probe_rank=(probe_rank or None),
                              probe_rho=1.0, probe_rho_step=probe_rho_step,
                              probe_gamma=probe_gamma, **_mk)
        print(f"[optim] MANAS probe: gamma={probe_gamma:g} rho_step={probe_rho_step:g} "
              f"rank={probe_rank or 'full'} micro_vote=True (engages at >= {muon.probe_min_votes} votes/step)",
              flush=True)
    elif optim == "muon":
        muon = FusedMuon(groups, **_mk)
    else:
        raise ValueError(f"unknown optim {optim!r}; valid: muon, manas")
    # act_scale_lr: radial_theta is the exponent LOGIT (p = sigmoid(theta)), and the measured depth
    # ramp runs p 0.11 -> 0.93, i.e. theta from about -2.1 to +2.6. AdamW at adam_lr=5e-4 moves it
    # ~0.5 over 2000 steps, so it needs its own group and its own lr. wd=0: decay on an exponent
    # would just re-impose the lr/wd equilibrium this axis exists to escape.
    _is_as = lambda n: ("radial_theta" in n or "xsa_alpha" in n
                        or "attn_res_carry_theta" in n)
    a_scale = [p for n, p in other if _is_as(n)]
    rest = [p for n, p in other if not _is_as(n)]
    if a_scale and act_scale_lr:
        print(f"[optim] act scales: {len(a_scale)} params -> AdamW lr={act_scale_lr:g} wd=0", flush=True)
        adamw = torch.optim.AdamW([{"params": rest},
                                   {"params": a_scale, "lr": act_scale_lr, "weight_decay": 0.0}],
                                  lr=adam_lr, weight_decay=wd)
    else:
        adamw = torch.optim.AdamW([p for _, p in other], lr=adam_lr, weight_decay=wd)
    return [muon, adamw], len(stacks) + len(mats), len(other)
