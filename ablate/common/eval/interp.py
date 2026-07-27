"""Interpretability metrics collected DURING eval forwards (hooks on the MoE experts of BOTH arms):
  - expert utilization: per-expert load, balance-entropy (1=perfectly balanced), coeff-of-variation, max/min.
  - router confidence: mean top-1 gate weight, mean routing entropy over the top-k, frac(top1>0.5).
Zero training cost (only active inside the `collect` context around eval). Works for BiBoFusedExperts and
Qwen3MoeExperts (both take (hidden, top_k_index, top_k_weights))."""
from .. import _paths  # noqa: F401
import math
import torch

_EXPERT_CLASSES = ("BiBoFusedExperts", "Qwen3MoeExperts")


class MoEStats:
    def __init__(self, model, num_experts):
        self.E = num_experts
        self.counts = torch.zeros(num_experts, dtype=torch.float64)
        self.top1_sum = 0.0
        self.entropy_sum = 0.0
        self.top1_gt_half = 0
        self.tokens = 0
        self._handles = []
        for _, mod in model.named_modules():
            if mod.__class__.__name__ in _EXPERT_CLASSES:
                self._handles.append(mod.register_forward_pre_hook(self._hook))

    @torch.no_grad()
    def _hook(self, module, args):
        # args = (hidden_states, top_k_index, top_k_weights); idx (N,k) expert ids, w (N,k) gate weights
        idx, w = args[1], args[2]
        self.counts += torch.bincount(idx.reshape(-1).to("cpu"), minlength=self.E).double()
        w = w.detach().float().cpu()
        # L1 divisor, NOT sum(w): identical for a non-negative gate (sigmoid), but a SIGNED gate (situ
        # with norm_topk_prob=0) can drive sum(w) through ~0 and the ratio explodes -- measured
        # top1_weight of -2.2e6 on a real run. sum|w| >= max|w| keeps this diagnostic in [-1,1].
        # The MODEL is unaffected either way; this is only the measurement.
        p = w / w.abs().sum(-1, keepdim=True).clamp_min(1e-9)
        self.top1_sum += p.max(-1).values.sum().item()
        self.entropy_sum += (-(p * (p.clamp_min(1e-12)).log()).sum(-1)).sum().item()
        self.top1_gt_half += (p.max(-1).values > 0.5).sum().item()
        self.tokens += w.shape[0]

    def result(self):
        n = max(self.tokens, 1)
        total = self.counts.sum().clamp_min(1)
        load = (self.counts / total)                              # fraction of assignments per expert
        nz = load[load > 0]
        balance_entropy = float(-(nz * nz.log()).sum() / math.log(self.E)) if self.E > 1 else 1.0
        return {
            "expert_load": [round(x, 4) for x in load.tolist()],
            "balance_entropy": round(balance_entropy, 4),          # 1.0 = perfectly balanced, 0 = collapsed
            "load_cov": round(float(load.std() / load.mean().clamp_min(1e-9)), 4),
            "max_expert_load": round(float(load.max()), 4),
            "min_expert_load": round(float(load.min()), 4),
            "router_top1_weight": round(self.top1_sum / n, 4),     # mean confidence in the chosen expert
            "router_entropy": round(self.entropy_sum / n, 4),      # mean entropy over the top-k gate (nats)
            "router_frac_top1_gt_0.5": round(self.top1_gt_half / n, 4),
            "tokens_seen": self.tokens,
        }

    def close(self):
        for h in self._handles:
            h.remove()


class RouterTrace:
    """Router mechanics DURING TRAINING, with no GPU->CPU sync in the hot path.

    MoEStats above is eval-only because its hook calls .cpu() on every MoE layer of every forward --
    that is a device sync per layer per step, which would badly slow training. Here everything
    accumulates into DEVICE buffers and there is exactly ONE transfer per flush() (i.e. per
    log_every), not one per layer per step.

    Hooks are the same point as MoEStats: forward_pre_hook on the expert module, whose args are
    (hidden_states, top_k_index, top_k_weights). Eval forwards are skipped (module.training False)
    so the training trace is not polluted by eval-distribution routing. Under gradient checkpointing
    a layer may be hooked twice per step; every reported number is a ratio, so double-counting
    cancels."""

    def __init__(self, model, num_experts, device):
        self.E = int(num_experts)
        self.counts = torch.zeros(self.E, device=device, dtype=torch.float32)
        # [split_top1, entropy, tokens, RAW_top1, RAW_sum, RAW_min, boundary_gap, n_router_fwd, input_cv]
        self.acc = torch.zeros(9, device=device, dtype=torch.float32)
        self._handles = []
        # Expert-block boundaries, so the ±Identity specials get their own load channels. Read off the
        # first expert module (every layer shares the config). n_glu == E for a Qwen arm / no specials,
        # in which case the special channels are constant 0 and simply uninformative.
        self.n_glu, self.pos_end = self.E, self.E
        for _, mod in model.named_modules():
            if mod.__class__.__name__ in _EXPERT_CLASSES:
                if not self._handles:
                    self.n_glu = int(getattr(mod, "num_polyglu_experts", self.E))
                    self.pos_end = int(getattr(mod, "pos_end", self.E))
                self._handles.append(mod.register_forward_pre_hook(self._hook))
        # Routers, for the selection-BOUNDARY gap (rank-k vs rank-k+1 raw score). The expert hook
        # above never sees the full score vector -- it only gets (hidden, idx, weights) -- so this
        # has to be read off the router itself. It is the gap the balancing bias competes against.
        for _, mod in model.named_modules():
            if mod.__class__.__name__ == "BiBoMoERouter":
                mod._probe_gap = True
                mod.boundary_gap = torch.zeros((), device=device)
                mod.input_cv = torch.zeros((), device=device)
                self._handles.append(mod.register_forward_hook(self._router_hook))

    @torch.no_grad()
    def _router_hook(self, module, args, output):
        if not module.training or module.boundary_gap is None:
            return
        self.acc[6] += module.boundary_gap
        self.acc[7] += 1.0
        if module.input_cv is not None:
            self.acc[8] += module.input_cv     # CV of ||h_t|| AT THE ROUTER INPUT, before any router norm

    @torch.no_grad()
    def _hook(self, module, args):
        if not module.training:
            return                                    # eval forwards use MoEStats, not this
        idx, w = args[1], args[2]
        self.counts += torch.bincount(idx.detach().reshape(-1), minlength=self.E).to(self.counts.dtype)
        wr = w.detach().float()
        # RAW weights, exactly as the MoE consumes them -- no rescaling, so these are the ground truth
        # (bounded by the gate: sigmoid (0,1), situ [-0.207,1)). router_w_sum IS the branch-magnitude
        # channel: ~1.0 when norm_topk_prob=1, ~1.34 for un-normalized sigmoid.
        self.acc[3] += wr.max(-1).values.sum()
        self.acc[4] += wr.sum()
        self.acc[5] += wr.min(-1).values.sum()
        # derived SPLIT (how lopsided the mixture is), L1-normalized so a signed gate can't explode it
        p = wr / wr.abs().sum(-1, keepdim=True).clamp_min(1e-9)
        self.acc[0] += p.max(-1).values.sum()
        self.acc[1] += (-(p.abs() * p.abs().clamp_min(1e-12).log()).sum(-1)).sum()
        self.acc[2] += p.shape[0]

    @torch.no_grad()
    def flush(self):
        """Return the metrics for the interval and reset. Exactly one device->host transfer."""
        if float(self.E) < 2:
            return {}
        load = self.counts / self.counts.sum().clamp_min(1.0)
        bal = -(load * load.clamp_min(1e-12).log()).sum() / math.log(self.E)   # 0*log0 == 0, as intended
        n = self.acc[2].clamp_min(1.0)
        # Fraction of top-k slots landing on the special block, and on the -Identity half of it. This
        # is the channel the glu_token_budget knob controls (budget r => special_load -> 1-r) and the
        # one that says whether the router WANTS signed pass-through or is only being pushed to it.
        packed = torch.stack([bal, self.acc[0] / n, self.acc[1] / n, load.max(), self.acc[2],
                              self.acc[3] / n, self.acc[4] / n, self.acc[5] / n,
                              load[self.n_glu:].sum(), load[self.pos_end:].sum(),
                              self.acc[6] / self.acc[7].clamp_min(1.0),
                              self.acc[8] / self.acc[7].clamp_min(1.0)])
        b, t1, ent, mx, ntok, rw1, rws, rwmin, spl, negl, gap, icv = packed.cpu().tolist()   # <-- THE ONLY SYNC
        self.counts.zero_(); self.acc.zero_()
        if ntok < 1:
            return {}                                     # nothing accumulated (e.g. logged before a step)
        return {"train/balance_entropy": b, "train/router_top1_weight": t1,
                "train/router_entropy": ent, "train/max_expert_load": mx,
                "train/router_w_top1": rw1,      # RAW top-1 weight (gate-bounded)
                "train/router_w_sum": rws,       # RAW sum over top-k = the branch magnitude channel
                "train/router_w_min": rwmin,     # RAW min weight (negative => an expert is SUBTRACTED)
                "train/special_load": spl,       # share of top-k slots on the ±Identity block
                "train/neg_identity_load": negl,  # ... of which, on the -Identity half
                # mean rank-k minus rank-(k+1) RAW score gap = the selection boundary the bias must
                # close. bias_update_factor u >> this gap => one step flips a large fraction of
                # tokens => the balancer overshoots and dithers instead of converging.
                "train/router_boundary_gap": gap,
                # CV of ||h_t|| across tokens AT THE ROUTER INPUT, measured BEFORE any
                # router_input_norm. This is the headroom for that knob: logits scale with
                # ||h_t||, so a large CV means the gate temperature is effectively per-token.
                "train/router_input_cv": icv}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class collect:
    """Context manager: `with collect(model, num_experts) as c: <eval forwards>; stats = c.result()`."""
    def __init__(self, model, num_experts):
        self._stats = MoEStats(model, num_experts)

    def __enter__(self):
        return self._stats

    def __exit__(self, *exc):
        self._stats.close()
        return False
