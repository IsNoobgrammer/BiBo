"""Muon(own) / AdamW split for the ASR model (train_asr.py --optim muown) -- the rule in ONE place, printable.

Muon orthogonalises the TRAILING 2D slice of a parameter and batches over any leading dim (tkf kernels/sm75/muon.py
buckets by matrix shape), so what gets decorrelated is decided by the matrix shape. Parameters whose stored shape is
not the matrix carry `muon_rc` (tkf FusedMuon views them as numel // (r*c) x (r, c); parity_check/parity_muon_rc.py:
bitwise equal to a real parameter of that shape):
  muon       2D Linear weights: encoder FFN / attention (q k v out pos), pre-encode output projection, joint enc /
             pred projections, CTC-head GLU up / down (gate+value stacked = ONE matrix, as the LLM's gate_up)
  muon_flat  pointwise convs (out, in, 1[, 1]) -> muon_rc (out, in)
  muon_gate  prediction-net LSTM weight_ih / weight_hh (4H, in), gates i f g o stacked -> muon_rc (H, in): PER GATE
  adamw      1D (biases, LayerNorm, BatchNorm); depthwise / spatial convs (per-channel filters); pos_bias_u/v (per-head
             vectors stored as (H, d)); vocab embeddings and output heads (joint output, CTC readout, CTC
             self-conditioning feedback embeddings, prediction-net embedding)

    python voice/asr/muon_groups.py --tok .../v2047 --ctc_head glu_glu
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ADAMW_NAMES = [                                    # 2D+ parameters that are heads / embeddings / vectors
    (r"^joint\.joint_net\.\d+\.weight$", "joint vocab output head"),
    (r"^decoder\.prediction\.embed\.weight$", "prediction-net token embedding"),
    (r"^ctc_decoder\.head\.r\.\d+\.out\.weight$", "CTC vocab output head"),
    (r"^ctc_decoder\.head\.cond\.\d+\.(cur|prev)\.weight$", "CTC self-conditioning feedback embedding"),
    (r"\.pos_bias_[uv]$", "per-head bias vectors"),
]


def assign(model):
    """[(name, param, group, why)] for every trainable parameter; sets p.muon_rc where the matrix is reshaped."""
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        hit = next((why for pat, why in ADAMW_NAMES if re.search(pat, n)), None)
        if p.ndim < 2:
            g, why = "adamw", "1D (bias / norm)"
        elif hit:
            g, why = "adamw", hit
        elif re.search(r"\.lstm\.weight_(ih|hh)_l\d+$", n):
            H = p.shape[0] // 4
            p.muon_rc = (H, p.shape[1])
            g, why = "muon_gate", f"LSTM gates i f g o: 4 x ({H}, {p.shape[1]})"
        elif p.ndim == 2:
            g, why = "muon", "2D matrix"
        elif p.ndim in (3, 4) and all(s == 1 for s in p.shape[2:]) and p.shape[1] > 1:
            p.muon_rc = (p.shape[0], p.shape[1])
            g, why = "muon_flat", f"pointwise conv -> ({p.shape[0]}, {p.shape[1]})"
        else:
            g, why = "adamw", f"depthwise / spatial conv {tuple(p.shape)}"
        out.append((n, p, g, why))
    return out


def report(model, rows=None):
    rows = rows or assign(model)
    tot, agg = collections.Counter(), collections.OrderedDict()
    for n, p, g, why in rows:
        k = re.sub(r"layers\.\d+\.", "layers.*.", n)
        a = agg.setdefault(k, [g, tuple(p.shape), 0, 0, why])
        a[2] += 1
        a[3] += p.numel()
        tot[g] += p.numel()
    print(f"[muon_groups] {'group':9s} {'shape':>18s} {'count':>5s} {'params':>11s}  name  (why)", flush=True)
    for k, (g, s, c, num, why) in agg.items():
        print(f"[muon_groups] {g:9s} {str(s):>18s} {c:5d} {num:11,d}  {k}  ({why})", flush=True)
    allp = sum(tot.values())
    muon = sum(v for g, v in tot.items() if g.startswith("muon"))
    print(f"[muon_groups] TOTAL muon {muon / 1e6:.2f}M ({100 * muon / allp:.1f}%: " +
          ", ".join(f"{g} {v / 1e6:.2f}M" for g, v in sorted(tot.items()) if g.startswith("muon")) +
          f") | adamw {tot['adamw'] / 1e6:.2f}M ({100 * tot['adamw'] / allp:.1f}%) | all {allp / 1e6:.2f}M", flush=True)
    return rows


class _State(__import__("collections.abc").abc.MutableMapping):
    """optimizer.state of the combined optimizer: routes each parameter to the sub-optimizer that owns it (Lightning
    moves / restores optimizer state through this mapping on resume)."""

    def __init__(self, opts):
        self.opts = opts
        self.own = {id(q): o for o in opts for gr in o.param_groups for q in gr["params"]}

    def __getitem__(self, p):
        return self.own[id(p)].state[p]

    def __setitem__(self, p, v):
        self.own[id(p)].state[p] = v

    def __delitem__(self, p):
        del self.own[id(p)].state[p]

    def __iter__(self):
        for o in self.opts:
            yield from o.state

    def __len__(self):
        return sum(len(o.state) for o in self.opts)


class Combined(__import__("torch").optim.Optimizer):
    """Several optimizers behind ONE (Lightning's automatic optimization takes one): param_groups are the sub-
    optimizers' own group dicts (a scheduler on this object sets their lr), step / zero_grad / state_dict fan out."""

    def __init__(self, opts):                     # no super().__init__: the groups belong to the sub-optimizers
        self.opts = list(opts)
        self.defaults = {}
        self.param_groups = [g for o in self.opts for g in o.param_groups]
        self.state = _State(self.opts)
        self._optimizer_step_pre_hooks, self._optimizer_step_post_hooks = {}, {}
        self._optimizer_state_dict_pre_hooks, self._optimizer_state_dict_post_hooks = {}, {}
        self._optimizer_load_state_dict_pre_hooks, self._optimizer_load_state_dict_post_hooks = {}, {}
        self._zero_grad_profile_name = "Optimizer.zero_grad#Combined.zero_grad"

    def step(self, closure=None):
        import torch
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self.opts:
            o.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["combined"]):
            o.load_state_dict(s)

    def __repr__(self):
        return "Combined(" + ", ".join(type(o).__name__ for o in self.opts) + ")"


def build(model, lr, muon_lr=None, wd=1e-3, muon_wd=None, momentum=0.95, variant="muown", ns="dsv4", rows=None):
    """FusedMuon (tkf sm120, variant muown, scale "adam": update RMS 0.2 so the AdamW lr band applies) on the muon*
    groups + fused AdamW on the rest, as one Combined optimizer. Prints the assignment."""
    import torch
    rows = report(model, rows)
    tkf = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")
    sys.path.insert(0, tkf)
    from kernels.sm120.muon import FusedMuon
    mats = [p for _n, p, g, _w in rows if g.startswith("muon")]
    rest = [p for _n, p, g, _w in rows if g == "adamw"]
    muon = FusedMuon([{"params": mats}], lr=muon_lr or lr, momentum=momentum,
                     weight_decay=wd if muon_wd is None else muon_wd, variant=variant, scale="adam", ns_coeffs=ns)
    adamw = torch.optim.AdamW(rest, lr=lr, betas=(0.9, 0.98), weight_decay=wd, fused=True)
    print(f"[muon_groups] optimizer: FusedMuon(variant={variant}, ns={ns}, lr={muon_lr or lr:g}, wd={wd if muon_wd is None else muon_wd:g}, "
          f"momentum={momentum}) on {len(mats)} tensors + fused AdamW(lr={lr:g}, wd={wd:g}) on {len(rest)}", flush=True)
    return Combined([muon, adamw])


def warmup_cosine(opt, warmup, max_steps, min_lr):
    """NeMo CosineAnnealing's shape per param group: linear warm-up, cosine to min_lr (absolute, as NeMo)."""
    import math
    import torch
    bases = [g["lr"] for g in opt.param_groups]

    def f(base):
        floor = min(min_lr / base, 1.0)

        def lam(t):
            if t < warmup:
                return (t + 1) / warmup
            x = min(1.0, (t - warmup) / max(1, max_steps - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * x))
        return lam
    return torch.optim.lr_scheduler.LambdaLR(opt, [f(b) for b in bases])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", default="stt_en_fastconformer_hybrid_large_streaming_multi")
    ap.add_argument("--tok", default=None)
    ap.add_argument("--ctc_head", default=None)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = (nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cpu") if a.nemo.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(a.nemo, map_location="cpu"))
    if a.tok:
        m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
    if a.ctc_head:
        import selfcond_head
        selfcond_head.enable(m, a.ctc_head)
    report(m)
