"""Does the Muown (+ AdamW) Combined optimizer resume EXACTLY? (train_asr --optim muown, muon_groups.build)

Run A: 10 steps straight. Run B: 5 steps, state_dict() -> torch.save/load (as a Lightning checkpoint does) -> FRESH
model copy + fresh optimizer + load_state_dict -> 5 more steps with the same grads. A and B must end bitwise equal.
The toy model has every Muon group the ASR model has (2D, pointwise conv muon_rc, LSTM per-gate muon_rc) + AdamW
params (bias, norm, embedding).

    TKF=/home/marimo/work/tkf_ctc /home/marimo/asrenv/bin/python voice/asr/test_optim_resume.py
"""
import copy
import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import muon_groups  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.ModuleDict({
            "lin": torch.nn.Linear(512, 512), "ff": torch.nn.Linear(512, 2048),
            "pw": torch.nn.Conv1d(512, 1024, 1), "norm": torch.nn.LayerNorm(512)})
        self.decoder = torch.nn.ModuleDict({"lstm": torch.nn.LSTM(640, 640), "embed": torch.nn.Embedding(128, 640)})


def grads(model, step):
    g = torch.Generator(device="cuda").manual_seed(1000 + step)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, device="cuda", generator=g) * 0.01


def run(model, opt, steps, start):
    for s in range(start, start + steps):
        grads(model, s)
        opt.step()
        opt.zero_grad()


def main():
    torch.manual_seed(0)
    base = Toy().cuda()
    a = copy.deepcopy(base)
    opt_a = muon_groups.build(a, 5e-4, wd=1e-3)
    run(a, opt_a, 10, 0)

    if os.environ.get("CONTROL"):                       # two fresh optimizers, 10 straight steps each: same bits?
        c = copy.deepcopy(base)
        opt_c = muon_groups.build(c, 5e-4, wd=1e-3)
        run(c, opt_c, 10, 0)
        dc = max((x - y).abs().max().item() for x, y in zip(a.parameters(), c.parameters()))
        print(f"CONTROL straight vs straight (fresh optimizer): worst {dc:.3e}", flush=True)
    b = copy.deepcopy(base)
    opt_b = muon_groups.build(b, 5e-4, wd=1e-3)
    run(b, opt_b, 5, 0)
    buf = io.BytesIO()
    torch.save({"opt": opt_b.state_dict(), "model": b.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)
    b2 = copy.deepcopy(base)
    b2.load_state_dict(ck["model"])
    opt_b2 = muon_groups.build(b2, 5e-4, wd=1e-3)
    opt_b2.load_state_dict(ck["opt"])
    if os.environ.get("FIX_STEP"):                      # diagnosis: restore FusedMuon's step counter by hand
        opt_b2.opts[0]._step_count = opt_b.opts[0]._step_count
    print("step counters: straight", [getattr(o, "_step_count", None) for o in opt_a.opts],
          "| resumed", [getattr(o, "_step_count", None) for o in opt_b2.opts], flush=True)
    if os.environ.get("DIAG"):                          # which FusedMuon state survives the round trip?
        o1, o2 = opt_b.opts[0], opt_b2.opts[0]
        p1 = [q for g in o1.param_groups for q in g["params"]]
        p2 = [q for g in o2.param_groups for q in g["params"]]
        def flat(st, pre=""):
            for k, v in st.items():
                if isinstance(v, dict):
                    yield from flat(v, pre + k + ".")
                else:
                    yield pre + k, v
        for i, (q1, q2) in enumerate(zip(p1, p2)):
            s1, s2 = dict(flat(o1.state.get(q1, {}))), dict(flat(o2.state.get(q2, {})))
            for k in sorted(set(s1) | set(s2)):
                a_, b_ = s1.get(k), s2.get(k)
                same = (a_ is not None and b_ is not None and (torch.equal(a_, b_) if torch.is_tensor(a_) else a_ == b_))
                dt = f" dtype {getattr(a_, 'dtype', None)} -> {getattr(b_, 'dtype', None)}"
                print(f"  param {i} {tuple(q1.shape)} {k}: {'same' if same else 'DIFFERS'}{dt}"
                      + ("" if same else f" ({type(a_).__name__} vs {type(b_).__name__})"), flush=True)
        extra = [k for k in vars(o1) if k.startswith("_") and not callable(getattr(o1, k))]
        print("  FusedMuon private attrs:", {k: type(getattr(o1, k)).__name__ for k in extra}, flush=True)
    run(b2, opt_b2, 5, 5)

    worst = 0.0
    for (n, pa), (_, pb) in zip(a.named_parameters(), b2.named_parameters()):
        d = (pa - pb).abs().max().item()
        worst = max(worst, d)
        if d > 0:
            print(f"  DIFF {n}: max |straight - resumed| = {d:.3e} (|w| {pa.abs().max().item():.3e})", flush=True)
    print("RESUME EXACT" if worst == 0 else f"RESUME MISMATCH: worst {worst:.3e}", flush=True)


if __name__ == "__main__":
    main()
