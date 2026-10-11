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
