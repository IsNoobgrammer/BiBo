"""Does the Muown (+ AdamW) Combined optimizer resume EXACTLY? (train_asr --optim muown, muon_groups.build)

Run A: TOTAL (20) steps straight. Run B: SPLIT steps, state_dict() -> torch.save/load (as a Lightning checkpoint does)
-> FRESH model copy + fresh optimizer + load_state_dict -> the rest with the same grads. A and B must end bitwise equal.
SPLIT defaults to 12, past FusedMuon's 10-step NS-router probe window (a split inside it, e.g. SPLIT=5, never saw the
router's locked choices go missing on resume: Oct 11 2026, the e2e run diverged from its 2nd resumed step).
The toy model has every Muon group the ASR model has (2D, pointwise conv muon_rc, LSTM per-gate muon_rc) + AdamW
params (bias, norm, embedding).

    TKF=/home/marimo/work/tkf_ctc /home/marimo/asrenv/bin/python voice/asr/test_optim_resume.py
    XPROC=1 ...   the resumed half runs in a FRESH PROCESS, as a real resume does: per-process state (NS-router
                  decisions, kernel caches, cuDNN seeds) starts empty there and not in the in-process B above
"""
import copy
import io
import os
import subprocess
import sys
import tempfile

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


TOTAL, SPLIT = int(os.environ.get("TOTAL", 20)), int(os.environ.get("SPLIT", 12))


def child(path):
    """XPROC: the resumed half, in a fresh process -- load base + checkpoint, run the rest, compare with the straight run."""
    d = torch.load(path, weights_only=False)
    b2 = Toy().cuda()
    b2.load_state_dict(d["ck"]["model"])
    opt_b2 = muon_groups.build(b2, 5e-4, wd=1e-3)
    opt_b2.load_state_dict(d["ck"]["opt"])
    run(b2, opt_b2, TOTAL - SPLIT, SPLIT)
    return compare(d["a"], b2)


def compare(a_sd, b2):
    worst = 0.0
    for n, pb in b2.named_parameters():
        d = (a_sd[n] - pb).abs().max().item()
        worst = max(worst, d)
        if d > 0:
            print(f"  DIFF {n}: max |straight - resumed| = {d:.3e} (|w| {a_sd[n].abs().max().item():.3e})", flush=True)
    print("RESUME EXACT" if worst == 0 else f"RESUME MISMATCH: worst {worst:.3e}", flush=True)
    return worst


def main():
    if os.environ.get("XPROC_CHILD"):
        sys.exit(0 if child(os.environ["XPROC_CHILD"]) == 0 else 1)
    torch.manual_seed(0)
    base = Toy().cuda()
    a = copy.deepcopy(base)
    opt_a = muon_groups.build(a, 5e-4, wd=1e-3)
    run(a, opt_a, TOTAL, 0)

    if os.environ.get("CONTROL"):                       # two fresh optimizers, 10 straight steps each: same bits?
        c = copy.deepcopy(base)
        opt_c = muon_groups.build(c, 5e-4, wd=1e-3)
        run(c, opt_c, TOTAL, 0)
        dc = max((x - y).abs().max().item() for x, y in zip(a.parameters(), c.parameters()))
        print(f"CONTROL straight vs straight (fresh optimizer, {TOTAL} steps): worst {dc:.3e}", flush=True)
    b = copy.deepcopy(base)
    opt_b = muon_groups.build(b, 5e-4, wd=1e-3)
    run(b, opt_b, SPLIT, 0)
    buf = io.BytesIO()
    torch.save({"opt": opt_b.state_dict(), "model": b.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)
    if os.environ.get("XPROC"):
        path = os.path.join(tempfile.mkdtemp(), "xproc.pt")
        torch.save({"ck": ck, "a": {n: q.detach().clone() for n, q in a.named_parameters()}}, path)
        print(f"XPROC: resuming at step {SPLIT} in a fresh process", flush=True)
        sys.exit(subprocess.run([sys.executable, os.path.abspath(__file__)],
                                env={**os.environ, "XPROC_CHILD": path}).returncode)
    if os.environ.get("LEGACY"):                        # a checkpoint from before tkf saved FusedMuon's step counter
        for s in ck["opt"]["combined"]:
            s.pop("fused_muon", None)
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
    run(b2, opt_b2, TOTAL - SPLIT, SPLIT)
    worst = compare({n: q.detach() for n, q in a.named_parameters()}, b2)
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
