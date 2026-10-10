"""Box check: SelfCondN.fused_loss (tkf selfcond_ctc_pass) == eager forward + ctc() for the whole multi-pass head --
loss and every parameter gradient -- then the per-step time of both at a 1,200 s batch.

    TKF=/home/marimo/work/tkf_ctc python voice/asr/test_fused_head.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import ctc_heads as ch  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
ok = True
for name, B, T, U in (("S2_glu_glu", 16, 101, 30), ("S2_lin_glu", 16, 101, 30), ("G3", 16, 101, 30)):
    V, d = 2049, 512
    head = (ch.make_head(name, d, V) if name != "G3" else ch.SelfCondN(d, V, ["lin", "glu", "glu"], (0.2, 0.3, 0.5))).to(dev)
    h = torch.randn(B, T, d, device=dev)
    hl = torch.randint(T // 2, T + 1, (B,), device=dev)
    hl[0] = T
    tlen = torch.randint(5, U + 1, (B,), device=dev)
    tgt = torch.randint(0, V - 1, (B, U), device=dev)
    res = {}
    for mode in ("eager", "fused"):
        head.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "fused":
                loss = head.fused_loss(h, hl, tgt, tlen, V - 1)
            else:
                loss = head.final_w * ch.ctc(head(h), hl, tgt, tlen, V - 1)
                for z, w in head.auxs:
                    loss = loss + w * ch.ctc(z, hl, tgt, tlen, V - 1)
        loss.backward()
        res[mode] = (loss.item(), {n: p.grad.detach().clone() for n, p in head.named_parameters()})
    le, lf = res["eager"][0], res["fused"][0]
    worst = max(((res["fused"][1][n] - g).norm() / g.norm().clamp(min=1e-12)).item() for n, g in res["eager"][1].items())
    good = abs(le - lf) / abs(le) < 2e-3 and worst < 3e-2
    ok &= good
    print(f"{name}: loss eager {le:.5f} fused {lf:.5f} | worst parameter-grad rel diff {worst:.2e} {'ok' if good else 'FAIL'}")

# speed: the whole head (2-pass GLU), B 150 x T 101 = 1,200 s of audio
B, T, U, V, d = 150, 101, 30, 2049, 512
head = ch.make_head("S2_glu_glu", d, V).to(dev)
h = torch.randn(B, T, d, device=dev)
hl = torch.full((B,), T, device=dev)
tlen = torch.full((B,), U, device=dev)
tgt = torch.randint(0, V - 1, (B, U), device=dev)


def step(mode):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if mode == "fused":
            loss = head.fused_loss(h, hl, tgt, tlen, V - 1)
        else:
            loss = head.final_w * ch.ctc(head(h), hl, tgt, tlen, V - 1)
            for z, w in head.auxs:
                loss = loss + w * ch.ctc(z, hl, tgt, tlen, V - 1)
    loss.backward()


for mode in ("eager", "fused"):
    for _ in range(3):
        step(mode)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(10):
        step(mode)
    torch.cuda.synchronize()
    print(f"S2_glu_glu {mode}: {1000 * (time.perf_counter() - t0) / 10:.2f} ms per 1,200 s batch (fwd+bwd, 2 CTC losses), "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")
print("FUSED_HEAD_OK" if ok else "FUSED_HEAD_FAIL")
