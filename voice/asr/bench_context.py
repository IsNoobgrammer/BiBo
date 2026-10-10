"""Training cost per look-ahead: encoder fwd+bwd on one 1,200 s batch (80 x 15 s), run5 kernels, synthetic audio.

    python voice/asr/bench_context.py --nemo asr/exp/run5/run5.eval.nemo

Contexts: the trained [70, r] set, full context through NeMo's attention ([-1, -1]: fused_attn falls back) and full
context through the fused kernel as ONE chunk ([0, T-1]: chunked_limited with a chunk covering the utterance = full
attention); the last two must agree (parity printed).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", default="stt_en_fastconformer_hybrid_large_streaming_multi",
                    help=".nemo path or a pretrained name (throughput depends only on the architecture)")
    ap.add_argument("--utts", type=int, default=80)
    ap.add_argument("--sec", type=float, default=15.0)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    import fused_attn
    import fused_conv
    import fused_layer
    m = (nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cuda") if a.nemo.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(a.nemo, map_location="cuda")).cuda().train()   # same encoder
    for k in (fused_layer, fused_attn, fused_conv):
        k.enable(m)
    enc = m.encoder
    torch.manual_seed(0)
    n = int(a.sec * 16000)
    audio = torch.randn(a.utts, n, device="cuda") * 0.05
    alen = torch.full((a.utts,), n, device="cuda")
    with torch.no_grad():
        f, fl = m.preprocessor(input_signal=audio, length=alen)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        T = enc(audio_signal=f, length=fl)[0].shape[-1]   # the encoder's real frame count (not a guess: +-1 frame
                                                          # puts the last frame in its own chunk)

    def step(ctx, keep=False):
        enc.att_context_size_all = [ctx]           # sampled context = this one
        enc.set_default_att_context_size(ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e, el = enc(audio_signal=f, length=fl)
        loss = (e.float() * torch.linspace(-1, 1, e.shape[1], device="cuda")[None, :, None]).mean()
        loss.backward()
        out = e.detach().float() if keep else None
        g = enc.layers[8].self_attn.linear_q.weight.grad.detach().float().clone() if keep else None
        enc.zero_grad(set_to_none=True)
        return out, g

    def bench(ctx):
        for _ in range(10):                        # Triton autotune runs once per new context shape
            step(ctx)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(a.iters):
            step(ctx)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / a.iters * 1000, torch.cuda.max_memory_allocated() / 2**30

    print(f"batch {a.utts} x {a.sec:.0f} s = {a.utts * a.sec:.0f} s of audio, T = {T} encoder frames", flush=True)
    rows = [("80 ms   [70,1]", [70, 1]), ("240 ms  [68,3]", [68, 3]), ("480 ms  [70,6]", [70, 6]),
            ("1040 ms [70,13]", [70, 13]), ("full, NeMo attention [-1,-1]", [-1, -1]),
            ("full, fused one chunk [0,T-1]", [0, T - 1])]
    for name, ctx in rows:
        bench(ctx)                                 # pass 1: warm every shape
    for rep in (1, 2):
        for name, ctx in rows:
            ms, gb = bench(ctx)
            print(f"rep {rep}  {name:32s} {ms:7.1f} ms/step  {a.utts * a.sec / ms * 1000:8.0f} audio-s/s (encoder fwd+bwd)  peak {gb:5.1f} GB", flush=True)
    m.eval()                                       # dropout off: the two paths must match exactly-ish
    (e1, g1), (e2, g2) = step([-1, -1], keep=True), step([0, T - 1], keep=True)
    rel = lambda x, y: ((x - y).norm() / y.norm()).item()
    print(f"parity full NeMo vs fused one-chunk: out {rel(e2, e1):.2e}  dWq(layer 8) {rel(g2, g1):.2e}", flush=True)


if __name__ == "__main__":
    main()
