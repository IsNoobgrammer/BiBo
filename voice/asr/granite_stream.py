"""Streaming emulation of IBM Granite 5.0 TurboCTC (an offline, block-attention CTC model) on the meeting clips.

Every STEP seconds the encoder runs on a window [t - LEFT, t + STEP + R] of raw audio; the CTC frames whose time
falls in [t, t + STEP) are kept and appended (80 ms frames, 12.5 Hz). Display latency = STEP + R (+ compute).
Decoding: greedy over the appended per-frame log-probs with a blank penalty (blank = id 0), detokenised by the
processor's own CTC decoder (raw frame ids in: it groups repeats and drops blanks itself; collapsing first and
decoding again merged real repeated tokens, which cost 1.2 WER points in bench_asr's first version).

    PYTHONPATH=/home/marimo/hfpkgs python voice/asr/granite_stream.py --r 0.24 0.48 --out /home/marimo/work/asr/bench/granite_stream
Writes OUT/<tag>/{refs.jsonl, hyp.tsv} per (R, penalty) and a full-context reference (tag full) for the decode check;
score each with score_ctc_decode.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

A = "/home/marimo/work/asr"
MEET = (("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt"))
MID = "ibm-granite/granite-speech-5.0-470m-turboctc"
SR, FR = 16000, 0.08


def logprobs(m, proc, wavs):
    """Per-frame log-probs for a batch of equal-length raw-audio windows: [B, T, V] on CPU (float32)."""
    inp = proc(list(wavs), sampling_rate=SR, device=m.device)
    inp.to(m.device, dtype=m.dtype)
    with torch.inference_mode():
        return m(**inp).logits.float().log_softmax(-1).cpu()


def stream(m, proc, x, left, step, r, batch):
    """Appended log-probs [T, V] of one recording under the windowing above."""
    n, hop = len(x), int(step * SR)
    starts = list(range(0, n, hop))
    out = [None] * len(starts)
    for b0 in range(0, len(starts), batch):
        idx = starts[b0:b0 + batch]
        wins, offs = [], []
        for t in idx:
            a = max(0, t - int(left * SR)); e = min(n, t + hop + int(r * SR))
            wins.append(x[a:e]); offs.append((t - a) / SR)
        L = max(len(w) for w in wins)                        # pad on the right with silence to batch
        lp = logprobs(m, proc, [np.pad(w, (0, L - len(w))) for w in wins])
        for k, (t, off) in enumerate(zip(idx, offs)):
            f0 = int(round(off / FR)); f1 = f0 + int(round(min(hop, n - t) / SR / FR))
            out[b0 + k] = lp[k, f0:f1]
    return torch.cat(out)


def decode(proc, lp, bp):
    x = lp.clone(); x[:, 0] -= bp
    return proc.batch_decode(x.argmax(-1)[None].tolist(), skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, nargs="+", default=[0.0, 0.24, 0.48, 1.04], help="look-ahead seconds")
    ap.add_argument("--left", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.32)
    ap.add_argument("--bps", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained(MID)
    m = AutoModelForCTC.from_pretrained(MID, device_map="cuda", torch_dtype=torch.bfloat16).eval()
    clips = []
    for w, ref in MEET:
        x, sr = sf.read(f"{A}/eval_meeting/{w}", dtype="float32"); assert sr == SR
        clips.append((f"{A}/eval_meeting/{w}", x.mean(1) if x.ndim > 1 else x, open(f"{A}/eval_meeting/{ref}", encoding="utf-8").read()))

    def write(tag, hyps, mset):
        d = f"{a.out}/{tag}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/refs.jsonl", "w", encoding="utf-8") as f:
            for p, _, ref in clips:
                f.write(json.dumps({"file": p, "set": mset, "ref": ref}, ensure_ascii=False) + "\n")
        with open(f"{d}/hyp.tsv", "w", encoding="utf-8") as f:
            for (p, _, _), h in zip(clips, hyps):
                f.write(f"{p}\t{h}\n")

    # decode check: whole recording in one pass (the official generate path scored 13.76)
    full = [logprobs(m, proc, [x])[0] for _, x, _ in clips]
    for bp in a.bps:
        write(f"full_bp{bp:g}", [decode(proc, lp, bp) for lp in full], "meeting_full")
    print("GSTREAM full done", flush=True)
    for r in a.r:
        lps = [stream(m, proc, x, a.left, a.step, r, a.batch) for _, x, _ in clips]
        for bp in a.bps:
            write(f"r{int(r * 1000)}_bp{bp:g}", [decode(proc, lp, bp) for lp in lps], f"meeting_r{int(r * 1000)}")
        print(f"GSTREAM r {r} done (step {a.step}, left {a.left}, latency {a.step + r:.2f} s)", flush=True)
    print("GSTREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
