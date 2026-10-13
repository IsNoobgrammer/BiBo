"""How far are a teacher's per-token KD targets from the transcript, in words? The targets (--kd_rnnt_mode pos:
teacher forced through the reference, blank excluded, frames marginalised) hold one distribution per reference token;
their argmax per position, detokenised and scored with score.normalize_v2 against the reference = the reconstructed
WER (length-preserving: substitutions, plus word merges / splits at piece boundaries). Low = the target repeats the
transcript; where it differs, it is what the teacher would have written there.

    TKF=... python voice/asr/teacher_recon.py --teacher nvidia/parakeet-tdt-0.6b-v2 --la -1 --n 512 --meeting
"""
import argparse
import json
import os
import random
import sys

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_kd
from score import edits, normalize_v2

A = "/home/marimo/work/asr"
MEET = (("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt"))


def targets(t, tdt, wavs, texts):
    from kernels.sm120.rnnt_joint import rnnt_joint_node_logp
    from kernels.sm120.tdt_joint import tdt_joint_pos_targets
    L = max(len(w) for w in wavs)
    sig = torch.zeros(len(wavs), L)
    for j, w in enumerate(wavs):
        sig[j, :len(w)] = torch.from_numpy(w)
    slen = torch.tensor([len(w) for w in wavs])
    ys = [t.tokenizer.text_to_ids(x) for x in texts]
    y = torch.zeros(len(ys), max(max(len(q) for q in ys), 1), dtype=torch.long)
    for j, q in enumerate(ys):
        y[j, :len(q)] = torch.tensor(q)
    ylen = torch.tensor([len(q) for q in ys])
    sig, slen, y, ylen = sig.cuda(), slen.cuda(), y.cuda(), ylen.cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc, elen = t.forward(input_signal=sig, input_signal_length=slen)
        dec, _, _ = t.decoder(targets=y, target_length=ylen)
        f = t.joint.project_encoder(enc.transpose(1, 2))
        g = t.joint.project_prednet(dec.transpose(1, 2))
    lin = t.joint.joint_net[-1]
    if tdt:
        P, _ = tdt_joint_pos_targets(f, g, lin.weight, lin.bias, y, elen, ylen, tdt)
    else:
        P = rnnt_joint_node_logp(f, g, lin.weight, lin.bias, y, elen, ylen, pos_targets=True)[4]
    return [t.tokenizer.ids_to_text(P[j, :len(ys[j])].argmax(-1).tolist()) for j in range(len(ys))]


def score(refs, hyps):
    s = d = i = n = 0
    for r, h in zip(refs, hyps):
        rw, hw = normalize_v2(r), normalize_v2(h)
        a, b, c = edits(rw, hw)
        s, d, i, n = s + a, d + b, i + c, n + len(rw)
    n = max(n, 1)
    return 100 * (s + d + i) / n, 100 * s / n, 100 * d / n, 100 * i / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", required=True, help="init[@ckpt]")
    ap.add_argument("--la", type=int, nargs="+", default=[3, -1])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--meeting", action="store_true", help="also the 2 meeting recordings (whole clip, one pass each)")
    a = ap.parse_args()
    sys.path.insert(0, teacher_kd.TKF)
    rows = [json.loads(l) for l in open(f"{A}/en3/train_cleanA.jsonl")]
    rows = [r for r in rows if r["text"].strip() and r["duration"] <= 20]
    random.Random(23).shuffle(rows)                  # the same clips as teacher_probe.py
    rows = sorted(rows[:a.n], key=lambda r: r["duration"])
    meet = [(sf.read(f"{A}/eval_meeting/{w}", dtype="float32")[0],
             " ".join(normalize_v2(open(f"{A}/eval_meeting/{r}", encoding="utf-8").read()))) for w, r in MEET]
    for spec in a.teacher:
        init, ckpt = (spec.split("@", 1) + [None])[:2]
        t = teacher_kd.load_teacher(init, ckpt, None if "parakeet" in init else [13, 6, 3, 1, 0, -1], fused=True)
        tdt = [int(d) for d in (t.cfg.model_defaults.get("tdt_durations", None) or [])]
        ctxs = {c[1]: list(c) for c in t.encoder.att_context_size_all}
        name = (ckpt.split(":")[-1].split("/")[0] if ckpt else init.split("/")[-1])[:40]
        for la in a.la:
            if tdt and la != -1:
                continue                                 # Parakeet: full attention only
            if not tdt:
                t.encoder.set_default_att_context_size(ctxs.get(la, [-1, -1] if la < 0 else [70, la]))
            hyps = []
            for i in range(0, len(rows), a.bs):
                bt = rows[i:i + a.bs]
                w = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in bt]
                hyps += targets(t, tdt, [x.mean(1) if x.ndim > 1 else x for x in w], [r["text"] for r in bt])
            w_, s_, d_, i_, n_ = score([r["text"] for r in rows], hyps)
            print(f"RECON {name:40s} la {la:3d} | train {len(rows)} clips {n_} words | WER {w_:.2f} "
                  f"(S {s_:.2f} D {d_:.2f} I {i_:.2f})", flush=True)
            if a.meeting:
                try:
                    mh = [targets(t, tdt, [x.mean(1) if x.ndim > 1 else x], [r])[0] for x, r in meet]
                    w_, s_, d_, i_, n_ = score([r for _, r in meet], mh)
                    print(f"RECON {name:40s} la {la:3d} | meeting 2 clips {n_} words | WER {w_:.2f} "
                          f"(S {s_:.2f} D {d_:.2f} I {i_:.2f})", flush=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"RECON {name} la {la} | meeting: out of memory (whole-clip pass)", flush=True)
        del t
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
