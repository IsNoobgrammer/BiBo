"""How much does a teacher's per-token target say beyond the transcript? On the same training clips, for each
teacher: the per-token-position targets used by --kd_rnnt_mode pos (teacher forced through the reference, blank
excluded, frames marginalised) and per reference token: p(reference token), top-1 == reference, entropy (nats),
mass on other tokens. Low p(ref) / high entropy = the teacher carries information the transcript loss does not.

    TKF=... python voice/asr/teacher_probe.py --manifest en3/train_cleanA.jsonl --n 512 \
        --teacher nvidia/nemotron-speech-streaming-en-0.6b[@hf:repo:ckpt] ... --la 3 -1
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_kd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--teacher", nargs="+", required=True, help="init[@ckpt], e.g. nvidia/x@hf:repo:path/last.ckpt")
    ap.add_argument("--la", type=int, nargs="+", default=[3, -1])
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    sys.path.insert(0, teacher_kd.TKF)
    from kernels.sm120.rnnt_joint import rnnt_joint_node_logp
    from kernels.sm120.tdt_joint import tdt_joint_pos_targets
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if r["text"].strip() and r["duration"] <= 20]
    random.Random(a.seed).shuffle(rows)
    rows = sorted(rows[:a.n], key=lambda r: r["duration"])
    for spec in a.teacher:
        init, ckpt = (spec.split("@", 1) + [None])[:2]
        t = teacher_kd.load_teacher(init, ckpt, None if "parakeet" in init else [13, 6, 3, 1, 0, -1], fused=True)
        tdt = [int(d) for d in (t.cfg.model_defaults.get("tdt_durations", None) or [])]
        ctxs = {c[1]: list(c) for c in t.encoder.att_context_size_all}
        for la in a.la:
            if tdt and la != -1:
                continue                                   # Parakeet: full attention only
            if not tdt:
                t.encoder.set_default_att_context_size(ctxs.get(la, [-1, -1] if la < 0 else [70, la]))
            pr, t1, ent, ntok = 0.0, 0, 0.0, 0
            for i in range(0, len(rows), a.bs):
                bt = rows[i:i + a.bs]
                wav = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in bt]
                wav = [w.mean(1) if w.ndim > 1 else w for w in wav]
                L = max(len(w) for w in wav)
                sig = torch.zeros(len(wav), L)
                for j, w in enumerate(wav):
                    sig[j, :len(w)] = torch.from_numpy(w)
                slen = torch.tensor([len(w) for w in wav])
                ys = [t.tokenizer.text_to_ids(r["text"]) for r in bt]
                U = max(len(y) for y in ys)
                y = torch.zeros(len(ys), max(U, 1), dtype=torch.long)
                for j, yy in enumerate(ys):
                    y[j, :len(yy)] = torch.tensor(yy)
                ylen = torch.tensor([len(yy) for yy in ys])
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
                for j in range(len(bt)):
                    u = int(ylen[j])
                    p = P[j, :u].double()
                    ref = y[j, :u]
                    pr += p.gather(1, ref[:, None]).sum().item()
                    t1 += (p.argmax(1) == ref).sum().item()
                    ent += -(p * p.clamp_min(1e-30).log()).sum().item()
                    ntok += u
            print(f"PROBE {spec.split('/')[-1][:60]:60s} la {la:3d} | tokens {ntok} | p(ref) {pr / ntok:.4f} | "
                  f"top1==ref {100 * t1 / ntok:.2f}% | other mass {1 - pr / ntok:.4f} | entropy {ent / ntok:.4f} nats",
                  flush=True)
        del t
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
