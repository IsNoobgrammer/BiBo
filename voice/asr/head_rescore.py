"""Rescore saved CTC heads (ctc_heads.py --save) on the meeting clips with decode-time blank penalties.

    python voice/asr/head_rescore.py --heads asr/heads/g_vs_gr_seed24.pt --names G_sc3_mlp Gr_sc3_radial --bp 0 0.5 1.0

Greedy CTC with `delta` subtracted from the blank logit (blank_penalty.py's rule, in-process), every look-ahead.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctc_heads import MEETING, encode, make_head, set_lookahead  # noqa: E402
from score import normalize, wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--bp", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--las", type=int, nargs="+", default=[0, 1, 3, 6, 13])
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    torch.set_grad_enabled(False)
    ck = torch.load(a.heads, map_location="cpu", weights_only=False)
    m = nemo_asr.models.ASRModel.restore_from(ck["nemo"], map_location="cuda").eval()
    if ck.get("tok"):
        m.change_vocabulary(new_tokenizer_dir=ck["tok"], new_tokenizer_type="bpe")
    heads = {}
    for n in a.names:
        h = make_head(n, ck["d"], ck["c"])
        h.load_state_dict(ck["heads"][n])
        heads[n] = h.cuda().eval()
    blank = ck["c"] - 1
    clips = [(sf.read(os.path.join(a.meeting, w))[0].astype(np.float32),
              normalize(open(os.path.join(a.meeting, r), encoding="utf-8").read())) for w, r in MEETING]
    words = [len(r) for _, r in clips]
    for la in a.las:
        set_lookahead(m.encoder, la)
        encs = [encode(m, torch.from_numpy(x)[None].cuda(), torch.tensor([len(x)]).cuda()) for x, _ in clips]
        for n, h in heads.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zs = [h(e).float()[0, : int(el[0])] for e, el in encs]
            for bp in a.bp:
                res = []
                for z, (_, ref) in zip(zs, clips):
                    z = z.clone()
                    z[:, blank] -= bp
                    ids = z.argmax(-1)
                    keep = (ids != blank) & (ids != F.pad(ids, (1, 0), value=-1)[:-1])
                    hyp = normalize(m.tokenizer.ids_to_text(ids[keep].tolist()))
                    res.append((wer(ref, hyp), len(hyp)))
                pooled = sum(w * k for (w, _), k in zip(res, words)) / sum(words)
                print(f"RESULT look-ahead {la * 80:4d} ms {n:14s} bp {bp:.1f} | meeting pooled {100 * pooled:5.2f} | " +
                      " ".join(f"{100 * w:5.1f}({k}/{r})" for (w, k), r in zip(res, words)), flush=True)
    print("RESCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
