"""Why does a radial-readout CTC head drop fewer words? Compare saved heads (ctc_heads.py --save) frame by frame.

    python voice/asr/head_diag.py --heads asr/heads/g_vs_gr_seed24.pt --names G_sc3_mlp Gr_sc3_radial [--las 1 3 6]

Hypothesis: radial normsilu (silu(g/r) * r**p, p = sigmoid(theta)) scales the readout's hidden state by r**p instead
of ~r, so WEAK frames (small r = RMS of the final readout's pre-activation g) keep larger token logits -> blank wins
less often there -> fewer deletions. Per head, on the meeting clips: the learned p, then frames binned into deciles
of that head's own r with the mean P(blank) and the share of frames that emit a token. The hypothesis predicts a
LOWER P(blank) for the radial head in the low-r deciles, and about the same in the high ones.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctc_heads import MEETING, encode, make_head, set_lookahead  # noqa: E402


def readout(head):
    """The MLP whose activation differs between the SiLU and radial twins: the head's final readout."""
    for attr in ("r3", "mlp"):
        if hasattr(head, attr):
            return getattr(head, attr)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--las", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    torch.set_grad_enabled(False)
    ck = torch.load(a.heads, map_location="cpu", weights_only=False)
    m = nemo_asr.models.ASRModel.restore_from(ck["nemo"], map_location="cuda").eval()
    heads = {}
    for n in a.names:
        h = make_head(n, ck["d"], ck["c"])
        h.load_state_dict(ck["heads"][n])
        heads[n] = h.cuda().eval()
        thetas = [(k, torch.sigmoid(v).item()) for k, v in h.named_parameters() if k.endswith("radial_theta")]
        print(f"[p] {n}: " + (", ".join(f"{k.split('.')[0]} p={p:.3f}" for k, p in thetas) or "SiLU (no p)"), flush=True)
    blank = ck["c"] - 1
    clips = [sf.read(os.path.join(a.meeting, w))[0].astype(np.float32) for w, _ in MEETING]
    for r_la in a.las:
        set_lookahead(m.encoder, r_la)
        stats = {}
        for n, h in heads.items():
            rs, pb, emit = [], [], []
            cap = {}
            hook = readout(h).l1.register_forward_hook(lambda mod, i, o: cap.__setitem__("g", o))
            for x in clips:
                e, el = encode(m, torch.from_numpy(x)[None].cuda(), torch.tensor([len(x)]).cuda())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = h(e).float()[0, : int(el[0])]
                g = cap["g"].float()[0, : int(el[0])]
                rs.append(g.square().mean(-1).sqrt().cpu().numpy())
                p = z.softmax(-1)
                pb.append(p[:, blank].cpu().numpy())
                emit.append((z.argmax(-1) != blank).cpu().numpy())
            hook.remove()
            stats[n] = tuple(np.concatenate(v) for v in (rs, pb, emit))
        print(f"\n== look-ahead {r_la * 80} ms: {len(stats[a.names[0]][0])} frames over the 3 meeting clips")
        print("decile of r | " + " | ".join(f"{n}: r  P(blank)  emit%" for n in a.names))
        for q in range(10):
            row = []
            for n in a.names:
                r, pb, em = stats[n]
                lo, hi = np.quantile(r, [q / 10, (q + 1) / 10])
                sel = (r >= lo) & (r <= hi)
                row.append(f"{r[sel].mean():6.2f}  {pb[sel].mean():.3f}  {100 * em[sel].mean():5.1f}")
            print(f"   d{q}      | " + " | ".join(row))
        print("   all      | " + " | ".join(f"{s[0].mean():6.2f}  {s[1].mean():.3f}  {100 * s[2].mean():5.1f}" for s in stats.values()))
    print("HEAD_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
