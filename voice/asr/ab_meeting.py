"""Meeting-clip WER (EVAL ONLY) of train_asr runs that carry a self-conditioned CTC head (--ctc_head), CTC and RNN-T.

The saved .nemo describes the stock CTC decoder, so NeMo's streaming script cannot load it: restore non-strict, put the
head back (selfcond_head.enable), then load the full state dict strictly. Encoding is ctc_heads.encode (the training
look-ahead mask, as head_rescore.py); CTC greedy with blank penalty + English lock, RNN-T greedy (hybrid runs) with the
same penalty / lock via blank_penalty.apply.

    python voice/asr/ab_meeting.py --nemo exp_ab/ab-ctc_only/ab-ctc_only.nemo --head glu_glu --bp 0 0.5 1.0
"""
import argparse
import json
import os
import sys
import tarfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blank_penalty  # noqa: E402
import selfcond_head  # noqa: E402
from ctc_heads import MEETING, encode, set_lookahead  # noqa: E402
from score import edits, normalize, normalize_v2, wer  # noqa: E402


def load(path, head, ckpt=None):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(path, map_location="cuda", strict=head == "none")
    if head != "none":                                  # "none": a stock-CTC run (run5), loaded as saved
        selfcond_head.enable(m, head)
        with tarfile.open(path) as t:
            sd = torch.load(t.extractfile(next(n for n in t.getnames() if n.endswith("model_weights.ckpt"))),
                            map_location="cuda", weights_only=False)
        m.load_state_dict(sd, strict=True)
    if ckpt:                                            # score another checkpoint of the same run (e.g. an earlier epoch)
        cks = ckpt if isinstance(ckpt, (list, tuple)) else [ckpt]
        sds = [torch.load(c, map_location="cpu", weights_only=False)["state_dict"] for c in cks]
        sd = sds[0] if len(sds) == 1 else {   # --ckpt A B C: uniform weight average (float tensors; buffers like
            k: (sum(d[k].float() for d in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else v   # step counts
            for k, v in sds[-1].items()}                                                                # from the last)
        m.load_state_dict(sd, strict=True)
    if hasattr(m, "joint"):       # RNN-T greedy: a captured CUDA graph would replay the first blank penalty forever
        from omegaconf import open_dict
        dec = m.cfg.decoding
        with open_dict(dec):
            dec.greedy.use_cuda_graph_decoder = False
        m.change_decoding_strategy(dec, decoder_type="rnnt")
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", nargs="+", required=True)
    ap.add_argument("--head", default="glu_glu", help="self-conditioned head kind, or none for a stock-CTC run (run5)")
    ap.add_argument("--ckpt", nargs="+", default=None,
                    help="Lightning .ckpt whose weights replace the .nemo's (same architecture); several = their average")
    ap.add_argument("--norm", choices=["v1", "v2"], default="v2",
                    help="v2 (default since Oct 12 2026): contractions expanded on both sides; v1 = the old numbers")
    ap.add_argument("--clips", nargs="+", default=["clip16k.wav", "g5_16k.wav"],
                    help="meeting clips to score; default the 2 distinct recordings (c2m.wav is an excerpt of "
                         "clip16k.wav, so it double-counted); 'all' = the old 3")
    ap.add_argument("--logits", default=None, help="save the CTC logits (per look-ahead / clip, fp16) to this .npz")
    ap.add_argument("--dump", default=None, help="write every CTC hypothesis (per look-ahead / penalty / clip) to this JSON")
    ap.add_argument("--rnnt", action="store_true", help="also decode the RNN-T head (hybrid runs)")
    ap.add_argument("--bp", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--las", type=int, nargs="+", default=[0, 1, 3, 6, 13])
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--tok", default="/home/marimo/work/asr/en3/tok/tokenizer_spe_bpe_v2047/tokenizer.model")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    global normalize
    normalize = normalize_v2 if a.norm == "v2" else normalize
    meeting = [(w, r) for w, r in MEETING if "all" in a.clips or w in a.clips]
    clips = [(sf.read(os.path.join(a.meeting, w))[0].astype(np.float32),
              normalize(open(os.path.join(a.meeting, r), encoding="utf-8").read())) for w, r in meeting]
    n = sum(len(r) for _, r in clips)

    def report(name, dec, la, bp, hyps):           # per-clip WER, pooled WER and its S / D / I split, hyp words
        sdi = [edits(ref, h) for (_, ref), h in zip(clips, hyps)]
        s, d, i = (sum(x[k] for x in sdi) for k in range(3))
        print(f"MEET {name:24s} {dec:4s} la {'full' if la < 0 else str(la * 80) + ' ms':>7s} bp {bp:.2f} | pooled {100 * (s + d + i) / n:5.2f} "
              f"(S {100 * s / n:5.2f} D {100 * d / n:5.2f} I {100 * i / n:5.2f}) | "
              + " ".join(f"{100 * wer(ref, h):5.1f}" for (_, ref), h in zip(clips, hyps))
              + f" | hyp/ref words {sum(map(len, hyps))}/{n}", flush=True)
    dump, saved = {}, {}
    for path in a.nemo:
        m = load(path, a.head, a.ckpt)
        name = os.path.basename(path)[:-5] + (f":avg{len(a.ckpt)}" if a.ckpt and len(a.ckpt) > 1 else
                                              ":" + os.path.basename(os.path.dirname(a.ckpt[0])) if a.ckpt else "")
        lock = blank_penalty.devanagari_ids(a.tok)
        blank = m.ctc_decoder.num_classes_with_blank - 1
        for la in a.las:
            set_lookahead(m.encoder, la)
            encs = [encode(m, torch.from_numpy(x)[None].cuda(), torch.tensor([len(x)]).cuda()) for x, _ in clips]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctc = (lambda e: m.ctc_decoder(encoder_output=e.transpose(1, 2))) if a.head == "none" else m.ctc_decoder.head
                zs = [ctc(e).float()[0, : int(el[0])] + lock for e, el in encs]
            if a.logits:
                saved.update({f"{name}|{la}|{i}": z.half().cpu().numpy() for i, z in enumerate(zs)})
            for bp in a.bp:
                hyps = []
                for z in zs:
                    z = z.clone()
                    z[:, blank] -= bp
                    ids = z.argmax(-1)
                    keep = (ids != blank) & (ids != F.pad(ids, (1, 0), value=-1)[:-1])
                    hyps.append(normalize(m.tokenizer.ids_to_text(ids[keep].tolist())))
                report(name, "ctc", la, bp, hyps)
                dump.setdefault(name, {}).setdefault(str(la), {})[f"{bp:.2f}"] = [" ".join(h) for h in hyps]
                if a.rnnt:
                    blank_penalty.apply(bp, head="rnnt", mask_ids=lock)
                    hyps = []
                    for e, el in encs:
                        hyp = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e.transpose(1, 2), encoded_lengths=el)
                        hyp = hyp[0] if isinstance(hyp, tuple) else hyp
                        hyps.append(normalize(hyp[0].text))
                    blank_penalty.apply(0.0, head="rnnt")
                    report(name, "rnnt", la, bp, hyps)
        del m
        torch.cuda.empty_cache()
    if a.logits:
        np.savez(a.logits, **saved, blank=np.array(blank), clips=np.array([w for w, _ in meeting]),
                 refs=np.array([" ".join(r) for _, r in clips]))
        print(f"logits -> {a.logits}", flush=True)
    if a.dump:
        json.dump({"refs": [" ".join(r) for _, r in clips], "clips": [w for w, _ in meeting], "hyps": dump},
                  open(a.dump, "w", encoding="utf-8"))
        print(f"dumped -> {a.dump}", flush=True)
    print("AB_MEETING_DONE", flush=True)


if __name__ == "__main__":
    main()
