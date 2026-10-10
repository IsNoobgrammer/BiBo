"""Meeting-clip WER (EVAL ONLY) of train_asr runs that carry a self-conditioned CTC head (--ctc_head), CTC and RNN-T.

The saved .nemo describes the stock CTC decoder, so NeMo's streaming script cannot load it: restore non-strict, put the
head back (selfcond_head.enable), then load the full state dict strictly. Encoding is ctc_heads.encode (the training
look-ahead mask, as head_rescore.py); CTC greedy with blank penalty + English lock, RNN-T greedy (hybrid runs) with the
same penalty / lock via blank_penalty.apply.

    python voice/asr/ab_meeting.py --nemo exp_ab/ab-ctc_only/ab-ctc_only.nemo --head glu_glu --bp 0 0.5 1.0
"""
import argparse
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
from score import normalize, wer  # noqa: E402


def load(path, head):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(path, map_location="cuda", strict=False)
    selfcond_head.enable(m, head)
    with tarfile.open(path) as t:
        sd = torch.load(t.extractfile(next(n for n in t.getnames() if n.endswith("model_weights.ckpt"))),
                        map_location="cuda", weights_only=False)
    m.load_state_dict(sd, strict=True)
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", nargs="+", required=True)
    ap.add_argument("--head", default="glu_glu")
    ap.add_argument("--rnnt", action="store_true", help="also decode the RNN-T head (hybrid runs)")
    ap.add_argument("--bp", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--las", type=int, nargs="+", default=[0, 1, 3, 6, 13])
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--tok", default="/home/marimo/work/asr/en2/tok/tokenizer_spe_bpe_v2047/tokenizer.model")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    clips = [(sf.read(os.path.join(a.meeting, w))[0].astype(np.float32),
              normalize(open(os.path.join(a.meeting, r), encoding="utf-8").read())) for w, r in MEETING]
    words = [len(r) for _, r in clips]
    pooled = lambda res: 100 * sum(w * k for w, k in zip(res, words)) / sum(words)
    for path in a.nemo:
        m = load(path, a.head)
        name = os.path.basename(path)[:-5]
        lock = blank_penalty.devanagari_ids(a.tok)
        blank = m.ctc_decoder.num_classes_with_blank - 1
        for la in a.las:
            set_lookahead(m.encoder, la)
            encs = [encode(m, torch.from_numpy(x)[None].cuda(), torch.tensor([len(x)]).cuda()) for x, _ in clips]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zs = [m.ctc_decoder.head(e).float()[0, : int(el[0])] + lock for e, el in encs]
            for bp in a.bp:
                res = []
                for z, (_, ref) in zip(zs, clips):
                    z = z.clone()
                    z[:, blank] -= bp
                    ids = z.argmax(-1)
                    keep = (ids != blank) & (ids != F.pad(ids, (1, 0), value=-1)[:-1])
                    res.append(wer(ref, normalize(m.tokenizer.ids_to_text(ids[keep].tolist()))))
                print(f"MEET {name:22s} ctc  la {la * 80:4d} ms bp {bp:.1f} | pooled {pooled(res):5.2f} | " +
                      " ".join(f"{100 * w:5.1f}" for w in res), flush=True)
                if a.rnnt:
                    blank_penalty.apply(bp, head="rnnt", mask_ids=lock)
                    res = []
                    for (e, el), (_, ref) in zip(encs, clips):
                        hyp = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e.transpose(1, 2), encoded_lengths=el)
                        hyp = hyp[0] if isinstance(hyp, tuple) else hyp
                        res.append(wer(ref, normalize(hyp[0].text)))
                    blank_penalty.apply(0.0, head="rnnt")
                    print(f"MEET {name:22s} rnnt la {la * 80:4d} ms bp {bp:.1f} | pooled {pooled(res):5.2f} | " +
                          " ".join(f"{100 * w:5.1f}" for w in res), flush=True)
        del m
        torch.cuda.empty_cache()
    print("AB_MEETING_DONE", flush=True)


if __name__ == "__main__":
    main()
