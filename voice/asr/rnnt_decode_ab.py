"""RNN-T decoding A/B on the meeting clips: greedy vs beam search, with / without internal-LM subtraction.

    python voice/asr/rnnt_decode_ab.py --nemo exp/run5/run5.eval.nemo [--las 1 3 13] [--bp 0.5]

NeMo's beam decoders cannot carry partial hypotheses across streaming chunks, so every variant decodes the encoder
output of the WHOLE clip under the chunked look-ahead mask -- for a cache-aware encoder that is exactly what the
streaming loop computes -- and greedy goes through the same path as the control. Blank penalty --bp and the English
script lock on every variant (blank_penalty.py patches the joint, so beam search sees them too).
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blank_penalty as BP  # noqa: E402
from ctc_heads import MEETING, set_lookahead  # noqa: E402
from score import normalize, wer  # noqa: E402

VARIANTS = [("greedy", 0.0), ("greedy", 0.1), ("greedy", 0.2), ("greedy", 0.3), ("beam4", 0.0), ("beam4", 0.2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--tok", default="/home/marimo/work/asr/run1/tok/tokenizer_spe_bpe_v4096/tokenizer.model")
    ap.add_argument("--las", type=int, nargs="+", default=[1, 3, 13])
    ap.add_argument("--bp", type=float, default=0.5)
    ap.add_argument("--variants", nargs="+", default=None, help="e.g. beam4:0 beam4:0.2 (default: all)")
    a = ap.parse_args()
    import copy
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    torch.set_grad_enabled(False)
    m = nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cuda").eval()
    lock = BP.devanagari_ids(a.tok)
    clips = [(sf.read(os.path.join(a.meeting, w))[0].astype(np.float32),
              normalize(open(os.path.join(a.meeting, r), encoding="utf-8").read())) for w, r in MEETING]
    words = [len(r) for _, r in clips]
    for r in a.las:
        set_lookahead(m.encoder, r)
        encs = []
        for x, _ in clips:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f, fl = m.preprocessor(input_signal=torch.from_numpy(x)[None].cuda(), length=torch.tensor([len(x)]).cuda())
                e, el = m.encoder(audio_signal=f, length=fl)
            encs.append((e.float(), el))
        for strategy, ilm in ([(v.split(":")[0], float(v.split(":")[1])) for v in a.variants] if a.variants else VARIANTS):
            BP.apply(a.bp, "rnnt", mask_ids=lock, ilm=ilm)
            cfg = copy.deepcopy(m.cfg.decoding)
            with open_dict(cfg):
                cfg.strategy = "greedy_batch" if strategy == "greedy" else "malsd_batch"
                cfg.beam.beam_size = 4
            m.change_decoding_strategy(cfg, decoder_type="rnnt")
            res = []
            for (e, el), (_, ref) in zip(encs, clips):
                out = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e, encoded_lengths=el)
                out = out[0] if isinstance(out, tuple) else out
                h = out[0][0] if isinstance(out[0], list) else out[0]
                h = h.n_best_hypotheses[0] if hasattr(h, "n_best_hypotheses") else h       # beam: best of the n-best
                text = h.text if hasattr(h, "text") else h
                if not isinstance(text, str):                                               # token ids
                    text = m.tokenizer.ids_to_text([int(i) for i in h.y_sequence])
                hyp = normalize(text)
                res.append((wer(ref, hyp), len(hyp)))
            pooled = sum(w * n for (w, _), n in zip(res, words)) / sum(words)
            print(f"RESULT look-ahead {r * 80:4d} ms {strategy:6s} ILM {ilm:.1f} | pooled {100 * pooled:5.2f} | " +
                  " ".join(f"{100 * w:5.1f}({n}/{ref_n})" for (w, n), ref_n in zip(res, words)), flush=True)
    print("DECODE_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
