"""Which parameter of the ASR model goes to Muon(own) and which to AdamW -- the rule in ONE place, printable.

Muon orthogonalises the TRAILING 2D slice of a parameter and batches over any leading dim (kernels/sm75/muon.py
buckets by shape[-2:]), so the stored shape decides what gets decorrelated. Rules (Muon convention: hidden-layer
weight MATRICES on Muon; embeddings, output heads, norms, biases, per-channel filters on AdamW):
  muon        2D Linear weights inside the encoder / pre-encode / joint projections / prediction-net LSTM
  muon_flat   Conv1d / Conv2d weights with kernel 1 (pointwise convs): really (out, in) matrices stored 3D / 4D --
              they MUST reach Muon as 2D (out, in), or each output row becomes its own (in, 1) "matrix"
  adamw       1D (norms, biases, BatchNorm), depthwise convs (out, 1, k) (per-channel filters, not matrices),
              strided 2D convs of the pre-encode (out, in, 3, 3), embeddings, vocab output heads
              (CTC head readout, joint output layer, self-conditioning feedback embeddings)

    python voice/asr/muon_groups.py --nemo stt_en_fastconformer_hybrid_large_streaming_multi --tok .../v2047 --ctc_head glu_glu
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HEAD_NAMES = ("joint.joint_net.", "ctc_decoder.", "decoder.prediction.embed")   # vocab-sized in/out layers


def assign(model):
    """[(name, param, group, why)] for every trainable parameter."""
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            g, why = "adamw", "1D (norm / bias)"
        elif any(n.startswith(h) for h in HEAD_NAMES) and not (n.startswith("ctc_decoder.") and _is_glu_hidden(n, p)):
            g, why = "adamw", "vocab head / embedding / feedback"
        elif p.ndim == 2:
            g, why = "muon", "2D matrix"
        elif p.ndim in (3, 4) and all(s == 1 for s in p.shape[2:]):
            g, why = "muon_flat", f"pointwise conv {tuple(p.shape)} -> ({p.shape[0]}, {p.shape[1]})"
        elif p.ndim == 3 and p.shape[1] == 1:
            g, why = "adamw", "depthwise conv (per-channel filter)"
        else:
            g, why = "adamw", f"strided / spatial conv {tuple(p.shape)}"
        out.append((n, p, g, why))
    return out


def _is_glu_hidden(n, p):
    """Inside the self-conditioned CTC head, the GLU readouts' hidden matrices (d -> 8d/3 -> d) are encoder-like
    hidden layers (Muon); its vocab projection and feedback embeddings are heads (AdamW)."""
    return p.ndim == 2 and p.shape[0] < 4000 and p.shape[1] < 4000 and min(p.shape) >= 256 and ".out." not in n \
        and ".cur." not in n and ".prev." not in n and not n.endswith(".r.0.weight")


def report(model):
    rows = assign(model)
    tot = collections.Counter()
    print(f"{'group':10s} {'shape':>22s} {'params':>11s}  name  (why)")
    for n, p, g, why in rows:
        tot[g] += p.numel()
        print(f"{g:10s} {str(tuple(p.shape)):>22s} {p.numel():11,d}  {n}  ({why})")
    allp = sum(tot.values())
    print("\nTOTALS: " + " | ".join(f"{g} {v / 1e6:.2f}M ({100 * v / allp:.1f}%)" for g, v in tot.most_common())
          + f" | all {allp / 1e6:.2f}M", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", default="stt_en_fastconformer_hybrid_large_streaming_multi")
    ap.add_argument("--tok", default=None)
    ap.add_argument("--ctc_head", default=None)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = (nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cpu") if a.nemo.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(a.nemo, map_location="cpu"))
    if a.tok:
        m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
    if a.ctc_head:
        import selfcond_head
        selfcond_head.enable(m, a.ctc_head)
    report(m)
