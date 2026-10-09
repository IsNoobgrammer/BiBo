"""Decode-time blank penalty (inference only): subtract `delta` from the blank score of every frame / joint output.

A model drops a word by choosing blank where a token belonged; a penalty on blank trades deletions for insertions.
  RNN-T: RNNTJoint.joint_after_projection output (blank = last class)
  CTC:   ConvASRDecoder output log-probs (blank = last class), with head="ctc"
Patched at the CLASS level, so it reaches greedy / batched / streaming decoding inside any script -- NeMo's
cache-aware streaming example included:

    import blank_penalty; blank_penalty.apply(0.5)            # RNN-T
    import blank_penalty; blank_penalty.apply(0.3, head="ctc")
    python voice/asr/blank_penalty.py 0.5 NeMo/examples/.../speech_to_text_cache_aware_streaming_infer.py args...
    python voice/asr/blank_penalty.py ctc:0.3 ...               # CTC head
    python voice/asr/blank_penalty.py ctc:0.5:en ...            # + English script lock (no Devanagari tokens)
"""
import os
import runpy
import sys


def _patch(cls, name, delta, mask_ids=None):
    key = f"_bp_orig_{name}"
    if getattr(cls, key, None) is None:
        setattr(cls, key, getattr(cls, name))
    orig = getattr(cls, key)
    if not delta and mask_ids is None:
        setattr(cls, name, orig)
        return

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if self.training:
            return out
        out = out.clone()
        out[..., -1] -= delta
        if mask_ids is not None:                  # script lock: these tokens can never be emitted
            # pinned + non_blocking: RNN-T greedy decoding runs under CUDA-graph capture, which refuses pageable copies
            out[..., mask_ids.to(out.device, non_blocking=True)] = -1e4
        return out

    setattr(cls, name, wrapped)


def devanagari_ids(tok_model):
    """Token ids whose piece contains a Devanagari character (U+0900-U+097F): the English-mode script lock, the
    same rule as RT Captions (on our meeting clips run5's CTC head emitted ~50 Devanagari words on g5, refs have 0)."""
    import sentencepiece as spm
    import torch
    sp = spm.SentencePieceProcessor(model_file=tok_model)
    ids = torch.tensor([i for i in range(sp.get_piece_size())
                        if any("ऀ" <= c <= "ॿ" for c in sp.id_to_piece(i))])
    return ids.pin_memory() if torch.cuda.is_available() else ids


def apply(delta, head="rnnt", mask_ids=None):
    if head == "ctc":
        from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
        _patch(ConvASRDecoder, "forward", delta, mask_ids)
    else:
        from nemo.collections.asr.modules.rnnt import RNNTJoint
        _patch(RNNTJoint, "joint_after_projection", delta, mask_ids)


if __name__ == "__main__":
    # spec: [head:]delta[:en]   e.g. 0.5 / ctc:0.5 / rnnt:0:en (":en" = English script lock, needs ASR_TOK or the default)
    parts = sys.argv[1].split(":")
    lock = parts[-1] == "en" and parts.pop()
    head, d = (parts[0], parts[1]) if len(parts) == 2 else ("rnnt", parts[0])
    tok = os.environ.get("ASR_TOK", "/home/marimo/work/asr/run1/tok/tokenizer_spe_bpe_v4096/tokenizer.model")
    apply(float(d), head=head, mask_ids=devanagari_ids(tok) if lock else None)
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")
