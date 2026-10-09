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
"""
import runpy
import sys


def _patch(cls, name, delta):
    key = f"_bp_orig_{name}"
    if getattr(cls, key, None) is None:
        setattr(cls, key, getattr(cls, name))
    orig = getattr(cls, key)
    if not delta:
        setattr(cls, name, orig)
        return

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if self.training:
            return out
        out = out.clone()
        out[..., -1] -= delta
        return out

    setattr(cls, name, wrapped)


def apply(delta, head="rnnt"):
    if head == "ctc":
        from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
        _patch(ConvASRDecoder, "forward", delta)
    else:
        from nemo.collections.asr.modules.rnnt import RNNTJoint
        _patch(RNNTJoint, "joint_after_projection", delta)


if __name__ == "__main__":
    spec = sys.argv[1]
    head, _, d = spec.rpartition(":")
    apply(float(d), head=head or "rnnt")
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")
