"""RNN-T decode-time blank penalty: subtract `delta` from the blank logit of every joint output (inference only).

An RNN-T drops a word by choosing blank where a token belonged; a penalty on blank trades deletions for insertions.
Patched at the CLASS level (RNNTJoint.joint_after_projection), so it reaches greedy / batched / streaming decoding
inside any script -- NeMo's cache-aware streaming example included:

    import blank_penalty; blank_penalty.apply(1.0)
    python voice/asr/blank_penalty.py 1.0 NeMo/examples/.../speech_to_text_cache_aware_streaming_infer.py args...
"""
import runpy
import sys


def apply(delta):
    from nemo.collections.asr.modules.rnnt import RNNTJoint
    if getattr(RNNTJoint, "_bp_orig", None) is None:
        RNNTJoint._bp_orig = RNNTJoint.joint_after_projection
    orig = RNNTJoint._bp_orig
    if not delta:
        RNNTJoint.joint_after_projection = orig
        return

    def joint_after_projection(self, f, g):
        out = orig(self, f, g)
        if self.training:
            return out
        out = out.clone()
        out[..., -1] -= delta                       # blank = the last class (NeMo RNNT convention)
        return out

    RNNTJoint.joint_after_projection = joint_after_projection


if __name__ == "__main__":
    apply(float(sys.argv[1]))
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")
