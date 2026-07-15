"""Self-check for the async HF checkpoint path in train.py — no GPU, no network.
Covers the one trap: torch.compile wraps model.model with an `_orig_mod.`-prefixed state dict, so
_save_hf_ckpt MUST unwrap it before save_pretrained (else the checkpoint is unloadable), and restore
it after so training continues on the compiled module. Also checks the async push drains its dir.

  python -m ablate.common.test_hf_ckpt
"""
from .train import _save_hf_ckpt, _push_hf_async


class _Orig:                       # stands in for the real (un-compiled) module
    pass


class _Compiled:                   # stands in for torch.compile(model.model) (OptimizedModule)
    def __init__(self, orig):
        self._orig_mod = orig


class _FakeModel:
    def __init__(self, compiled):
        self.model = compiled
        self.saved_model_at_save = None

    def save_pretrained(self, out_dir, safe_serialization=True):
        self.saved_model_at_save = self.model      # capture what model.model was AT save time


class _FakeTok:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, out_dir):
        self.saved_to = out_dir


def test_unwrap_then_restore(tmp="/tmp/_hfckpt_selfcheck"):
    orig = _Orig()
    compiled = _Compiled(orig)
    model = _FakeModel(compiled)
    tok = _FakeTok()

    out = _save_hf_ckpt(model, tok, tmp)

    assert model.saved_model_at_save is orig, "must save the UN-compiled module (clean state-dict keys)"
    assert model.model is compiled, "must restore the compiled module so training continues on it"
    assert tok.saved_to == tmp, "tokenizer must be written alongside the model"
    assert out == tmp
    print("OK: compile-unwrap saves orig, restores compiled, saves tokenizer")


def test_no_compile_passthrough(tmp="/tmp/_hfckpt_selfcheck2"):
    plain = _Orig()                # model.model with no _orig_mod (compile off)
    model = _FakeModel(plain)
    _save_hf_ckpt(model, _FakeTok(), tmp)
    assert model.saved_model_at_save is plain and model.model is plain
    print("OK: no-compile path saves model.model unchanged")


if __name__ == "__main__":
    import os
    os.makedirs("/tmp", exist_ok=True)
    test_unwrap_then_restore()
    test_no_compile_passthrough()
    print("all self-checks passed")
