"""Self-check for the async HF checkpoint path in train.py — no GPU, no network.
Covers the traps: (1) torch.compile wraps model.model with an `_orig_mod.`-prefixed state dict, so
_save_hf_ckpt MUST unwrap before save (else unloadable) and restore after so training continues on the
compiled module; (2) bf16 save casts WEIGHTS only in a COPY (live fp32 params + buffers untouched) and
restores config.torch_dtype.

  python -m ablate.common.test_hf_ckpt
"""
import torch
from .train import _save_hf_ckpt, _eager


class _Orig:                       # stands in for the real (un-compiled) module
    pass


class _Compiled:                   # stands in for torch.compile(model.model) (OptimizedModule)
    def __init__(self, orig):
        self._orig_mod = orig


class _Cfg:
    def __init__(self):
        self.torch_dtype = torch.float32


class _FakeModel:
    """One fp32 weight `w` and one fp32 buffer `buf`; live tensors must survive save untouched."""
    def __init__(self, compiled):
        self.model = compiled
        self.config = _Cfg()
        self.w = torch.ones(4, 4, dtype=torch.float32)       # a 2D matrix param -> bf16
        self.norm = torch.ones(4, dtype=torch.float32)       # a 1D RMSNorm gain -> stays fp32
        self.buf = torch.ones(2, dtype=torch.float32) * 3    # a buffer (e.g. RoPE inv_freq) -> stays fp32
        self.saved_model_at_save = None
        self.saved_sd = None
        self.dtype_at_save = None

    def named_parameters(self):
        return [("w", self.w), ("norm", self.norm)]

    def state_dict(self):
        return {"w": self.w, "norm": self.norm, "buf": self.buf}

    def save_pretrained(self, out_dir, state_dict=None, safe_serialization=True):
        self.saved_model_at_save = self.model
        self.saved_sd = state_dict
        self.dtype_at_save = self.config.torch_dtype


class _FakeTok:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, out_dir):
        self.saved_to = out_dir


def test_unwrap_bf16_restore(tmp="/tmp/_hfckpt_selfcheck"):
    orig = _Orig()
    model = _FakeModel(_Compiled(orig))
    tok = _FakeTok()
    _save_hf_ckpt(model, tok, tmp)

    assert model.saved_model_at_save is orig, "must save the UN-compiled module (clean keys)"
    assert isinstance(model.model, _Compiled), "must restore the compiled module"
    assert model.saved_sd["w"].dtype == torch.bfloat16, "2D matrix must be saved as bf16"
    assert model.saved_sd["norm"].dtype == torch.float32, "1D norm gain must stay fp32"
    assert model.saved_sd["buf"].dtype == torch.float32, "buffer must stay full precision"
    assert model.w.dtype == torch.float32, "LIVE weight must remain fp32 (master weights untouched)"
    assert model.dtype_at_save == torch.bfloat16, "config.torch_dtype must be bf16 at save time"
    assert model.config.torch_dtype == torch.float32, "config.torch_dtype must be restored after"
    assert tok.saved_to == tmp
    print("OK: bf16 matrices + fp32 norms/buffers, live fp32 untouched, compile unwrap/restore, dtype restore")


def test_no_compile_passthrough(tmp="/tmp/_hfckpt_selfcheck2"):
    plain = _Orig()                # model.model with no _orig_mod (compile off)
    model = _FakeModel(plain)
    _save_hf_ckpt(model, _FakeTok(), tmp)
    assert model.saved_model_at_save is plain and model.model is plain
    print("OK: no-compile path saves model.model unchanged")


def test_eager_swaps_and_restores():
    orig = _Orig()
    model = _FakeModel(_Compiled(orig))
    with _eager(model):
        assert model.model is orig, "eval must run on the un-compiled module"
    assert isinstance(model.model, _Compiled), "compiled module must be restored after"
    # even if the body raises, the compiled module must be restored
    try:
        with _eager(model):
            raise ValueError("boom")
    except ValueError:
        pass
    assert isinstance(model.model, _Compiled), "must restore compiled module even on exception"
    print("OK: _eager swaps to orig, restores compiled (incl. on exception)")


if __name__ == "__main__":
    import os
    os.makedirs("/tmp", exist_ok=True)
    test_unwrap_bf16_restore()
    test_no_compile_passthrough()
    test_eager_swaps_and_restores()
    print("all self-checks passed")
