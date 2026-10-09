"""NeMo CTCLoss -> tkf deterministic ctc_loss (train_asr.py --fused_ctc).

torch's CUDA ctc_loss backward accumulates with atomics: with it, a fixed seed (and a fixed data stream) still gives
different gradient bits every run (det_probe.py). triton-kernel-fused kernels/sm120/ctc_loss.py: same loss and
gradient formula (parity_check/parity_ctc_loss.py vs torch fp64), no atomics, bitwise repeatable.

    import fused_ctc; fused_ctc.enable(model)
"""
import os
import sys
import types

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def enable(model):
    sys.path.insert(0, TKF)
    from kernels.sm120.ctc_loss import ctc_loss
    loss = model.ctc_loss
    assert loss._apply_reduction or loss.config_reduction == "none", loss.config_reduction  # torch 'mean'/'sum' not done

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        nll = ctc_loss(log_probs, targets, input_lengths, target_lengths, self._blank, self.zero_infinity)
        return self.reduce(nll, target_lengths) if self._apply_reduction else nll

    loss.forward = types.MethodType(forward, loss)
    print(f"[fused_ctc] CTC loss -> tkf ctc_loss (blank {loss._blank}, zero_infinity {loss.zero_infinity}, "
          f"reduction {loss.config_reduction})", flush=True)
