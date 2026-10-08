"""Swap NeMo RNNTJoint's fused loss loop for tkf's fused joint + RNN-T loss kernel (train_asr.py --fused_joint).

NeMo (fuse_loss_wer, fused_batch_size 2) runs joint -> fp32 cast -> numba loss two utterances at a time, with a host
sync per pair. triton-kernel-fused kernels/sm120/rnnt_joint.py does the whole batch in one op on packed lattice rows
(no padding, logits never materialised): bench/bench_rnnt_joint.py, 5-25x on the 1200 s buckets, and
parity_check/parity_rnnt_joint.py gates it against fp64 and NeMo's own numba loss. Same loss value NeMo reports
((1 + FastEmit) * mean_batch NLL), same gradient formula; dropout draws its own mask (Triton RNG).

    import fused_joint; fused_joint.enable(model)            # after change_vocabulary / setup, before fit
    python voice/asr/fused_joint.py --check                   # patched vs NeMo forward on one real batch
"""
import os
import sys
import types

import torch

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def enable(model):
    sys.path.insert(0, TKF)
    from kernels.sm120.rnnt_joint import rnnt_joint_loss

    joint = model.joint
    net = list(joint.joint_net)
    assert isinstance(net[0], torch.nn.ReLU) and isinstance(net[-1], torch.nn.Linear), net
    drop = net[1] if len(net) == 3 else None
    assert joint._fuse_loss_wer and model.loss.reduction == "mean_batch"
    assert getattr(joint, "masking_prob", 0) <= 0 and not joint.is_adapter_available()
    assert getattr(joint, "temperature", 1.0) == 1.0
    lam = float((model.cfg.loss.get("warprnnt_numba_kwargs") or {}).get("fastemit_lambda", 0.0))
    nemo_forward = joint.forward

    def forward(self, encoder_outputs, decoder_outputs=None, encoder_lengths=None, transcripts=None,
                transcript_lengths=None, compute_wer=False, keep_hypotheses=False):
        if decoder_outputs is None:                                  # WER-only call: NeMo's path
            return nemo_forward(encoder_outputs=encoder_outputs, decoder_outputs=None, encoder_lengths=encoder_lengths,
                                transcripts=transcripts, transcript_lengths=transcript_lengths,
                                compute_wer=compute_wer, keep_hypotheses=keep_hypotheses)
        f = self.project_encoder(encoder_outputs.transpose(1, 2))
        g = self.project_prednet(decoder_outputs.transpose(1, 2))
        p = drop.p if (drop is not None and self.training) else 0.0
        loss, _ = rnnt_joint_loss(f, g, net[-1].weight, net[-1].bias, transcripts, encoder_lengths,
                                  transcript_lengths, fastemit_lambda=lam, dropout=p)
        wer = wer_num = wer_denom = None
        hyp = []
        if compute_wer:                                              # NeMo's per-sub-batch WER, on the whole batch
            if self.training:
                sync = self.wer._to_sync
                self.wer._to_sync = False
            self.wer.update(predictions=encoder_outputs.detach(), predictions_lengths=encoder_lengths,
                            targets=transcripts.detach(), targets_lengths=transcript_lengths)
            hyp = self.wer.get_hypotheses() if keep_hypotheses else []
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()
            if self.training:
                self.wer._to_sync = sync
        self.hypotheses = hyp if keep_hypotheses else None
        return loss, wer, wer_num, wer_denom

    joint.forward = types.MethodType(forward, joint)
    print(f"[fused_joint] NeMo joint loss -> tkf rnnt_joint (FastEmit {lam}, dropout {drop.p if drop else 0})",
          flush=True)
    return nemo_forward


def _check():
    """One real batch through the run1 model: NeMo's fused loop vs the kernel, eval mode (no dropout) + one grad."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from("/home/marimo/work/asr/exp/run1/run1.nemo").cuda().eval()
    B, T, U = 16, 200, 40
    torch.manual_seed(0)
    audio = torch.randn(B, T * 1280, device="cuda") * 0.05
    alen = torch.full((B,), T * 1280, device="cuda") - torch.randint(0, T * 640, (B,), device="cuda")
    y = torch.randint(5, m.tokenizer.vocab_size, (B, U), device="cuda")
    ylen = torch.randint(U // 2, U + 1, (B,), device="cuda")
    res = {}
    for name in ("nemo", "fused"):
        if name == "fused":
            enable(m)
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, elen = m.forward(input_signal=audio, input_signal_length=alen)
            dec, _, _ = m.decoder(targets=y, target_length=ylen)
            loss, _, _, _ = m.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=elen, transcripts=y,
                                    transcript_lengths=ylen)
        loss.backward()
        gw = m.joint.joint_net[-1].weight.grad.float().clone()
        genc = m.encoder.layers[-1].feed_forward2.linear2.weight.grad.float().clone()
        res[name] = (loss.item(), gw, genc)
        print(f"  {name}: loss {loss.item():.6f}", flush=True)
    (l0, w0, e0), (l1, w1, e1) = res["nemo"], res["fused"]
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    print(f"loss rel {abs(l1 - l0) / abs(l0):.2e}  dW_joint rel {rel(w1, w0):.2e}  dW_enc_last rel {rel(e1, e0):.2e}")


if __name__ == "__main__" and "--check" in sys.argv:
    _check()
