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
    # The kernel needs N = sum(T_enc * (U + 1)) to size its buffers; asking the GPU for it is a host sync that drains
    # the queue every step. Every term is known on the CPU before the batch is copied: audio lengths (or
    # aug_online's new lengths, computed on the CPU), token lengths, and NeMo's own length formulas.
    from nemo.collections.asr.parts.submodules.subsampling import calc_length
    feat, pre = model.preprocessor.featurizer, model.encoder.pre_encode
    host = {"alen": None, "ylen": None, "checked": 0}
    before = model.on_before_batch_transfer

    def on_before_batch_transfer(batch, dataloader_idx):
        batch = before(batch, dataloader_idx)
        if isinstance(batch, (tuple, list)) and len(batch) >= 4 and not batch[1].is_cuda:
            host["alen"], host["ylen"] = batch[1].clone(), batch[3].clone()
        else:
            host["alen"] = host["ylen"] = None
        if "aug_online" in sys.modules:
            sys.modules["aug_online"].STATE["cpu_lens"] = None          # set again by this step's augmentation
        return batch

    model.on_before_batch_transfer = on_before_batch_transfer

    def enc_len_host(alen):
        f = feat.get_seq_len(alen.long())
        f = torch.where(alen == 0, torch.zeros_like(f), f)               # NeMo's zero-length rule
        return calc_length(f, all_paddings=pre._left_padding + pre._right_padding, kernel_size=pre._kernel_size,
                           stride=pre._stride, ceil_mode=pre._ceil_mode, repeat_num=pre._sampling_num)

    def n_rows_host(encoder_lengths, transcript_lengths):
        aug = sys.modules.get("aug_online")
        alen = aug.STATE.get("cpu_lens") if aug is not None and aug.STATE.get("cpu_lens") is not None else host["alen"]
        ylen = host["ylen"]
        if alen is None or ylen is None or alen.shape[0] != encoder_lengths.shape[0]:
            return None                                                  # unknown on the host: the synced path
        el = enc_len_host(alen)
        if host["checked"] < 3:                                          # plumbing check: 3 one-off syncs, then trust
            assert torch.equal(el.long(), encoder_lengths.long().cpu()), "host encoder lengths != the encoder's"
            assert torch.equal(ylen.long(), transcript_lengths.long().cpu()), "host token lengths != the batch's"
            host["checked"] += 1
            if host["checked"] == 3:
                print("[fused_joint] host-side lattice sizes match the device's (3 checks): no per-step sync", flush=True)
        return int((el.long() * (ylen.long() + 1)).sum())

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
                                  transcript_lengths, fastemit_lambda=lam, dropout=p,
                                  n_rows=n_rows_host(encoder_lengths, transcript_lengths))
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
    """One batch through the run1 model: NeMo's fused loop vs the kernel, no dropout / SpecAugment, loss + grads."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from("/home/marimo/work/asr/exp/run1/run1.nemo").cuda().train()
    m.spec_augmentation = None                                       # cudnn LSTM backward needs train mode: switch
    for mod in m.modules():                                          # off every random op instead
        if isinstance(mod, torch.nn.Dropout):
            mod.eval()
    m.joint.eval()
    B, T, U = 16, 200, 40
    torch.manual_seed(0)
    audio = torch.randn(B, T * 1280, device="cuda") * 0.05
    alen = torch.full((B,), T * 1280, device="cuda") - torch.randint(0, T * 640, (B,), device="cuda")
    y = torch.randint(5, m.tokenizer.vocab_size, (B, U), device="cuda")
    ylen = torch.randint(U // 2, U + 1, (B,), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        enc0, elen = m.forward(input_signal=audio, input_signal_length=alen)
        dec0, _, _ = m.decoder(targets=y, target_length=ylen)
    enc0, dec0 = enc0.detach(), dec0.detach()
    res = {}
    for name in ("nemo", "nemo again", "fused"):                     # identical inputs: only the joint differs
        if name == "fused":
            enable(m)
        m.zero_grad()
        enc, dec = enc0.clone().requires_grad_(), dec0.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _, _ = m.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=elen, transcripts=y,
                                    transcript_lengths=ylen)
        loss.backward()
        res[name] = (loss.item(), m.joint.joint_net[-1].weight.grad.float().clone(), enc.grad.float(), dec.grad.float())
        print(f"  {name}: loss {loss.item():.6f}", flush=True)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    for other in ("nemo again", "fused"):
        a, b = res[other], res["nemo"]
        print(f"{other} vs nemo: loss rel {abs(a[0] - b[0]) / abs(b[0]):.2e}  dW_joint {rel(a[1], b[1]):.2e}  "
              f"d_enc {rel(a[2], b[2]):.2e}  d_dec {rel(a[3], b[3]):.2e}", flush=True)


if __name__ == "__main__" and "--check" in sys.argv:
    _check()
