"""Self-conditioned CTC head for the NeMo hybrid model (train_asr.py --ctc_head lin_glu | glu_glu | lin_glu_glu ...).

Replaces NeMo's CTC decoder (one Conv1d 1x1 = Linear d -> V+1) with ctc_heads.SelfCondN: N passes, each pass's guess
(softmax, current + previous frame) fed back into the encoder features, GLU readouts (8d/3, SiLU, residual).
Training: the head's loss is tkf's fused pass (kernels/sm120/selfcond_ctc.py), computed inside ctc_decoder.forward;
NeMo's ctc_loss then returns it (so hybrid_rnnt_ctc's training_step, the CTC weight and the batch-WER logging stay
untouched). The per-pass losses use NeMo's own CTC reduction. Eval / decoding: the eager head, log_softmax.

    import selfcond_head; selfcond_head.enable(model, "lin_glu", weights=(0.3, 0.7))
NOTE: the saved .nemo still describes the stock CTC decoder; restoring it needs enable() before load_state_dict.

On-policy distillation (enable(..., distill=lam), train_asr --ctc_distill): every training step the model's own
transducer head (RNN-T / TDT) greedy-decodes the batch under no_grad; its token timestamps become per-frame targets
(token at its emission frame, the next free frame on collisions, blank elsewhere) and the CTC head gets
lam * sum-over-frames cross-entropy to them on top of its CTC loss: the fresh CTC head is taught the alignment the
pretrained decoder already knows.
"""
import types

import torch
import torch.nn as nn

import ctc_heads


class SelfCondCTCDecoder(nn.Module):
    def __init__(self, stock, d, readouts, weights):
        super().__init__()
        self.num_classes_with_blank = stock.num_classes_with_blank
        self.vocabulary = stock.vocabulary
        self.temperature = 1.0
        self.head = ctc_heads.SelfCondN(d, self.num_classes_with_blank, list(readouts), tuple(weights))
        self.stash = None                    # (targets, target lengths, input lengths, reduce) set by the model
        self.loss = None
        self.distill = 0.0                   # weight of the transducer-alignment cross-entropy (0 = off)
        self.distill_only = False            # True: the head learns ONLY from the transducer (no text CTC loss)
        self.hard = 1.0                      # weight of the text CTC loss when distilling (teacher_kd: 0.3)
        self.frame_tgt = None                # (B, T) targets from the transducer's greedy decode, set per step
        self.distill_loss = None
        self.distill_mode = "greedy"         # "lattice": targets = the fused joint's per-frame lattice posteriors
        self.lattice_src = None              # () -> (idx, logp) each (B, T, K), set by enable()

    def forward(self, encoder_output):
        h = encoder_output.transpose(1, 2)                                # (B, T, d)
        if self.training and self.stash is not None:
            tgt, tlen, hl, reduce = self.stash
            self.loss, z = self.head.fused_loss(h, hl, tgt, tlen, self.num_classes_with_blank - 1, reduce=reduce,
                                                return_logits=True)
            lat = self.lattice_src() if (self.distill > 0 and self.distill_mode == "lattice") else None
            if lat is not None:                                          # soft per-frame targets from the lattice:
                idx, lpt = lat                                           # one fused KL (kl_div_topk), no decoding,
                z2 = z                                                   # on the fused pass's own logits (its grad
                                                                         # joins the pass's fused backward)
                B, T, V = z2.shape
                Tt = min(T, idx.shape[1])
                K = idx.shape[2]
                kl = _kl_topk()(z2[:, :Tt].reshape(-1, V), idx[:, :Tt].reshape(-1, K),
                                lpt[:, :Tt].reshape(-1, K)).view(B, Tt)
                mask = torch.arange(Tt, device=kl.device)[None, :] < hl[:, None]
                self.distill_loss = (kl * mask).sum(1).mean()            # per-utterance sum, as CTC mean_batch
            elif self.distill > 0 and self.frame_tgt is not None:        # eager head pass WITH grad for the CE
                logp = self.head(h).float().log_softmax(-1)
                T = min(logp.shape[1], self.frame_tgt.shape[1])
                nll = -logp[:, :T].gather(-1, self.frame_tgt[:, :T, None]).squeeze(-1)
                mask = torch.arange(T, device=nll.device)[None, :] < hl[:, None]
                self.distill_loss = (nll * mask).sum(1).mean()           # per-utterance sum, as CTC mean_batch
            if self.distill > 0 and self.distill_loss is not None:
                self.loss = (self.distill * self.distill_loss if self.distill_only
                             else self.hard * self.loss + self.distill * self.distill_loss)
            return z.detach().float().log_softmax(-1)                     # batch-WER only (no grad)
        return (self.head(h).float() / self.temperature).log_softmax(-1)

    # NeMo's change_decoding_strategy on hybrid models calls these on ctc_decoder (a NeuralModule API)
    def freeze(self, *args, **kw):         # NeMo passes partial=...
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def unfreeze(self, *args, **kw):
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()


def _kl_topk():
    import os
    import sys
    tkf = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")
    if tkf not in sys.path:
        sys.path.insert(0, tkf)
    from kernels.sm120.kl_div import kl_div_topk
    return kl_div_topk


def _frame_targets(model, enc, enc_len, blank):
    """greedy transducer decode of this batch (no grad, eval mode for the decoder / joint) -> (B, T) frame targets"""
    dec_m, joint = model.decoder, model.joint
    modes = (dec_m.training, joint.training)
    dec_m.eval()
    joint.eval()
    try:
        with torch.no_grad():
            hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                  return_hypotheses=True)
    finally:
        dec_m.train(modes[0])
        joint.train(modes[1])
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    B, T = enc.shape[0], enc.shape[2]
    ft = torch.full((B, T), blank, dtype=torch.long)
    for b, h in enumerate(hyps):
        ys = h.y_sequence.tolist() if torch.is_tensor(h.y_sequence) else list(h.y_sequence)
        ts = h.timestamp.tolist() if torch.is_tensor(h.timestamp) else list(h.timestamp)
        assert len(ts) == len(ys), f"transducer hypothesis without per-token timestamps ({len(ts)} vs {len(ys)})"
        nxt = 0
        for y, t in zip(ys, ts):
            t = max(int(t), nxt)
            if t >= T:
                break
            ft[b, t] = int(y)
            nxt = t + 1
    return ft.to(enc.device, non_blocking=True)


def enable(model, kind, weights=None, distill=0.0, distill_only=False, distill_mode="greedy", hard=1.0):
    readouts = kind.split("_")
    weights = weights or {2: (0.3, 0.7), 3: (0.2, 0.3, 0.5)}[len(readouts)]
    stock = model.ctc_decoder
    d = model.cfg.encoder.d_model
    dec = SelfCondCTCDecoder(stock, d, readouts, weights).to(next(stock.parameters()).device)
    model.ctc_decoder = dec
    dec.distill = float(distill)
    dec.distill_only = bool(distill_only)
    dec.hard = float(hard)
    dec.distill_mode = distill_mode
    if dec.distill > 0 and distill_mode == "lattice":       # the fused joint (train_asr --fused_joint) writes them
        model.joint._frame_k = 4
        dec.lattice_src = lambda: getattr(model.joint, "_frame_targets", None)
    assert not dec.distill_only or dec.distill > 0, "distill_only needs distill > 0"
    if dec.distill > 0 and distill_mode == "greedy":         # greedy decode inside training: no CUDA-graph decoder
        from omegaconf import open_dict
        dcfg = model.cfg.decoding
        with open_dict(dcfg):
            dcfg.strategy = "greedy_batch"
            dcfg.greedy.use_cuda_graph_decoder = False
        cur = getattr(model, "cur_decoder", None)
        model.change_decoding_strategy(dcfg, decoder_type="rnnt", verbose=False)
        if cur is not None:                  # validation keeps decoding with the head it used before
            model.cur_decoder = cur
    loss = model.ctc_loss
    orig_fwd = model.training_step

    # training_step computes encoded, then ctc_decoder(encoder_output=encoded), then ctc_loss(...): give the decoder
    # the targets / lengths first (wrap training_step), then let ctc_loss return the fused loss it computed.
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch[:4]
        dec.stash = [transcript, transcript_len, None, lambda nll, tl_: loss.reduce(nll, tl_)]
        try:
            out = orig_fwd(batch, batch_nb)
            if dec.distill_loss is not None:
                self.log("core/loss_distill", dec.distill_loss.detach(), on_step=True, on_epoch=False)
            return out
        finally:
            dec.stash, dec.loss, dec.frame_tgt, dec.distill_loss = None, None, None, None

    enc_fwd = model.encoder.forward

    def encoder_forward(*args, **kwargs):                                 # capture the encoded lengths
        out = enc_fwd(*args, **kwargs)
        if dec.stash is not None:
            dec.stash[2] = out[1]
            if dec.distill > 0 and dec.distill_mode == "greedy":
                dec.frame_tgt = _frame_targets(model, out[0], out[1], dec.num_classes_with_blank - 1)
        return out

    def ctc_loss_forward(self, log_probs, targets, input_lengths, target_lengths):
        if dec.loss is not None:
            return dec.loss
        return orig_loss_fwd(log_probs=log_probs, targets=targets, input_lengths=input_lengths,
                             target_lengths=target_lengths)

    orig_loss_fwd = loss.forward
    model.training_step = types.MethodType(training_step, model)
    model.encoder.forward = encoder_forward
    loss.forward = types.MethodType(ctc_loss_forward, loss)
    n = sum(p.numel() for p in dec.parameters())
    print(f"[selfcond_head] CTC decoder -> SelfCondN {readouts} weights {weights}, {n / 1e6:.2f}M params, "
          f"{dec.num_classes_with_blank} classes; training loss via tkf selfcond_ctc_pass"
          + (f"; {'ONLY ' if dec.distill_only else '+ '}{dec.distill} x transducer distillation ({dec.distill_mode})"
             if dec.distill > 0 else ""), flush=True)
    return dec
