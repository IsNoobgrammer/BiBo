"""Self-conditioned CTC head for the NeMo hybrid model (train_asr.py --ctc_head lin_glu | glu_glu | lin_glu_glu ...).

Replaces NeMo's CTC decoder (one Conv1d 1x1 = Linear d -> V+1) with ctc_heads.SelfCondN: N passes, each pass's guess
(softmax, current + previous frame) fed back into the encoder features, GLU readouts (8d/3, SiLU, residual).
Training: the head's loss is tkf's fused pass (kernels/sm120/selfcond_ctc.py), computed inside ctc_decoder.forward;
NeMo's ctc_loss then returns it (so hybrid_rnnt_ctc's training_step, the CTC weight and the batch-WER logging stay
untouched). The per-pass losses use NeMo's own CTC reduction. Eval / decoding: the eager head, log_softmax.

    import selfcond_head; selfcond_head.enable(model, "lin_glu", weights=(0.3, 0.7))
NOTE: the saved .nemo still describes the stock CTC decoder; restoring it needs enable() before load_state_dict.
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

    def forward(self, encoder_output):
        h = encoder_output.transpose(1, 2)                                # (B, T, d)
        if self.training and self.stash is not None:
            tgt, tlen, hl, reduce = self.stash
            self.loss, z = self.head.fused_loss(h, hl, tgt, tlen, self.num_classes_with_blank - 1, reduce=reduce,
                                                return_logits=True)
            return z.float().log_softmax(-1)                              # batch-WER only (no grad)
        return (self.head(h).float() / self.temperature).log_softmax(-1)


def enable(model, kind, weights=None):
    readouts = kind.split("_")
    weights = weights or {2: (0.3, 0.7), 3: (0.2, 0.3, 0.5)}[len(readouts)]
    stock = model.ctc_decoder
    d = model.cfg.encoder.d_model
    dec = SelfCondCTCDecoder(stock, d, readouts, weights).to(next(stock.parameters()).device)
    model.ctc_decoder = dec
    loss = model.ctc_loss
    orig_fwd = model.training_step

    # training_step computes encoded, then ctc_decoder(encoder_output=encoded), then ctc_loss(...): give the decoder
    # the targets / lengths first (wrap training_step), then let ctc_loss return the fused loss it computed.
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch[:4]
        dec.stash = [transcript, transcript_len, None, lambda nll, tl_: loss.reduce(nll, tl_)]
        try:
            return orig_fwd(batch, batch_nb)
        finally:
            dec.stash, dec.loss = None, None

    enc_fwd = model.encoder.forward

    def encoder_forward(*args, **kwargs):                                 # capture the encoded lengths
        out = enc_fwd(*args, **kwargs)
        if dec.stash is not None:
            dec.stash[2] = out[1]
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
          f"{dec.num_classes_with_blank} classes; training loss via tkf selfcond_ctc_pass", flush=True)
    return dec
