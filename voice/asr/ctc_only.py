"""CTC-only training of the NeMo hybrid model (train_asr.py --ctc_only): no RNN-T at all.

The hybrid's training_step runs the RNN-T prediction net, the RNN-T joint over every (frame, token, vocab) cell and
the RNN-T loss, and mixes the CTC loss in at ctc_loss_weight. Here: encoder -> CTC decoder -> CTC loss, nothing else.
The RNN-T decoder / joint are frozen (no gradients, so the optimizer skips them) and never run; validation decodes
and scores with CTC. Logged under NeMo's keys (train_loss, train_ctc_loss, training_batch_wer, val_wer / _num /
_denom, val_loss), so W&B and train_asr's per-source val aggregation work unchanged.

    import ctc_only; ctc_only.enable(model)          # before selfcond_head.enable (which wraps training_step)
"""
import types


def enable(model):
    for mod in (model.decoder, model.joint):
        mod.requires_grad_(False)

    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch[:4]
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal
        log_probs = self.ctc_decoder(encoder_output=encoded)
        loss = self.ctc_loss(log_probs=log_probs, targets=transcript, input_lengths=encoded_len,
                             target_lengths=transcript_len)
        logs = {"train_loss": loss, "train_ctc_loss": loss, "learning_rate": self._optimizer.param_groups[0]["lr"],
                "global_step": float(self.trainer.global_step)}
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.ctc_wer.update(predictions=log_probs, targets=transcript, targets_lengths=transcript_len,
                                predictions_lengths=encoded_len)
            wer, _, _ = self.ctc_wer.compute()
            self.ctc_wer.reset()
            logs["training_batch_wer"] = wer
        self.log_dict(logs)
        return {"loss": loss}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch[:4]
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        log_probs = self.ctc_decoder(encoder_output=encoded)
        logs = {}
        if self.compute_eval_loss:
            logs["val_loss"] = self.ctc_loss(log_probs=log_probs, targets=transcript, input_lengths=encoded_len,
                                             target_lengths=transcript_len)
        self.ctc_wer.update(predictions=log_probs, targets=transcript, targets_lengths=transcript_len,
                            predictions_lengths=encoded_len)
        wer, num, den = self.ctc_wer.compute()
        self.ctc_wer.reset()
        logs.update(val_wer=wer, val_wer_num=num, val_wer_denom=den)
        return logs

    model.training_step = types.MethodType(training_step, model)
    model.validation_pass = types.MethodType(validation_pass, model)
    frozen = sum(p.numel() for m in (model.decoder, model.joint) for p in m.parameters())
    print(f"[ctc_only] RNN-T decoder + joint frozen and skipped ({frozen / 1e6:.1f}M params); CTC-only training and "
          f"validation", flush=True)
