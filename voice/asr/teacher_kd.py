"""Teacher -> student distillation for a hybrid transducer student (train_asr.py --teacher ...).

The student (e.g. the 114M streaming hybrid with --new_vocab on the TEACHER's tokenizer, so token ids match) learns
from a frozen transducer teacher (e.g. the fine-tuned Nemotron RNN-T-only model) on every training batch:
  RNN-T head  loss = hard_weight * RNN-T nll + kd_weight * collapsed lattice KL (every (t, u) node, {label, blank,
              rest}), fused into the student's joint backward (tkf rnnt_joint_loss(kd=...))
  CTC head    loss = hard_weight * CTC + kd_weight * KL(teacher per-frame emission distribution || CTC head),
              targets = the teacher lattice's per-frame top-4 (tkf rnnt_joint_node_logp(frame_targets=4)),
              KL by kl_div_topk on the fused self-conditioned pass's own logits (selfcond_head, lattice mode)
The teacher runs no-grad in bf16 on the batch's (augmented) audio, without SpecAugment, at the look-ahead the
student's encoder sampled for this batch (so a 240 ms student is taught by a 240 ms teacher). Its own lattice uses
the same reference transcript, lengths and packing as the student's, so node i of one is node i of the other.

    import teacher_kd; teacher_kd.enable(student, "nvidia/nemotron-speech-streaming-en-0.6b", ckpt, lookaheads)
"""
import os
import sys

import torch

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def _ctx(left, r):
    return [-1, -1] if r < 0 else [left, r]                 # train_asr --lookaheads' rule (240 ms = [70, 3])


def load_teacher(init, ckpt=None, lookaheads=None, fused=True):
    import nemo.collections.asr as nemo_asr
    t = (nemo_asr.models.ASRModel.restore_from(init) if init.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(init))
    if lookaheads:                                   # the look-ahead set the teacher was fine-tuned with
        left = t.encoder.att_context_size_all[0][0]
        t.encoder.att_context_size_all = [_ctx(left, r) for r in lookaheads]
    if ckpt:
        if ckpt.startswith("hf:"):                   # hf:<repo>:<path>
            from huggingface_hub import hf_hub_download
            _, repo, path = ckpt.split(":", 2)
            ckpt = hf_hub_download(repo, path)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        miss, unexp = t.load_state_dict(sd, strict=False)
        params = {n for n, _ in t.named_parameters()}
        assert not [k for k in miss if k in params], f"teacher ckpt lacks parameters: {miss[:8]}"
        print(f"[teacher_kd] teacher {init} + {ckpt}: {len(sd)} tensors (missing buffers {len(miss)}, "
              f"unused {len(unexp)})", flush=True)
    t = t.cuda().eval()
    for p in t.parameters():
        p.requires_grad_(False)
    t.spec_augmentation = None
    if fused:                                        # same kernels as training (no-grad forward)
        import fused_attn
        import fused_conv
        import fused_layer
        fused_layer.enable(t)
        fused_attn.enable(t)
        fused_conv.enable(t)
    return t


def enable(model, init, ckpt=None, lookaheads=None, kd_weight=0.7, hard_weight=0.3, frame_k=4, fused=True):
    sys.path.insert(0, TKF)
    from kernels.sm120.rnnt_joint import rnnt_joint_node_logp
    teacher = load_teacher(init, ckpt, lookaheads, fused)
    sample = "the quick brown fox jumps over the lazy dog's medical appointment on tuesday"
    a, b = model.tokenizer.text_to_ids(sample), teacher.tokenizer.text_to_ids(sample)
    assert a == b and model.tokenizer.vocab_size == teacher.tokenizer.vocab_size, \
        f"student and teacher tokenizers differ ({model.tokenizer.vocab_size} vs {teacher.tokenizer.vocab_size} pieces)"
    tctx = {c[1]: list(c) for c in teacher.encoder.att_context_size_all}
    state = {"batch": None, "checked": 0}
    orig_step = model.training_step

    def training_step(batch, *args, **kw):         # Lightning passes batch_idx only if the signature names it
        state["batch"] = batch[:4]                   # after on_after_batch_transfer: the augmented audio
        try:
            out = orig_step(batch, *args, **kw)
            kd = getattr(model.joint, "_kd_value", None)
            if kd is not None:
                model.log("core/loss_kd_rnnt", kd, on_step=True, on_epoch=False)
            return out
        finally:
            state["batch"] = None
            model.joint._kd_value = None

    @torch.no_grad()
    def teach(encoder_lengths, transcripts, transcript_lengths, n_rows):
        if state["batch"] is None:
            return None
        sig, slen = state["batch"][0], state["batch"][1]
        ctx = getattr(model.encoder, "_fused_ctx", None)
        assert ctx is not None, "teacher_kd needs --fused_attn on the student (its sampled look-ahead)"
        r = ctx[0][1]
        teacher.encoder.set_default_att_context_size(tctx.get(r, _ctx(70, r)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, elen = teacher.forward(input_signal=sig, input_signal_length=slen)
            dec, _, _ = teacher.decoder(targets=transcripts, target_length=transcript_lengths)
            f = teacher.joint.project_encoder(enc.transpose(1, 2))
            g = teacher.joint.project_prednet(dec.transpose(1, 2))
        if state["checked"] < 3:                     # 3 one-off syncs: the two lattices must line up node for node
            assert torch.equal(elen.long(), encoder_lengths.long()), \
                f"teacher / student frame counts differ: {elen.tolist()[:4]} vs {encoder_lengths.tolist()[:4]}"
            state["checked"] += 1
        lin = teacher.joint.joint_net[-1]
        tpb, tpy, tnll, ft = rnnt_joint_node_logp(f, g, lin.weight, lin.bias, transcripts, encoder_lengths,
                                                  transcript_lengths, n_rows=n_rows, frame_targets=frame_k)
        return tpb, tpy, ft

    model.training_step = training_step
    model.joint._teacher = teach
    model.joint._kd_w = float(kd_weight)
    model.joint._hard_w = float(hard_weight)
    model._kd_teacher = [teacher]                    # a list: not registered as a submodule (not saved, not trained)
    n = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher_kd] teacher {n / 1e6:.0f}M params frozen; student loss per head = {hard_weight} x hard + "
          f"{kd_weight} x KD (RNN-T: collapsed lattice KL; CTC: teacher per-frame top-{frame_k}); teacher look-ahead "
          f"follows the student's sampled context {sorted(tctx)}", flush=True)
    return teacher
