"""Teacher -> student distillation for a hybrid transducer student (train_asr.py --teacher ...).

The student (e.g. the 114M streaming hybrid with --new_vocab on the TEACHER's tokenizer, so token ids match) learns
from a frozen transducer teacher (e.g. the fine-tuned Nemotron RNN-T-only model) on every training batch:
  RNN-T head  loss = hard_weight * RNN-T nll + kd_weight * collapsed lattice KL (every (t, u) node, {label, blank,
              rest}), fused into the student's joint backward (tkf rnnt_joint_loss(kd=...))
  CTC head    loss = hard_weight * CTC + kd_weight * KL(teacher per-frame emission distribution || CTC head),
              targets = the teacher's ALIGNMENT, one token or blank per frame (tkf rnnt_joint_node_logp(frame_targets=1)),
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


def dead_pieces(tok, manifest=None):
    """ids of the teacher's cased / punctuated pieces ('▁The', '..', '<unk>') that the training transcripts never
    produce: a masked piece that DID occur as a target would cost ~1e4 nats, so the whole manifest is checked"""
    import json
    import re
    ok = re.compile(r"^▁?[a-z0-9']+$")
    sp = tok.tokenizer
    cand = {i for i in range(sp.get_piece_size()) if not ok.match(sp.id_to_piece(i))}
    if manifest:
        texts = [json.loads(line)["text"] for line in open(manifest, encoding="utf-8")]
        used = set()
        for ids in sp.encode(texts):
            used.update(ids)
        print(f"[teacher_kd] {len(texts)} transcripts: {len(cand & used)} cased / punctuated pieces DO occur and stay "
              f"unmasked: {[sp.id_to_piece(i) for i in sorted(cand & used)][:20]}", flush=True)
        cand -= used
    return sorted(cand)


def mask_outputs(model, ids):
    """student: the dead pieces' output biases held at -1e4 (logit ~ -inf in training, validation, export) in every
    vocabulary readout (RNN-T joint + each CTC pass); their weight / bias gradients are zeroed"""
    V1 = model.joint.joint_net[-1].out_features
    lins = [model.joint.joint_net[-1]] + [m for m in model.ctc_decoder.modules()
                                          if isinstance(m, torch.nn.Linear) and m.out_features == V1]
    idx = {}                                          # per device: built while the model is still on the CPU

    def on(dev):
        if dev not in idx:
            idx[dev] = torch.tensor(ids, dtype=torch.long, device=dev)
        return idx[dev]

    def zero_rows(g):
        return g.index_fill(0, on(g.device), 0.0)

    for lin in lins:
        lin.weight.register_hook(zero_rows)
        lin.bias.register_hook(zero_rows)

    @torch.no_grad()
    def reset():                                     # every step (weight decay must not pull them back up)
        for lin in lins:
            lin.bias.index_fill_(0, on(lin.bias.device), -1e4)

    reset()
    return reset, len(lins)


def enable(model, init, ckpt=None, lookaheads=None, kd_weight=0.7, hard_weight=0.3, frame_k=1, fused=True,
           rnnt_mode="node",
           mask=True, manifest=None, teacher_bp=0.0, teacher_la=None):
    sys.path.insert(0, TKF)
    from kernels.sm120.rnnt_joint import rnnt_joint_node_logp
    teacher = load_teacher(init, ckpt, lookaheads, fused)
    sample = "the quick brown fox jumps over the lazy dog's medical appointment on tuesday"
    a, b = model.tokenizer.text_to_ids(sample), teacher.tokenizer.text_to_ids(sample)
    assert a == b and model.tokenizer.vocab_size == teacher.tokenizer.vocab_size, \
        f"student and teacher tokenizers differ ({model.tokenizer.vocab_size} vs {teacher.tokenizer.vocab_size} pieces)"
    tctx = {c[1]: list(c) for c in teacher.encoder.att_context_size_all}
    state = {"batch": None, "checked": 0, "mass": 0}
    dead = dead_pieces(teacher.tokenizer, manifest) if mask else []
    reset = None
    if dead:
        reset, nlin = mask_outputs(model, dead)
        print(f"[teacher_kd] student: {len(dead)} of {teacher.tokenizer.vocab_size} pieces masked (bias -1e4, no grad) "
              f"in {nlin} readouts -- never produced by lowercase transcripts: "
              f"{[teacher.tokenizer.tokenizer.id_to_piece(i) for i in dead[:12]]} ...", flush=True)
    dead_t = torch.tensor(dead, dtype=torch.long, device="cuda") if dead else None
    orig_step = model.training_step

    def training_step(batch, *args, **kw):         # Lightning passes batch_idx only if the signature names it
        if reset is not None:
            reset()
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
        r = ctx[0][1] if teacher_la is None else teacher_la   # fixed teacher look-ahead (-1 = full) or the student's
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
        if dead_t is not None and state["mass"] < 3:  # how much the teacher puts on the masked pieces (one utterance)
            t0, u0 = int(encoder_lengths[0]), int(transcript_lengths[0]) + 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = teacher.joint.joint_net(f[0, :t0, None] + g[0, None, :u0])
            pr = lg.float().softmax(-1)
            dm = pr.index_select(-1, dead_t).sum(-1)
            print(f"[teacher_kd] teacher mass on masked pieces (utt 0, {t0}x{u0} nodes): mean {dm.mean().item():.2e} "
                  f"max {dm.max().item():.2e}; on non-blank mass: {(dm.sum() / (1 - pr[..., -1]).sum()).item():.2e}",
                  flush=True)
            state["mass"] += 1
        tpb, tpy, tnll, ft = rnnt_joint_node_logp(f, g, lin.weight, lin.bias, transcripts, encoder_lengths,
                                                  transcript_lengths, n_rows=n_rows, frame_targets=frame_k,
                                                  blank_penalty=teacher_bp)
        return tpb, tpy, ft, tnll

    model.training_step = training_step
    model.joint._teacher = teach
    model.joint._kd_w = float(kd_weight)
    model.joint._kd_mode = rnnt_mode           # node: lattice KL | fullsum: |nll - teacher nll|
    model.joint._hard_w = float(hard_weight)
    model._kd_teacher = [teacher]                    # a list: not registered as a submodule (not saved, not trained)
    n = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher_kd] teacher {n / 1e6:.0f}M params frozen; student loss per head = {hard_weight} x hard + "
          f"{kd_weight} x KD (RNN-T: collapsed lattice KL; CTC: CE to the teacher's one-token-per-frame alignment); teacher look-ahead "
          + (f"follows the student's sampled context {sorted(tctx)}" if teacher_la is None else f"fixed at {teacher_la}")
          + (f"; teacher blank penalty {teacher_bp} (as it decodes best)" if teacher_bp else ""), flush=True)
    return teacher
