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


def load_store(path, n_rows):
    """teacher_dump.py shards -> (start (n_rows,) int64 or -1, U (n_rows,) int32, idx (T, K) int16, p (T, K) float16,
    tok (T,) int16) indexed by manifest row"""
    import glob
    import numpy as np
    if path.startswith("hf:"):
        from huggingface_hub import snapshot_download
        _, repo, sub = path.split(":", 2)
        path = os.path.join(snapshot_download(repo, allow_patterns=[f"{sub}/*.npz"]), sub)
    files = sorted(glob.glob(os.path.join(path, "shard_*.npz")))
    assert files, f"no teacher_dump shards in {path}"
    start = np.full(n_rows, -1, np.int64)
    U = np.zeros(n_rows, np.int32)
    idx, p, tok, base = [], [], [], 0
    for fn in files:
        z = np.load(fn)
        r = z["row"]
        assert r.max() < n_rows, f"{fn}: row {r.max()} >= manifest rows {n_rows} (another manifest?)"
        start[r] = z["off"] + base
        U[r] = z["U"]
        idx.append(z["idx"]); p.append(z["p"]); tok.append(z["tok"])
        base += len(z["tok"])
    return start, U, np.concatenate(idx), np.concatenate(p), np.concatenate(tok), len(files)


def enable_store(model, path, kd_weight=0.7, hard_weight=1.0, mask=True, manifest=None):
    """per-token KD on the RNN-T head from DUMPED teacher targets (teacher_dump.py): no teacher forward in the step.
    The fast loader's batches carry their manifest row ids (5th element); an outermost on_before_batch_transfer
    takes them off (NeMo's step sees the usual 4) and the joint builds the dense (B, U, V) targets from the store.
    path: one store, or several 'LA=store' (one teacher_dump per teacher look-ahead): each batch then reads the
    store of the look-ahead the student sampled for it (a matched teacher, as with a live --teacher)."""
    import numpy as np
    n_rows = sum(1 for _ in open(manifest, encoding="utf-8"))
    paths = [path] if isinstance(path, str) else list(path)
    named = {int(x.split("=", 1)[0]): x.split("=", 1)[1] for x in paths if "=" in x.split(":", 1)[0]}
    assert len(named) in (0, len(paths)), f"--kd_targets: all 'LA=store' or one plain store, got {paths}"
    stores = {}                                      # look-ahead (None = one store for every batch) -> (idx, p)
    start = U = tok = None
    nf = 0
    for la, sp in (named.items() if named else [(None, paths[0])]):
        st_, U_, idx_, p_, tok_, nf_ = load_store(sp, n_rows)
        missing = int((st_ < 0).sum())
        assert missing == 0, f"teacher store {sp}: {missing} of {n_rows} manifest rows have no targets"
        if start is None:
            start, U, tok = st_, U_, tok_
        else:                                        # same manifest + tokenizer: identical token layout
            assert np.array_equal(U_, U) and np.array_equal(st_, start), f"store {sp}: token layout differs"
        stores[la] = (idx_, p_)
        nf += nf_
    idx = next(iter(stores.values()))[0]
    V1 = model.joint.joint_net[-1].out_features
    dead = dead_pieces(model.tokenizer, manifest) if mask else []
    reset = None
    if dead:
        reset, nlin = mask_outputs(model, dead)
        print(f"[teacher_kd] student: {len(dead)} pieces masked (bias -1e4, no grad) in {nlin} readouts", flush=True)
    dead_t = torch.tensor(dead, dtype=torch.long, device="cuda") if dead else None
    state = {"ids": None, "checked": 0, "by_la": {}}
    before = model.on_before_batch_transfer

    def on_before_batch_transfer(batch, dataloader_idx):
        if isinstance(batch, (tuple, list)) and len(batch) == 5:
            state["ids"] = batch[4].numpy()
            batch = tuple(batch[:4])
        return before(batch, dataloader_idx)

    model.on_before_batch_transfer = on_before_batch_transfer
    orig_step = model.training_step

    def training_step(batch, *args, **kw):
        if reset is not None:
            reset()
        try:
            out = orig_step(batch, *args, **kw)
            kd = getattr(model.joint, "_kd_value", None)
            if kd is not None:
                model.log("core/loss_kd_rnnt", kd, on_step=True, on_epoch=False)
            n = sum(state["by_la"].values())
            if n in (10, 200, 2000):                 # the matched stores are actually being used, per look-ahead
                print(f"[teacher_kd] store batches per student look-ahead after {n}: {dict(sorted(state['by_la'].items()))}",
                      flush=True)
            return out
        finally:
            model.joint._kd_value = None

    def teach(encoder_lengths, transcripts, transcript_lengths, n_rows_):
        ids = state["ids"]
        if ids is None:
            return None
        B, Um = transcripts.shape
        u = U[ids]
        if state["checked"] < 5:                     # join check: the stored tokens ARE this batch's labels
            tl = transcript_lengths.cpu().numpy()
            assert (u == tl).all(), f"store token counts != labels for rows {ids[u != tl][:4]}"
            tr = transcripts.cpu().numpy()
            for k in range(B):
                st = start[ids[k]]
                assert (tok[st:st + u[k]].astype(np.int64) == tr[k, :u[k]]).all(), f"store tokens != labels, row {ids[k]}"
            state["checked"] += 1
            if state["checked"] == 5:
                print("[teacher_kd] store join check: 5 batches, every row's stored tokens == its labels", flush=True)
        if None in stores:
            idx, p = stores[None]
        else:                                        # the look-ahead the student sampled for this batch
            ctx = getattr(model.encoder, "_fused_ctx", None)
            assert ctx is not None, "per-look-ahead stores need --fused_attn on the student (its sampled look-ahead)"
            r = ctx[0][1]
            assert r in stores, f"student look-ahead {r} has no store (have {sorted(stores)})"
            idx, p = stores[r]
            state["by_la"][r] = state["by_la"].get(r, 0) + 1
        pos = np.concatenate([np.arange(n) for n in u]) if u.sum() else np.zeros(0, np.int64)
        brow = np.repeat(np.arange(B), u)
        src = np.repeat(start[ids], u) + pos
        ii = torch.from_numpy(idx[src].astype(np.int64)).cuda(non_blocking=True)
        pp = torch.from_numpy(p[src].astype(np.float32)).cuda(non_blocking=True)
        dense = torch.zeros(B * Um, V1, device="cuda")
        flat = torch.from_numpy(brow * Um + pos).cuda(non_blocking=True)
        rows_ = torch.zeros(len(src), V1, device="cuda").scatter_(1, ii, pp)
        dense.index_copy_(0, flat, rows_)
        pbar = dense.view(B, Um, V1)
        if dead_t is not None:
            pbar.index_fill_(-1, dead_t, 0.0)
        pbar[..., V1 - 1] = 0.0
        pbar = pbar / pbar.sum(-1, keepdim=True).clamp_min(1e-30)
        return None, None, None, None, pbar

    model.training_step = training_step
    model.joint._teacher = teach
    model.joint._kd_w = float(kd_weight)
    model.joint._hard_w = float(hard_weight)
    model.joint._kd_mode = "pos"
    print(f"[teacher_kd] per-token KD from the dumped store(s) {paths}: {nf} shards, {len(tok)} tokens, {n_rows} rows, "
          f"top-{idx.shape[1]}; RNN-T loss = {hard_weight} x text + {kd_weight} x KD; no teacher model", flush=True)


def _dense_from_topk(rows, B, Um, V1, dead_t):
    """rows: per batch row (idx (U, K) int, p (U, K) float) -> (B, Um, V1) per-token targets on the GPU, renormalised
    over the stored top-K, blank and masked pieces 0"""
    import numpy as np
    U = [len(r[0]) for r in rows]
    if not sum(U):
        return torch.zeros(B, Um, V1, device="cuda")
    ii = torch.from_numpy(np.concatenate([r[0] for r in rows]).astype(np.int64)).cuda(non_blocking=True)
    pp = torch.from_numpy(np.concatenate([r[1] for r in rows]).astype(np.float32)).cuda(non_blocking=True)
    flat = torch.from_numpy(np.concatenate([b * Um + np.arange(u) for b, u in enumerate(U)]).astype(np.int64)).cuda()
    dense = torch.zeros(B * Um, V1, device="cuda")
    dense.index_copy_(0, flat, torch.zeros(len(ii), V1, device="cuda").scatter_(1, ii, pp))
    pbar = dense.view(B, Um, V1)
    if dead_t is not None:
        pbar.index_fill_(-1, dead_t, 0.0)
    pbar[..., V1 - 1] = 0.0
    return pbar / pbar.sum(-1, keepdim=True).clamp_min(1e-30)


def enable(model, init, ckpt=None, lookaheads=None, kd_weight=0.7, hard_weight=0.3, frame_k=1, fused=True,
           rnnt_mode="node",
           mask=True, manifest=None, teacher_bp=0.0, teacher_la=None, cache=False, cache_k=16):
    """cache=True (per-token mode, FIXED teacher look-ahead): each row's top-cache_k targets are kept in RAM the
    first time the teacher sees it (epoch 1); a batch whose rows are all cached skips the teacher (epoch 2+). Needs
    the fast loader's row ids (train_asr --kd_cache)."""
    sys.path.insert(0, TKF)
    assert not cache or (rnnt_mode == "pos" and teacher_la is not None and not frame_k), \
        "kd cache: per-token mode with a fixed --teacher_la (a cached target must not depend on the sampled context)"
    from kernels.sm120.rnnt_joint import rnnt_joint_node_logp
    teacher = load_teacher(init, ckpt, lookaheads, fused)
    tdt = [int(d) for d in (teacher.cfg.model_defaults.get("tdt_durations", None) or [])]   # Parakeet TDT teacher
    if tdt:
        assert rnnt_mode == "pos" and not frame_k, "a TDT teacher only supports per-token targets (--kd_rnnt_mode pos)"
        from kernels.sm120.tdt_joint import tdt_joint_pos_targets
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
    store = {}                                       # kd cache: manifest row -> (idx (U, K) int16, p (U, K) float16)
    stats = {"hit": 0, "miss": 0}
    if cache:
        before = model.on_before_batch_transfer

        def on_before_batch_transfer(batch, dataloader_idx):
            if isinstance(batch, (tuple, list)) and len(batch) == 5:
                state["ids"] = batch[4].numpy()
                batch = tuple(batch[:4])
            return before(batch, dataloader_idx)

        model.on_before_batch_transfer = on_before_batch_transfer
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
        ids = state.get("ids") if cache else None
        V1 = model.joint.joint_net[-1].out_features
        if ids is not None and all(int(i) in store for i in ids):   # epoch 2+: no teacher forward
            stats["hit"] += 1
            pbar = _dense_from_topk([store[int(i)] for i in ids], transcripts.shape[0], transcripts.shape[1], V1, dead_t)
            return None, None, None, None, pbar
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
        if tdt:                                      # per-token targets are frame-free: the teacher's own lattice
            lin = teacher.joint.joint_net[-1]
            pbar, _ = tdt_joint_pos_targets(f, g, lin.weight, lin.bias, transcripts, elen, transcript_lengths, tdt)
            tpb = tpy = tnll = ft = None
        if not tdt and state["checked"] < 3:         # 3 one-off syncs: the two lattices must line up node for node
            assert torch.equal(elen.long(), encoder_lengths.long()), \
                f"teacher / student frame counts differ: {elen.tolist()[:4]} vs {encoder_lengths.tolist()[:4]}"
            state["checked"] += 1
        lin = teacher.joint.joint_net[-1]
        if not tdt and dead_t is not None and state["mass"] < 3:  # how much the teacher puts on the masked pieces (one utterance)
            t0, u0 = int(encoder_lengths[0]), int(transcript_lengths[0]) + 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = teacher.joint.joint_net(f[0, :t0, None] + g[0, None, :u0])
            pr = lg.float().softmax(-1)
            dm = pr.index_select(-1, dead_t).sum(-1)
            print(f"[teacher_kd] teacher mass on masked pieces (utt 0, {t0}x{u0} nodes): mean {dm.mean().item():.2e} "
                  f"max {dm.max().item():.2e}; on non-blank mass: {(dm.sum() / (1 - pr[..., -1]).sum()).item():.2e}",
                  flush=True)
            state["mass"] += 1
        if not tdt:
            out = rnnt_joint_node_logp(f, g, lin.weight, lin.bias, transcripts, encoder_lengths,
                                       transcript_lengths, n_rows=n_rows, frame_targets=frame_k,
                                       blank_penalty=teacher_bp, pos_targets=rnnt_mode == "pos")
            tpb, tpy, tnll, ft = out[:4]
            pbar = out[4] if rnnt_mode == "pos" else None
        if pbar is not None and dead_t is not None:   # the student cannot emit masked pieces: renormalise without them
            pbar = pbar.index_fill(-1, dead_t, 0.0)
            pbar = pbar / pbar.sum(-1, keepdim=True).clamp_min(1e-30)
        if ids is not None and pbar is not None:      # epoch 1: keep each row's top-K for later epochs
            stats["miss"] += 1
            pk, ik = pbar.topk(cache_k, dim=-1)
            pk, ik = pk.cpu().numpy().astype("float16"), ik.cpu().numpy().astype("int16")
            ul = transcript_lengths.cpu().numpy()
            for b, i in enumerate(ids):
                store[int(i)] = (ik[b, :ul[b]], pk[b, :ul[b]])
            if stats["miss"] % 1000 == 0:
                print(f"[teacher_kd] cache: {len(store)} rows stored, {stats['hit']} cached batches so far", flush=True)
        return tpb, tpy, ft, tnll, pbar

    model.training_step = training_step
    model.joint._teacher = teach
    model.joint._kd_w = float(kd_weight)
    model.joint._kd_mode = rnnt_mode           # node: lattice KL | fullsum: |nll - teacher nll|
    model.joint._hard_w = float(hard_weight)
    model._kd_teacher = [teacher]                    # a list: not registered as a submodule (not saved, not trained)
    model._kd_cache_stats = stats
    n = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher_kd] teacher {n / 1e6:.0f}M params frozen; student loss per head = {hard_weight} x hard + "
          f"{kd_weight} x KD (RNN-T: collapsed lattice KL; CTC: CE to the teacher's one-token-per-frame alignment); teacher look-ahead "
          + (f"follows the student's sampled context {sorted(tctx)}" if teacher_la is None else f"fixed at {teacher_la}")
          + (f"; teacher blank penalty {teacher_bp} (as it decodes best)" if teacher_bp else ""), flush=True)
    return teacher
