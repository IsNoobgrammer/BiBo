"""CTC head A/B on a FROZEN encoder: every head variant trains at once on the same encoder outputs (one encoder pass
per batch, one CTC loss per head), then each is scored per look-ahead on the English val sets and the meeting clips.

    python voice/asr/ctc_heads.py --nemo exp/run5/run5.eval.nemo --train asr/en1/train.jsonl --val asr/en1/val_*.jsonl \
        --steps 2000

Heads (all causal: zero added latency, same behaviour in training and streaming at every look-ahead):
  A   linear          LayerNorm-free Linear d -> vocab+1 (what run5 has), retrained from scratch
  B   mlp             LN -> Linear d->d -> SiLU -> Linear d->vocab+1
  C2  swa2 + mlp      causal sliding-window attention over the current + previous 1 encoder frame (4 heads), then B
  C4  swa4 + mlp      same, window 4 (current + 3 previous)
  E   swa3_sc         input -> causal SWA(w=3) -> vocab (aux) -> + embed(softmax) -> SWA(w=3) -> vocab; no MLP
  F   selfcond_lin    D without the MLP
  Br/Dr/Jr/Gr         B / D / J / G with BiBo's radial normsilu in the MLP readout instead of SiLU (= Swish)
  J   selfcond_37     D with loss 0.7 final + 0.3 pass 1 (normalised like G/H); F uses the same weights
  G/H sc3_mlp/_lin     3-pass self-conditioning, CTC loss 0.2 / 0.3 / 0.5 on passes 1 / 2 / 3, readouts MLP / Linear
  D   selfcond        intermediate CTC posterior (current + previous frame) projected back into the features, then B;
                      loss = final + 0.3 * intermediate (self-conditioned CTC, as IBM Granite)
  run5                the trained run5 CTC head, frozen (reference: 17.6k joint steps, not comparable in training)
In streaming, C keeps the last W-1 encoder frames and D the previous frame's posterior as a cache (cf. the conv cache).
The encoder sees a random look-ahead per batch (the run5 mix); evaluation is the full utterance under each chunked
look-ahead mask, which is what the cache-aware streaming loop computes. English script lock on every head at eval.
"""
import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIX = {0: 0.05, 1: 0.20, 3: 0.40, 6: 0.15, 13: 0.20}
MEETING = (("c2m.wav", "ref2m.txt"), ("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt"))


def set_lookahead(enc, r):
    enc.set_default_att_context_size([70 - 70 % (r + 1), r])      # = the training mask (70 // (r+1) chunks)


class MLP(nn.Module):
    """LN -> Linear d->d -> act -> Linear d->vocab. act = SiLU (= Swish), or BiBo's radial normsilu:
    silu(g / r) * r ** sigmoid(theta), r = RMS(g) over the features, theta one learnable scalar (init 0 -> p 0.5)."""

    def __init__(self, d, c, radial=False):
        super().__init__()
        self.ln, self.l1, self.l2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, c)
        self.radial_theta = nn.Parameter(torch.zeros(())) if radial else None

    def forward(self, h):
        g = self.l1(self.ln(h))
        if self.radial_theta is None:
            return self.l2(F.silu(g))
        g32 = g.float()
        r = torch.sqrt(g32.square().mean(-1, keepdim=True) + 1e-6)
        return self.l2((F.silu(g32 / r) * r.pow(torch.sigmoid(self.radial_theta.float()))).to(g.dtype))


class Linear(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.l = nn.Linear(d, c)

    def forward(self, h):
        return self.l(h)


class SWA(nn.Module):
    """Causal local attention: frame t attends to frames t-W+1 .. t (zero-padded before the start, masked)."""

    def __init__(self, d, c, w, heads=4):
        super().__init__()
        self.w, self.h = w, heads
        self.ln, self.q, self.kv, self.o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.mlp = MLP(d, c)

    def forward(self, h):
        b, t, d = h.shape
        x = self.ln(h)
        win = torch.stack([F.pad(x, (0, 0, k, 0))[:, :t] for k in range(self.w)], 2)      # (b, t, w, d): t, t-1, ..
        valid = torch.arange(t, device=h.device)[:, None] >= torch.arange(self.w, device=h.device)[None]   # (t, w)
        q = self.q(x).view(b, t, self.h, 1, d // self.h)
        k, v = self.kv(win).view(b, t, self.w, 2, self.h, d // self.h).unbind(3)
        k, v = k.transpose(2, 3), v.transpose(2, 3)                                         # (b, t, h, w, dh)
        s = (q * k).sum(-1) / math.sqrt(d // self.h)                                        # (b, t, h, w)
        s = s.masked_fill(~valid[None, :, None], float("-inf"))
        a = (s.softmax(-1)[..., None] * v).sum(3).reshape(b, t, d)
        return self.mlp(h + self.o(a))


class Mix(nn.Module):
    """Causal local attention as a residual block (no MLP): x + O(attend(LN(x)) over frames t-W+1 .. t)."""

    def __init__(self, d, w, heads=4):
        super().__init__()
        self.w, self.h = w, heads
        self.ln, self.q, self.kv, self.o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, h):
        b, t, d = h.shape
        x = self.ln(h)
        win = torch.stack([F.pad(x, (0, 0, k, 0))[:, :t] for k in range(self.w)], 2)
        valid = torch.arange(t, device=h.device)[:, None] >= torch.arange(self.w, device=h.device)[None]
        q = self.q(x).view(b, t, self.h, 1, d // self.h)
        k, v = self.kv(win).view(b, t, self.w, 2, self.h, d // self.h).unbind(3)
        k, v = k.transpose(2, 3), v.transpose(2, 3)
        s = ((q * k).sum(-1) / math.sqrt(d // self.h)).masked_fill(~valid[None, :, None], float("-inf"))
        return h + self.o((s.softmax(-1)[..., None] * v).sum(3).reshape(b, t, d))


class SwaSelfCond(nn.Module):
    """input -> SWA(w) -> vocab (pass 1, aux CTC) -> + embed(softmax(pass 1)) -> SWA(w) -> vocab (final). No MLP.
    Pass 2's window holds the CONDITIONED frames, so it attends over what the previous w-1 frames predicted."""

    def __init__(self, d, c, w=3):
        super().__init__()
        self.m1, self.l1, self.emb, self.m2, self.l2 = Mix(d, w), nn.Linear(d, c), nn.Linear(c, d, bias=False),             Mix(d, w), nn.Linear(d, c)
        self.aux = None

    def forward(self, h):
        x1 = self.m1(h)
        z1 = self.l1(x1)
        self.aux = z1
        return self.l2(self.m2(x1 + self.emb(z1.float().softmax(-1).to(x1.dtype))))


class SelfCondLin(nn.Module):
    """D without the MLP: h + cur(p) + prev(p_prev) -> Linear. Loss final_w * final + aux_w * pass 1."""

    def __init__(self, d, c, final_w=1.0, aux_w=0.3):
        super().__init__()
        self.final_w, self.aux_w = final_w, aux_w
        self.l1, self.l2 = nn.Linear(d, c), nn.Linear(d, c)
        self.cur, self.prev = nn.Linear(c, d, bias=False), nn.Linear(c, d, bias=False)
        self.aux = None

    def forward(self, h):
        z1 = self.l1(h)
        self.aux = z1
        p = z1.float().softmax(-1).to(h.dtype)
        return self.l2(h + self.cur(p) + self.prev(F.pad(p, (0, 0, 1, 0))[:, : p.shape[1]]))


class SelfCond3(nn.Module):
    """3-pass self-conditioned CTC: z1 = Linear(h); h2 = h + cur1(p1) + prev1(p1 of frame t-1); z2 = R2(h2);
    h3 = h2 + cur2(p2) + prev2(p2 of t-1); z3 = R3(h3). Loss 0.2 z1 + 0.3 z2 + 0.5 z3. R = MLP or Linear."""

    def __init__(self, d, c, mlp, radial=False):
        super().__init__()
        self.l1 = nn.Linear(d, c)
        self.r2, self.r3 = (MLP(d, c, radial), MLP(d, c, radial)) if mlp else (nn.Linear(d, c), nn.Linear(d, c))
        self.cond = nn.ModuleList([nn.ModuleDict({"cur": nn.Linear(c, d, bias=False), "prev": nn.Linear(c, d, bias=False)})
                                   for _ in range(2)])
        self.auxs, self.final_w = None, 0.5

    def _feed(self, h, z, k):
        p = z.float().softmax(-1).to(h.dtype)
        return h + self.cond[k]["cur"](p) + self.cond[k]["prev"](F.pad(p, (0, 0, 1, 0))[:, : p.shape[1]])

    def forward(self, h):
        z1 = self.l1(h)
        h2 = self._feed(h, z1, 0)
        z2 = self.r2(h2)
        z3 = self.r3(self._feed(h2, z2, 1))
        self.auxs = [(z1, 0.2), (z2, 0.3)]
        return z3


class SelfCond(nn.Module):
    def __init__(self, d, c, final_w=1.0, aux_w=0.3, radial=False):
        super().__init__()
        self.final_w, self.aux_w = final_w, aux_w
        self.l1 = nn.Linear(d, c)
        self.cur, self.prev = nn.Linear(c, d, bias=False), nn.Linear(c, d, bias=False)
        self.mlp = MLP(d, c, radial)
        self.aux = None

    def forward(self, h):
        z1 = self.l1(h)
        self.aux = z1
        p = z1.float().softmax(-1).to(h.dtype)
        p_prev = F.pad(p, (0, 0, 1, 0))[:, : p.shape[1]]
        return self.mlp(h + self.cur(p) + self.prev(p_prev))


def ctc(logits, lens, tgt, tlen, blank):
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(lp, tgt, lens, tlen, blank=blank, reduction="mean", zero_infinity=True)


@torch.no_grad()
def encode(m, audio, lens):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f, fl = m.preprocessor(input_signal=audio, length=lens)
        e, el = m.encoder(audio_signal=f, length=fl)
    return e.transpose(1, 2).float(), el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--steps", type=int, default=2700, help="~8 min on the RTX PRO 6000 (1,500 steps took 4.5 min)")
    ap.add_argument("--minutes", type=float, default=13, help="training wall-clock budget (sets --steps after 100)")
    ap.add_argument("--tok", default=None, help="tokenizer dir for the heads (default: the model's own)")
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--las", type=int, nargs="+", default=[0, 1, 3, 6, 13], help="look-aheads to score (80 ms frames)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_sec", type=float, default=1200)
    ap.add_argument("--val_per_source", type=int, default=200)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--heads", nargs="+", default=["A_linear", "B_mlp", "C2_swa2", "C4_swa4", "D_selfcond"])
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf, open_dict
    from score import normalize, wer
    import blank_penalty
    torch.manual_seed(a.seed); random.seed(a.seed)
    m = nemo_asr.models.ASRModel.restore_from(a.nemo, map_location="cuda").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    old_tok, old_ctc = m.tokenizer, copy.deepcopy(m.ctc_decoder)       # run5's own head = the reference row
    lock = blank_penalty.devanagari_ids(os.path.join(os.path.dirname(a.nemo), "..", "..", "run1", "tok",
                                                      "tokenizer_spe_bpe_v4096", "tokenizer.model"))
    if a.tok:   # heads train from scratch: any vocabulary (the English 1,024 BPE), the frozen encoder does not care
        m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
        for p in m.parameters():
            p.requires_grad_(False)
    d, c = m.cfg.encoder.d_model, m.tokenizer.vocab_size + 1
    blank = c - 1
    print(f"[vocab] heads: {c - 1} tokens + blank ({a.tok or 'run5 tokenizer'}); run5 reference: "
          f"{old_ctc.decoder_layers[0].out_channels - 1} + blank", flush=True)
    zoo = {"A_linear": lambda: Linear(d, c), "B_mlp": lambda: MLP(d, c), "C2_swa2": lambda: SWA(d, c, 2),
           "C4_swa4": lambda: SWA(d, c, 4), "D_selfcond": lambda: SelfCond(d, c),
           "E_swa3_sc": lambda: SwaSelfCond(d, c, 3), "F_selfcond_lin": lambda: SelfCondLin(d, c, 0.7, 0.3),
           "J_selfcond_37": lambda: SelfCond(d, c, 0.7, 0.3),
           "G_sc3_mlp": lambda: SelfCond3(d, c, True), "H_sc3_lin": lambda: SelfCond3(d, c, False),
           "Br_mlp_radial": lambda: MLP(d, c, radial=True),
           "Dr_selfcond_radial": lambda: SelfCond(d, c, radial=True),
           "Jr_selfcond_radial": lambda: SelfCond(d, c, 0.7, 0.3, radial=True),
           "Gr_sc3_radial": lambda: SelfCond3(d, c, True, radial=True)}
    heads = nn.ModuleDict({k: zoo[k]() for k in a.heads}).cuda()
    for k, h in heads.items():
        print(f"[heads] {k}: {sum(p.numel() for p in h.parameters()) / 1e6:.2f}M", flush=True)
    theta = [p for n, p in heads.named_parameters() if n.endswith("radial_theta")]
    rest = [p for n, p in heads.named_parameters() if not n.endswith("radial_theta")]
    # radial theta: its own lr 0.01, no decay (BiBo: p needs its own lr)
    opt = torch.optim.AdamW([{"params": rest}, {"params": theta, "lr": 0.01, "weight_decay": 0.0}], lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s / a.steps, 1))))
    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=30,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=12, shuffle_buffer_size=10000, seed=a.seed, pin_memory=True, shard_seed="randomized",
                  concurrent_bucketing=False)
    m.setup_training_data(tr)
    rs, ps = list(MIX), list(MIX.values())
    t0, step, run = time.time(), 0, {k: 0.0 for k in heads}
    def batches():                                   # > 1 epoch when --steps asks for it, reshuffled per epoch
        for ep in range(1000):
            sm = getattr(m._train_dl.dataset, "sampler", None) or m._train_dl.sampler
            sm.set_epoch(ep)
            yield from m._train_dl

    for batch in batches():
        audio, alen, tgt, tlen = (x.cuda(non_blocking=True) for x in batch[:4])
        set_lookahead(m.encoder, random.choices(rs, ps)[0])
        h, hl = encode(m, audio, alen)
        loss = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k, head in heads.items():
                li = getattr(head, "final_w", 1.0) * ctc(head(h), hl, tgt, tlen, blank)
                if getattr(head, "aux", None) is not None:
                    li = li + getattr(head, "aux_w", 0.3) * ctc(head.aux, hl, tgt, tlen, blank)
                for z, w in getattr(head, "auxs", None) or []:                     # multi-pass heads
                    li = li + w * ctc(z, hl, tgt, tlen, blank)
                run[k] += li.item()
                loss = loss + li
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            print(f"[train] step {step} {time.time() - t0:.0f}s " + " ".join(f"{k} {v / 100:.3f}" for k, v in run.items()),
                  flush=True)
            run = {k: 0.0 for k in heads}
        if step == 100:
            t100 = time.time()
        if step == 300 and a.minutes:                # size the run to the wall-clock budget from the measured rate
            rate = 200 / (time.time() - t100)          # steps 100-300: past the data-loader warm-up (0-100 read 3.3/s, real 5.9/s)
            a.steps = int(300 + (a.minutes * 60 - (time.time() - t0)) * rate)
            print(f"[train] {rate:.2f} steps/s -> {a.steps} steps for {a.minutes} min", flush=True)
        if step >= a.steps:
            break
    heads.eval()
    # name -> (logits fn, tokenizer, blank id, logit bias): the new heads have no Devanagari tokens to lock
    every = {k: (v, m.tokenizer, blank, 0.0) for k, v in heads.items()}
    every["run5"] = (lambda h: old_ctc(encoder_output=h.transpose(1, 2)).float(), old_tok,
                     old_ctc.decoder_layers[0].out_channels - 1, lock)
    vals = []
    for p in sorted(sum((glob.glob(v) for v in a.val), [])):
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        random.Random(0).shuffle(rows)
        vals += rows[: a.val_per_source]
    vals.sort(key=lambda r: r["duration"])
    refs = [normalize(x["text"]) for x in vals]
    # audio read ONCE, padded batches built ONCE (was: re-read from disk at every look-ahead, batches of 32)
    vb = []
    for i in range(0, len(vals), a.eval_batch):
        xs = [sf.read(x["audio_filepath"])[0].astype(np.float32) for x in vals[i:i + a.eval_batch]]
        L = torch.tensor([len(x) for x in xs])
        A = torch.zeros(len(xs), int(L.max()))
        for j, x in enumerate(xs):
            A[j, : len(x)] = torch.from_numpy(x)
        vb.append((A, L, i))
    meet = [(sf.read(os.path.join(a.meeting, w))[0].astype(np.float32), normalize(open(os.path.join(a.meeting, r), encoding="utf-8").read()))
            for w, r in MEETING]
    words = [len(r) for _, r in meet]
    print(f"[eval] val {len(vals)} utterances, {len(vb)} batches; meeting clips {words} words; look-aheads {a.las}", flush=True)

    def texts(z, hl, tok, blank):
        """Greedy CTC on the GPU for a whole batch: argmax, drop blanks and repeats, ONE host copy."""
        ids = z.argmax(-1)
        valid = torch.arange(ids.shape[1], device=ids.device)[None] < hl[:, None]
        keep = (ids != blank) & (ids != F.pad(ids, (1, 0), value=-1)[:, :-1]) & valid
        ids, keep = ids.cpu().numpy(), keep.cpu().numpy()
        return [normalize(tok.ids_to_text(ids[j][keep[j]].tolist())) for j in range(len(ids))]

    for r in a.las:
        t1 = time.time()
        set_lookahead(m.encoder, r)
        errs = {k: [0, 0] for k in every}
        mt = {k: [] for k in every}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for A, L, i in vb:
                h, hl = encode(m, A.cuda(non_blocking=True), L.cuda())
                for k, (f, tok, bl, bias) in every.items():
                    for j, hyp in enumerate(texts(f(h).float() + bias, hl, tok, bl)):
                        ref = refs[i + j]
                        errs[k][0] += wer(ref, hyp) * len(ref); errs[k][1] += len(ref)
            for x, ref in meet:
                h, hl = encode(m, torch.from_numpy(x)[None].cuda(), torch.tensor([len(x)]).cuda())
                for k, (f, tok, bl, bias) in every.items():
                    hyp = texts(f(h).float() + bias, hl, tok, bl)[0]
                    mt[k].append((wer(ref, hyp), len(hyp)))
        for k in every:
            pooled = sum(w * n for (w, _), n in zip(mt[k], words)) / sum(words)
            print(f"RESULT look-ahead {r * 80:4d} ms {k:11s} | val {100 * errs[k][0] / errs[k][1]:5.2f} | meeting pooled "
                  f"{100 * pooled:5.2f} | " + " ".join(f"{100 * w:5.1f}({n})" for w, n in mt[k]), flush=True)
        print(f"[eval] look-ahead {r * 80} ms took {time.time() - t1:.0f}s", flush=True)
    print("HEADS_DONE", flush=True)


if __name__ == "__main__":
    main()
