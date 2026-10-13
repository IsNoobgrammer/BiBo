"""NeMo RelPositionMultiHeadAttention -> tkf fused banded relative-position attention (train_asr.py --fused_attn).

triton-kernel-fused kernels/sm120/relpos_attn.py: the chunked_limited band only, no T x T scores / rel_shift copies,
in NeMo's own dtypes (bf16 under bf16 autocast; avoid_float16_autocast_context only acts on fp16). 1.3x (4 s clips) to 4.4x (25 s) fwd+bwd vs NeMo's
core (bench/bench_relpos_attn.py); parity_check/parity_relpos_attn.py gates it against fp64 and NeMo-fp32.

The kernel needs the batch lengths and this step's att_context_size (training samples one of [70,13] [70,6] [70,1]
[70,0] per batch), not the (B, T, T) mask: ConformerEncoder._create_masks is wrapped to record both, and every
attention layer reads them. Anything else (streaming cache, other attention styles, a right-unlimited band [left >= 0, -1],
batched pos_emb) calls NeMo's forward. Full context [-1, -1] (offline / voice typing) runs in the kernel as one chunk.

    import fused_attn; fused_attn.enable(model)
"""
import os
import sys
import types

import torch

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def enable(model):
    sys.path.insert(0, TKF)
    from kernels.sm120.relpos_attn import relpos_attention
    enc = model.encoder
    if enc.att_context_style not in ("chunked_limited", "regular") or enc.self_attention_model != "rel_pos":
        print(f"[fused_attn] {enc.att_context_style} / {enc.self_attention_model}: not supported, NeMo attention kept")
        return
    regular = enc.att_context_style == "regular"     # Parakeet: at full context [-1, -1] its mask is the padding only,
                                                     # the same as chunked_limited's; any other regular band -> NeMo
    nemo_masks = enc._create_masks

    def create_masks(att_context_size, padding_length, max_audio_length, offset, device):
        enc._fused_ctx = (list(att_context_size), padding_length, offset)
        return nemo_masks(att_context_size, padding_length, max_audio_length, offset, device)

    enc._create_masks = create_masks

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        ctx = getattr(enc, "_fused_ctx", None)
        B, T = query.shape[0], query.shape[1]
        if (cache is not None or ctx is None or ctx[2] is not None or (ctx[0][1] < 0 and ctx[0][0] >= 0)
                or getattr(enc, "_fused_off", False)        # parity checks: force NeMo's attention
                or pos_emb.size(0) != 1
                or pos_emb.size(1) != 2 * T - 1 or key is not query or value is not query):
            return self._nemo_forward(query, key, value, mask, pos_emb, cache=cache)
        (left, right), lengths, _ = ctx
        if regular and (left >= 0 or right >= 0):
            return self._nemo_forward(query, key, value, mask, pos_emb, cache=cache)
        enc._fused_calls += 1                    # parity / plumbing checks assert the kernel really ran
        # NeMo's dtypes: under bf16 autocast its avoid_float16_autocast_context is a no-op, so the projections and the
        # attention matmuls are bf16 (softmax fp32); the kernel follows q's dtype. q, k, v = ONE GEMM (one input cast,
        # and the input grad is a single fp32-accumulated GEMM instead of three bf16 grads summed)
        lq, lk, lv = self.linear_q, self.linear_k, self.linear_v
        qkv = torch.nn.functional.linear(query, torch.cat([lq.weight, lk.weight, lv.weight]),   # bias-free models
                                         None if lq.bias is None else torch.cat([lq.bias, lk.bias, lv.bias]))  # (Nemotron)
        q, k, v = qkv.view(B, T, 3, self.h, self.d_k).unbind(2)
        p = self.linear_pos(pos_emb).view(-1, self.h, self.d_k)
        o = relpos_attention(q, k, v, p, self.pos_bias_u, self.pos_bias_v, lengths, left, right,
                             dropout=self.dropout.p if self.dropout.training else 0.0)
        return self.linear_out(o.reshape(B, T, self.h * self.d_k))

    enc._fused_calls = 0
    n = 0
    for layer in enc.layers:
        att = layer.self_attn
        att._nemo_forward = att.forward
        att.forward = types.MethodType(forward, att)
        n += 1
    print(f"[fused_attn] {n} rel_pos attention layers -> tkf relpos_attention"
          + (" (regular style: full context only)" if regular else ""), flush=True)


def _check():
    """run1 model, eval mode, real masks: each attention layer NeMo vs fused on the same input (fwd + grads)."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from("/home/marimo/work/asr/exp/run1/run1.nemo").cuda().eval()
    torch.manual_seed(0)
    audio = torch.randn(8, 16000 * 12, device="cuda") * 0.05
    alen = torch.tensor([16000 * 12 - 16000 * i for i in range(8)], device="cuda")
    enable(m)
    caps = {}
    hooks = [m.encoder.layers[i].self_attn.register_forward_pre_hook(
        lambda mod, a, kw, i=i: caps.setdefault(i, (a, kw)), with_kwargs=True) for i in (0, 8, 16)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m.forward(input_signal=audio, input_signal_length=alen)
    for h in hooks:
        h.remove()
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    for i, (a, kw) in caps.items():
        att = m.encoder.layers[i].self_attn
        args = dict(zip(("query", "key", "value", "mask", "pos_emb"), a)) | kw
        x0 = args["query"].detach()
        res = {}
        for name, fn in (("nemo", att._nemo_forward), ("fused", att.forward)):
            att.zero_grad()
            x = x0.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = fn(x, x, x, args["mask"], args["pos_emb"], cache=None)
            (y.float() * torch.linspace(-1, 1, y.shape[-1], device="cuda")).sum().backward()
            res[name] = (y.detach(), x.grad, att.linear_q.weight.grad.clone(), att.pos_bias_v.grad.clone(),
                         att.linear_pos.weight.grad.clone())
        n, f = res["nemo"], res["fused"]
        print(f"layer {i}: out {rel(f[0], n[0]):.2e}  dx {rel(f[1], n[1]):.2e}  dWq {rel(f[2], n[2]):.2e}  "
              f"d pos_bias_v {rel(f[3], n[3]):.2e}  dW_pos {rel(f[4], n[4]):.2e}  (context {m.encoder._fused_ctx[0]})",
              flush=True)


if __name__ == "__main__" and "--check" in sys.argv:
    _check()


def _check_pk():
    """Parakeet TDT 0.6B v2 (regular rel_pos, full context), no grad: (1) per layer NeMo bf16 vs fused bf16 vs the
    layer in fp64 on the same input; (2) end to end, the teacher's per-token KD targets on real clips with fused vs
    NeMo attention, and vs an fp32 NeMo forward. Prints PKATTN lines."""
    import json
    import random
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    from kernels.sm120.tdt_joint import tdt_joint_pos_targets
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
    sys.path.insert(0, TKF)
    enable(m)
    enc = m.encoder
    rows = [json.loads(l) for l in open("/home/marimo/work/asr/en3/train_cleanA.jsonl")]
    random.Random(5).shuffle(rows)
    rows = [r for r in rows if 3 <= r["duration"] <= 20][:16]
    wav = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in rows]
    L = max(len(w) for w in wav)
    sig = torch.zeros(len(wav), L, device="cuda")
    for j, w in enumerate(wav):
        sig[j, :len(w)] = torch.from_numpy(w)
    slen = torch.tensor([len(w) for w in wav], device="cuda")
    ys = [m.tokenizer.text_to_ids(r["text"]) for r in rows]
    y = torch.zeros(len(ys), max(len(t) for t in ys), dtype=torch.long, device="cuda")
    for j, t in enumerate(ys):
        y[j, :len(t)] = torch.tensor(t)
    ylen = torch.tensor([len(t) for t in ys], device="cuda")
    rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
    ok = True
    caps = {}
    hooks = [enc.layers[i].self_attn.register_forward_pre_hook(
        lambda mod, a, kw, i=i: caps.setdefault(i, (a, kw)), with_kwargs=True) for i in (0, 11, 23)]

    def run(fused, amp=True):
        enc._fused_off = not fused
        enc._fused_calls = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            e, el = m.forward(input_signal=sig, input_signal_length=slen)
            d, _, _ = m.decoder(targets=y, target_length=ylen)
            f = m.joint.project_encoder(e.transpose(1, 2))
            g = m.joint.project_prednet(d.transpose(1, 2))
        lin = m.joint.joint_net[-1]
        P, nll = tdt_joint_pos_targets(f, g, lin.weight, lin.bias, y, el, ylen, [0, 1, 2, 3, 4])
        return e.float(), P, nll, enc._fused_calls

    ef, Pf, nf, calls = run(True)
    for h in hooks:
        h.remove()
    for i, (a, kw) in sorted(caps.items()):
        att = enc.layers[i].self_attn
        args = dict(zip(("query", "key", "value", "mask", "pos_emb"), a)) | kw
        x = args["query"].detach()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yn = att._nemo_forward(x, x, x, args["mask"], args["pos_emb"], cache=None)
                yf = att.forward(x, x, x, args["mask"], args["pos_emb"], cache=None)
            att64 = __import__("copy").deepcopy(att).double()
            x64 = x.double()
            y64 = type(att).forward(att64, x64, x64, x64, args["mask"], args["pos_emb"].double(), cache=None)
        en, ef_ = rel(yn, y64), rel(yf, y64)
        r = ef_ <= 1.1 * en
        ok &= r
        print(f"PKATTN layer {i}: vs fp64 rel  NeMo bf16 {en:.2e}  fused bf16 {ef_:.2e}  [{'PASS' if r else 'FAIL'}]", flush=True)
    en_, Pn, nn_, _ = run(False)
    e32, P32, n32, _ = run(False, amp=False)
    m_ = torch.zeros_like(Pf[..., 0], dtype=torch.bool)
    for j in range(len(ys)):
        m_[j, :len(ys[j])] = True
    dfn = (Pf - Pn)[m_].abs().max().item()
    dn32 = (Pn - P32)[m_].abs().max().item()
    df32 = (Pf - P32)[m_].abs().max().item()
    r = calls == len(enc.layers) and df32 <= 1.1 * dn32 + 1e-3
    ok &= r
    print(f"PKATTN e2e: fused calls {calls}/{len(enc.layers)} | encoder rel vs fp32: NeMo bf16 {rel(en_, e32):.2e} "
          f"fused bf16 {rel(ef, e32):.2e} | KD targets max abs: fused-NeMo {dfn:.2e}, NeMo bf16-fp32 {dn32:.2e}, "
          f"fused-fp32 {df32:.2e} | nll mean fused {nf.mean().item():.3f} NeMo {nn_.mean().item():.3f} fp32 "
          f"{n32.mean().item():.3f} [{'PASS' if r else 'FAIL'}]", flush=True)
    print("PKATTN ALL PASS" if ok else "PKATTN FAILED", flush=True)


if __name__ == "__main__" and "--check_pk" in sys.argv:
    _check_pk()
