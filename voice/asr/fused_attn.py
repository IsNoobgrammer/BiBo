"""NeMo RelPositionMultiHeadAttention -> tkf fused banded relative-position attention (train_asr.py --fused_attn).

triton-kernel-fused kernels/sm120/relpos_attn.py: the chunked_limited band only, no T x T scores / rel_shift copies,
in NeMo's own dtypes (bf16 under bf16 autocast; avoid_float16_autocast_context only acts on fp16). 1.3x (4 s clips) to 4.4x (25 s) fwd+bwd vs NeMo's
core (bench/bench_relpos_attn.py); parity_check/parity_relpos_attn.py gates it against fp64 and NeMo-fp32.

The kernel needs the batch lengths and this step's att_context_size (training samples one of [70,13] [70,6] [70,1]
[70,0] per batch), not the (B, T, T) mask: ConformerEncoder._create_masks is wrapped to record both, and every
attention layer reads them. Anything else (streaming cache, other attention styles, unlimited right context, batched
pos_emb) calls NeMo's forward.

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
    if enc.att_context_style != "chunked_limited" or enc.self_attention_model != "rel_pos":
        print(f"[fused_attn] {enc.att_context_style} / {enc.self_attention_model}: not supported, NeMo attention kept")
        return
    nemo_masks = enc._create_masks

    def create_masks(att_context_size, padding_length, max_audio_length, offset, device):
        enc._fused_ctx = (list(att_context_size), padding_length, offset)
        return nemo_masks(att_context_size, padding_length, max_audio_length, offset, device)

    enc._create_masks = create_masks

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        ctx = getattr(enc, "_fused_ctx", None)
        B, T = query.shape[0], query.shape[1]
        if (cache is not None or ctx is None or ctx[2] is not None or ctx[0][1] < 0 or pos_emb.size(0) != 1
                or pos_emb.size(1) != 2 * T - 1 or key is not query or value is not query):
            return self._nemo_forward(query, key, value, mask, pos_emb, cache=cache)
        (left, right), lengths, _ = ctx
        # exactly NeMo's dtypes: under bf16 autocast its avoid_float16_autocast_context is a no-op, so the projections
        # and the attention matmuls are bf16 (softmax fp32); the kernel follows q's dtype
        x = query.float() if torch.is_autocast_enabled() else query
        q = self.linear_q(x).view(B, T, self.h, self.d_k)
        k = self.linear_k(x).view(B, T, self.h, self.d_k)
        v = self.linear_v(x).view(B, T, self.h, self.d_k)
        p = self.linear_pos(pos_emb).view(-1, self.h, self.d_k)
        o = relpos_attention(q, k, v, p, self.pos_bias_u, self.pos_bias_v, lengths, left, right,
                             dropout=self.dropout.p if self.dropout.training else 0.0)
        return self.linear_out(o.reshape(B, T, self.h * self.d_k))

    n = 0
    for layer in enc.layers:
        att = layer.self_attn
        att._nemo_forward = att.forward
        att.forward = types.MethodType(forward, att)
        n += 1
    print(f"[fused_attn] {n} rel_pos attention layers -> tkf relpos_attention", flush=True)


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
