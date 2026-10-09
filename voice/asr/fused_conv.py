"""NeMo ConformerConvolution + ConformerFeedForward elementwise middles -> tkf kernels (train_asr.py --fused_conv).

triton-kernel-fused kernels/sm120/conv_module.py: GLU + pad mask + causal depthwise conv + LayerNorm + SiLU between
the two pointwise linears, channel-last (no transposes, no fp32 cast copies); 6.2x fwd+bwd vs eager at a 30k-frame
batch. kernels/sm120/silu_dropout.py: the FFN's SiLU + dropout, one pass each way (mask from a seed); 2.0x.
Both round like eager under bf16 autocast (parity_check/parity_conv_module.py: ~1e-5 vs eager, equal distance to fp64).
Streaming caches / other norms / activations call NeMo's forward.

    import fused_conv; fused_conv.enable(model)
"""
import os
import sys
import types

import torch

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def enable(model):
    sys.path.insert(0, TKF)
    from kernels.sm120.conv_module import conv_module
    from kernels.sm120.silu_dropout import silu_dropout

    def conv_forward(self, x, pad_mask=None, cache=None):
        if cache is not None:
            return self._nemo_forward(x, pad_mask=pad_mask, cache=cache)
        g = self._apply_pointwise(x, self.pointwise_conv1)
        dc = self.depthwise_conv
        y = conv_module(g, pad_mask, dc.weight, dc.bias, self.batch_norm.weight, self.batch_norm.bias,
                        self.batch_norm.eps, dc._left_padding,
                        out_dtype=torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None)
        return self._apply_pointwise(y, self.pointwise_conv2)

    def ff_forward(self, x):
        h = silu_dropout(self.linear1(x), self.dropout.p if self.dropout.training else 0.0)
        return self.linear2(h)

    nc = nf = 0
    for layer in model.encoder.layers:
        c = layer.conv
        dc = c.depthwise_conv
        if (c.norm_type == "layer_norm" and c.pointwise_activation == "glu_" and dc.bias is not None
                and dc.stride == (1,) and dc.dilation == (1,) and dc._left_padding + dc._right_padding == dc.kernel_size[0] - 1):
            c._nemo_forward = c.forward
            c.forward = types.MethodType(conv_forward, c)
            nc += 1
        for ff in (layer.feed_forward1, layer.feed_forward2):
            if isinstance(ff.activation, torch.nn.SiLU):
                ff._nemo_forward = ff.forward
                ff.forward = types.MethodType(ff_forward, ff)
                nf += 1
    print(f"[fused_conv] {nc} conv modules -> tkf conv_module, {nf} feed-forwards -> tkf silu_dropout", flush=True)


def _check():
    """run1 model, eval mode (no dropout), bf16 autocast: each conv module / FFN NeMo vs fused on the same input."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from("/home/marimo/work/asr/exp/run1/run1.nemo").cuda().eval()
    c0 = m.encoder.layers[0].conv
    print("conv:", c0.norm_type, "K", c0.depthwise_conv.kernel_size, "pad", c0.depthwise_conv._left_padding,
          c0.depthwise_conv._right_padding, flush=True)
    torch.manual_seed(0)
    audio = torch.randn(8, 16000 * 12, device="cuda") * 0.05
    alen = torch.tensor([16000 * 12 - 16000 * i for i in range(8)], device="cuda")
    caps = {}
    hooks = [m.encoder.layers[i].conv.register_forward_pre_hook(
        lambda mod, a, kw, i=i: caps.setdefault(i, (a, kw)), with_kwargs=True) for i in (0, 8, 16)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m.forward(input_signal=audio, input_signal_length=alen)
    for h in hooks:
        h.remove()
    enable(m)
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    for i, (a, kw) in caps.items():
        L = m.encoder.layers[i]
        x0, pm = a[0].detach(), kw.get("pad_mask", a[1] if len(a) > 1 else None)
        for name, mod, call in (("conv", L.conv, lambda f, x: f(x, pad_mask=pm)), ("ff1", L.feed_forward1, lambda f, x: f(x))):
            res = {}
            for which in ("nemo", "fused"):
                mod.zero_grad()
                x = x0.clone().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = call(mod._nemo_forward if which == "nemo" else mod.forward, x)
                (y.float() * torch.linspace(-1, 1, y.shape[-1], device="cuda")).sum().backward()
                res[which] = [y.detach(), x.grad] + [p.grad.clone() for p in mod.parameters()]
            print(f"layer {i} {name}: out {rel(res['fused'][0], res['nemo'][0]):.1e}  dx {rel(res['fused'][1], res['nemo'][1]):.1e}  "
                  f"worst param grad {max(rel(f, n) for f, n in zip(res['fused'][2:], res['nemo'][2:])):.1e}", flush=True)


if __name__ == "__main__" and "--check" in sys.argv:
    _check()
