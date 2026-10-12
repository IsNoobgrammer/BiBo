"""NeMo ConformerConvolution + ConformerFeedForward elementwise middles -> tkf kernels (train_asr.py --fused_conv).

triton-kernel-fused kernels/sm120/conv_module.py: GLU + pad mask + causal depthwise conv + LayerNorm + SiLU between
the two pointwise linears, channel-last (no transposes, no fp32 cast copies); 6.2x fwd+bwd vs eager at a 30k-frame
batch. kernels/sm120/silu_dropout.py: the FFN's SiLU + dropout, one pass each way (mask from a seed); 2.0x.
Both round like eager under bf16 autocast (parity_check/parity_conv_module.py: ~1e-5 vs eager, equal distance to fp64).
BatchNorm (Parakeet) in training -> conv_module_bn; eval / streaming caches / other norms call NeMo's forward.

    import fused_conv; fused_conv.enable(model)
"""
import os
import sys
import types

import torch

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def enable(model):
    sys.path.insert(0, TKF)
    from kernels.sm120.conv_module import conv_module, conv_module_bn
    from kernels.sm120.silu_dropout import silu_dropout

    def conv_forward(self, x, pad_mask=None, cache=None):
        if cache is not None or (self.norm_type == "batch_norm" and not self.training):
            return self._nemo_forward(x, pad_mask=pad_mask, cache=cache)
        g = self._apply_pointwise(x, self.pointwise_conv1)
        dc = self.depthwise_conv
        cb = dc.bias if dc.bias is not None else self._fused_zero_b
        odt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
        n = self.batch_norm
        if self.norm_type == "batch_norm":       # training: batch stats + running-stat update (eval: NeMo's forward)
            y = conv_module_bn(g, pad_mask, dc.weight, cb, n.weight, n.bias, n.running_mean, n.running_var, n.momentum,
                               n.eps, dc._left_padding, out_dtype=odt)
            n.num_batches_tracked.add_(1)
        else:
            y = conv_module(g, pad_mask, dc.weight, cb, n.weight, n.bias, n.eps, dc._left_padding, out_dtype=odt)
        return self._apply_pointwise(y, self.pointwise_conv2)

    def ff_forward(self, x):
        h = silu_dropout(self.linear1(x), self.dropout.p if self.dropout.training else 0.0)
        return self.linear2(h)

    nc = nf = 0
    for layer in model.encoder.layers:
        c = layer.conv
        dc = c.depthwise_conv
        bn_ok = (c.norm_type == "batch_norm" and c.batch_norm.track_running_stats and c.batch_norm.momentum is not None
                 and c.batch_norm.affine)       # Parakeet: kernels/sm120/conv_module.py conv_module_bn
        if ((c.norm_type == "layer_norm" or bn_ok) and c.pointwise_activation == "glu_" and c.batch_norm.bias is not None
                and dc.stride == (1,) and dc.dilation == (1,) and getattr(dc, "_left_padding", None) is not None
                and dc._left_padding + dc._right_padding == dc.kernel_size[0] - 1):
            c._nemo_forward = c.forward
            # bias-free convs (Nemotron / Parakeet use_bias false): a zero bias adds exactly 0 in fp32, no grad
            if dc.bias is None:                  # non-persistent buffer: follows .cuda(), not in state_dict
                c.register_buffer("_fused_zero_b", torch.zeros(dc.out_channels, dtype=dc.weight.dtype,
                                                               device=dc.weight.device), persistent=False)
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
    src = os.environ.get("CHECK_MODEL", "/home/marimo/work/asr/exp/run1/run1.nemo")     # .nemo path or a hub id
    m = (nemo_asr.models.ASRModel.restore_from(src) if src.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(src)).cuda().eval()
    if os.environ.get("CHECK_STREAMING"):                 # a full-context hub model as train_asr --hub_streaming makes it
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hub_models
        m = hub_models.to_streaming(m, [13, 6, 3, 1, 0, -1]).cuda().eval()
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
        L.conv.train(L.conv.norm_type == "batch_norm")   # BN: training mode (batch stats) on both sides
        for name, mod, call in (("conv", L.conv, lambda f, x: f(x, pad_mask=pm)), ("ff1", L.feed_forward1, lambda f, x: f(x))):
            res = {}
            Wt = None
            for which in ("nemo", "fused", "fp64"):
                if which == "fp64":                       # ground truth: the module in fp64, NeMo's own forward
                    import copy
                    m64 = copy.deepcopy(mod).double()
                    m64.zero_grad()
                    x = x0.double().clone().requires_grad_()
                    y = type(mod).forward(m64, x, pad_mask=pm) if name == "conv" else type(mod).forward(m64, x)
                    params = m64.parameters()
                else:
                    mod.zero_grad()
                    x = x0.clone().requires_grad_()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = call(mod._nemo_forward if which == "nemo" else mod.forward, x)
                    params = mod.parameters()
                if Wt is None:                            # a random fixed output weight (a constant per channel is a
                    gen = torch.Generator(device="cuda").manual_seed(0)   # degenerate case for BatchNorm's backward)
                    Wt = torch.randn(y.shape, device="cuda", generator=gen)
                (y.double() * Wt.double()).sum().backward()
                res[which] = [y.detach(), x.grad] + [q.grad.clone() for q in params]
            t = res["fp64"]
            for which in ("nemo", "fused"):
                r = res[which]
                print(f"layer {i} {name} {which:>5} vs fp64: out {rel(r[0], t[0]):.1e}  dx {rel(r[1], t[1]):.1e}  "
                      f"worst param grad {max(rel(f, n) for f, n in zip(r[2:], t[2:])):.1e}", flush=True)
            print(f"layer {i} {name} fused vs nemo: out {rel(res['fused'][0], res['nemo'][0]):.1e}  "
                  f"dx {rel(res['fused'][1], res['nemo'][1]):.1e}", flush=True)

if __name__ == "__main__" and "--check" in sys.argv:
    _check()
