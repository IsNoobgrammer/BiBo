"""NeMo ConformerLayer with its 5 residual / dropout / LayerNorm boundaries fused (train_asr.py --fused_layer).

Each `residual = residual + dropout(x) * factor; y = LayerNorm(residual)` (4 per layer, + the layer's first LN)
becomes one tkf kernel forward and one backward (triton-kernel-fused kernels/sm120/res_drop_ln.py), which rounds
like eager: bit-identical LN weight / bias grads in bf16 mode, ~1e-5 elsewhere (fp32 LN reduction order).
Only the plain training path is replaced (rel_pos attention, no streaming caches, no adapters / tensor access);
anything else calls NeMo's original forward. Dropout draws its own mask (Triton RNG).

    import fused_layer; fused_layer.enable(model)
"""
import os
import sys
import types

import torch

TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def enable(model):
    sys.path.insert(0, TKF)
    from kernels.sm120.res_drop_ln import res_dropout_layernorm as rdl

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        if (cache_last_channel is not None or cache_last_time is not None or self.self_attention_model != "rel_pos"
                or self.is_adapter_available() or self.is_access_enabled(getattr(self, "model_guid", None))):
            return self._nemo_forward(x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
                                      cache_last_channel=cache_last_channel, cache_last_time=cache_last_time)
        p = self.dropout.p if self.dropout.training else 0.0               # follows the Dropout module's mode
        ydt = torch.float32 if torch.is_autocast_enabled("cuda") else None    # autocast LN outputs fp32
        ln = lambda n: (n.weight, n.bias, n.eps)
        _, y = rdl(x, None, *ln(self.norm_feed_forward1), y_dtype=ydt)
        res, y = rdl(x, self.feed_forward1(y), *ln(self.norm_self_att), p=p, factor=self.fc_factor, y_dtype=ydt)
        h = self.self_attn(query=y, key=y, value=y, mask=att_mask, pos_emb=pos_emb, cache=None)
        res, y = rdl(res, h, *ln(self.norm_conv), p=p, y_dtype=ydt)
        res, y = rdl(res, self.conv(y, pad_mask=pad_mask, cache=None), *ln(self.norm_feed_forward2), p=p, y_dtype=ydt)
        _, y = rdl(res, self.feed_forward2(y), *ln(self.norm_out), p=p, factor=self.fc_factor, y_dtype=ydt)
        return y

    n = 0
    for layer in model.encoder.layers:
        layer._nemo_forward = layer.forward
        layer.forward = types.MethodType(forward, layer)
        n += 1
    print(f"[fused_layer] {n} conformer layers: residual + dropout + LayerNorm -> tkf res_drop_ln", flush=True)


def _check():
    """run1 model in eval mode (no dropout / stochastic depth), an encoder-only loss: NeMo vs fused, + noise row."""
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from("/home/marimo/work/asr/exp/run1/run1.nemo").cuda().eval()
    print("stochastic depth:", getattr(m.encoder, "layer_drop_probs", None), flush=True)
    torch.manual_seed(0)
    audio = torch.randn(8, 16000 * 6, device="cuda") * 0.05
    alen = torch.tensor([16000 * 6 - 4000 * i for i in range(8)], device="cuda")
    res = {}
    for name in ("nemo", "nemo again", "fused"):
        if name == "fused":
            enable(m)
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, _ = m.forward(input_signal=audio, input_signal_length=alen)
        loss = (enc.float() * torch.linspace(-1, 1, enc.shape[1], device="cuda")[None, :, None]).sum()
        loss.backward()
        res[name] = (enc.detach().float(), m.encoder.layers[3].norm_conv.weight.grad.float().clone(),
                     m.encoder.layers[0].feed_forward1.linear1.weight.grad.float().clone())
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    for other in ("nemo again", "fused"):
        print(f"{other} vs nemo: encoder out rel %.2e   d norm_conv.w (L3) rel %.2e   d ff1.linear1 (L0) rel %.2e" %
              tuple(rel(a, b) for a, b in zip(res[other], res["nemo"])), flush=True)


if __name__ == "__main__" and "--check" in sys.argv:
    _check()
