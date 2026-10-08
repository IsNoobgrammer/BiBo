"""Cost of each piece of ONE conformer layer (run1 model, a real Lhotse batch), forward + backward, in isolation.
Picks the next kernel targets. Measures only.

    python voice/asr/bench_layer.py --train R/train.jsonl [--layer 8] [--bf16]

Pieces (x17 layers per step): the whole layer; feed-forward 1 and 2 (LN + MLP); self-attention (LN + rel-pos MHA);
conv module (LN + pointwise -> GLU -> transpose -> depthwise -> BatchNorm -> transpose -> Swish -> pointwise);
and the residual + dropout + LayerNorm glue alone. For each: ms fwd+bwd and its top kernels.
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import triton
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--nemo", default="/home/marimo/work/asr/exp/run1/run1.nemo")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--batch_sec", type=float, default=1200)
    ap.add_argument("--bf16", action="store_true", help="bf16 weights, no autocast (--bf16_master)")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(a.nemo).cuda().train()
    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=30,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=4, shuffle_buffer_size=10000, seed=23)
    m.setup_training_data(tr)
    if a.bf16:
        import bf16_master
        bf16_master.to_bf16(m)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.bf16))
    layer = m.encoder.layers[a.layer]
    cap = {}
    h = layer.register_forward_pre_hook(lambda mod, args, kw: cap.update(args=args, kw=kw), with_kwargs=True)
    batch = next(iter(m._train_dl))
    with torch.no_grad(), amp():
        m.forward(input_signal=batch[0].cuda(), input_signal_length=batch[1].cuda())
    h.remove()
    x0 = (cap["args"][0] if cap["args"] else cap["kw"]["x"]).detach()
    kw = {k: v for k, v in cap["kw"].items() if k != "x"}
    print(f"layer {a.layer} input {tuple(x0.shape)} {x0.dtype}  (B, T frames, d_model)  "
          f"audio {float(batch[1].sum()) / 16000:.0f} s", flush=True)

    def run(fn):
        x = x0.clone().requires_grad_()
        with amp():
            y = fn(x)
        y.float().sum().backward()

    L = layer
    res = lambda f: (lambda x: x + L.dropout(f(x)))
    pieces = {
        "whole layer": lambda x: L(x, **kw),
        "ff1 (LN + MLP + res)": lambda x: x + L.dropout(L.feed_forward1(L.norm_feed_forward1(x))) * L.fc_factor,
        "self-attn (LN + MHA + res)": lambda x: x + L.dropout(L.self_attn(
            query=(n := L.norm_self_att(x)), key=n, value=n, mask=kw.get("att_mask"), pos_emb=kw.get("pos_emb"))),
        "conv module (LN + conv + res)": res(lambda x: L.conv(L.norm_conv(x), pad_mask=kw.get("pad_mask"))),
        "ff2 (LN + MLP + res)": lambda x: x + L.dropout(L.feed_forward2(L.norm_feed_forward2(x))) * L.fc_factor,
        "glue: res + dropout + LN (x4)": lambda x: L.norm_out(x + L.dropout(L.norm_conv(x + L.dropout(
            L.norm_self_att(x + L.dropout(L.norm_feed_forward1(x)) * 0.5))))),
    }
    for name, fn in pieces.items():
        for _ in range(3):
            run(fn)
        ms = triton.testing.do_bench(lambda: run(fn), warmup=5, rep=50)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                run(fn)
            torch.cuda.synchronize()
        ks = sorted(((e.self_device_time_total / 5 / 1000, e.key) for e in prof.key_averages()
                     if e.self_device_time_total > 0), reverse=True)
        gpu = sum(k[0] for k in ks)
        print(f"\n== {name}: {ms:.2f} ms fwd+bwd (x17 layers = {17 * ms:.0f} ms/step), GPU kernels {gpu:.2f} ms", flush=True)
        for t, k in ks[:8]:
            print(f"    {t:6.3f} ms {100 * t / gpu:5.1f}%  {k[:95]}")


if __name__ == "__main__":
    main()
