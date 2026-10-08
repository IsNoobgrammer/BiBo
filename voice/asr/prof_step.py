"""Where does ONE real training step go now? run1 model, Lhotse 1200 s buckets, bf16 autocast, AdamW, hybrid loss
(0.7 RNN-T + 0.3 CTC), NeMo's joint vs --fused_joint. Measures only.

    python voice/asr/prof_step.py --train R/train.jsonl [--steps 12] [--fused]

Prints CUDA-synchronised wall time per phase (data / encoder fwd / decoder+joint+loss fwd / CTC fwd / backward /
optimizer), audio-s per wall-s, and the top CUDA kernels (self time) over the profiled steps, grouped by part.
"""
import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GROUPS = [("joint+rnnt (tkf)", ("_logits_kernel", "_rowgrad", "_fix_e", "_dfg", "_hidden", "_lattice", "_combine_rows")),
          ("rnnt numba", ("compute_", "numba", "denominator", "compute_alphas", "compute_betas", "compute_grad")),
          ("attention", ("flash", "fmha", "attention", "softmax")),
          ("gemm", ("gemm", "cutlass", "cublas", "sm80_xmma", "sm90", "nvjet", "Kernel2")),
          ("conv", ("conv", "cudnn", "implicit", "winograd")),
          ("lstm", ("LSTM", "lstm", "RNN")),
          ("ctc", ("ctc",)),
          ("norm", ("layer_norm", "LayerNorm", "batch_norm", "BatchNorm")),
          ("optimizer", ("multi_tensor", "adam", "Adam", "foreach")),
          ("copy/cast", ("copy", "Copy", "cast")),
          ("elementwise", ("elementwise", "vectorized", "unrolled", "reduce"))]


def group(name):
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--nemo", default="/home/marimo/work/asr/exp/run1/run1.nemo")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--batch_sec", type=float, default=1200)
    ap.add_argument("--fused", action="store_true")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    torch.set_float32_matmul_precision("high")
    m = nemo_asr.models.ASRModel.restore_from(a.nemo).cuda().train()
    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=30,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=12, shuffle_buffer_size=10000, seed=23)
    m.setup_training_data(tr)
    if a.fused:
        import fused_joint
        fused_joint.enable(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5, betas=(0.9, 0.98), weight_decay=1e-3, fused=True)
    w = m.ctc_loss_weight
    it = iter(m._train_dl)
    sync = torch.cuda.synchronize
    rows, kern = [], {}
    prof = None
    for step in range(a.steps):
        sync()
        t0 = time.perf_counter()
        sig, sig_len, y, y_len = (x.cuda(non_blocking=True) for x in next(it)[:4])
        sync()
        t1 = time.perf_counter()
        if step == a.steps - 4:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, enc_len = m.forward(input_signal=sig, input_signal_length=sig_len)
            sync()
            t2 = time.perf_counter()
            dec, _, _ = m.decoder(targets=y, target_length=y_len)
            rnnt, _, _, _ = m.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=enc_len,
                                    transcripts=y, transcript_lengths=y_len)
            sync()
            t3 = time.perf_counter()
            ctc = m.ctc_loss(log_probs=m.ctc_decoder(encoder_output=enc), targets=y, input_lengths=enc_len,
                             target_lengths=y_len)
            loss = (1 - w) * rnnt + w * ctc
        sync()
        t4 = time.perf_counter()
        loss.backward()
        sync()
        t5 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        t6 = time.perf_counter()
        audio = float(sig_len.sum()) / 16000
        rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, audio))
        print(f"step {step:2d}  B={sig.shape[0]:3d}  {audio:6.0f} s audio  total {1000 * (t6 - t0):6.0f} ms  "
              f"loss {loss.item():.3f}", flush=True)
    prof.__exit__(None, None, None)
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            kern[e.key] = kern.get(e.key, 0) + e.self_device_time_total / 4 / 1000
    r = rows[2:]                                                    # skip warm-up (Triton / cuDNN autotune)
    mean = lambda i: 1000 * sum(x[i] for x in r) / len(r)
    names = ["data wait", "encoder fwd", "decoder+joint+rnnt fwd", "ctc fwd", "backward", "clip+optimizer"]
    tot = sum(mean(i) for i in range(6))
    print(f"\n== {'fused' if a.fused else 'nemo'} joint: {tot:.0f} ms/step, "
          f"{sum(x[6] for x in r) / sum(sum(x[:6]) for x in r):.0f} audio-s/s")
    for i, n in enumerate(names):
        print(f"  {n:24s} {mean(i):7.1f} ms  {100 * mean(i) / tot:5.1f}%")
    gk = {}
    for k, v in kern.items():
        gk[group(k)] = gk.get(group(k), 0) + v
    gt = sum(gk.values())
    print(f"\n  CUDA kernel time {gt:.0f} ms/step by part:")
    for g, v in sorted(gk.items(), key=lambda x: -x[1]):
        print(f"    {g:20s} {v:7.1f} ms  {100 * v / gt:5.1f}%")
    print("\n  top kernels:")
    for k, v in sorted(kern.items(), key=lambda x: -x[1])[:20]:
        print(f"    {v:7.2f} ms  {100 * v / gt:5.1f}%  [{group(k)}]  {k[:80]}")


if __name__ == "__main__":
    main()
