"""Where does ONE real training step go? run1 model, Lhotse buckets, hybrid loss (0.7 RNN-T + 0.3 CTC), AdamW.
Measures only.

    python voice/asr/prof_step.py --train R/train.jsonl [--fused] [--bf16_master] [--batch_sec 2400]
                                  [--phases] [--sync_debug] [--steps 14]

Default: true throughput -- steps run back to back with ONE sync at the end (CPU and GPU overlap as in training),
then the top CUDA kernels (self time) of 4 profiled steps grouped by part. --phases adds a second pass with a sync
after every phase (data / encoder / decoder+joint+rnnt / ctc / backward / optimizer): per-phase cost, but the syncs
stop the CPU running ahead, so its total is NOT the training speed. --sync_debug lists every host sync in one step
(file:line of the first frame outside torch) -- each one drains the GPU queue.
"""
import argparse
import collections
import contextlib
import os
import sys
import time
import traceback
import warnings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GROUPS = [("joint+rnnt (tkf)", ("_logits_kernel", "_rowgrad", "_fix_e", "_dfg", "_hidden", "_lattice", "_combine_rows")),
          ("rel-pos attn (tkf)", ("_rpa_fwd", "_bwd_q", "_bwd_kv", "_dp_diag")),
          ("conv module (tkf)", ("_cm_fwd", "_cm_bwd")),
          ("silu+dropout (tkf)", ("_sd_fwd", "_sd_bwd")),
          ("res+drop+LN (tkf)", ("_fwd", "_bwd")),
          ("rnnt numba", ("rnnt_loss", "numba")),
          ("attention", ("flash", "fmha", "attention", "softmax")),
          ("gemm", ("gemm", "cutlass", "cublas", "xmma", "nvjet", "Kernel2")),
          ("conv", ("conv", "cudnn", "implicit", "winograd")),
          ("lstm", ("LSTM", "lstm", "RNN")),
          ("ctc", ("ctc",)),
          ("norm", ("layer_norm", "LayerNorm", "batch_norm", "BatchNorm", "GammaBeta")),
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
    ap.add_argument("--nemo", default="stt_en_fastconformer_hybrid_large_streaming_multi", help=".nemo or pretrained name")
    ap.add_argument("--tok", default=None, help="switch to this tokenizer dir (e.g. en3's 2047 BPE)")
    ap.add_argument("--ctc_head", default=None, help="self-conditioned head (glu_glu, lin_glu): fused selfcond CTC loss")
    ap.add_argument("--fused_ctc", action="store_true")
    ap.add_argument("--ctc_only", action="store_true", help="no RNN-T decoder / joint / loss")
    ap.add_argument("--ctx", default="68,3", help="fixed look-ahead for every step: LEFT,RIGHT or full")
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--batch_sec", type=float, default=1200)
    ap.add_argument("--buckets", type=int, default=30, help="Lhotse length buckets (more = less padding)")
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--bf16_master", action="store_true")
    ap.add_argument("--fused_layer", action="store_true")
    ap.add_argument("--fused_attn", action="store_true")
    ap.add_argument("--fused_conv", action="store_true")
    ap.add_argument("--ops", action="store_true", help="also list aten ops by GPU time, with input shapes")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--sync_debug", action="store_true")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    torch.set_float32_matmul_precision("high")
    m = (nemo_asr.models.ASRModel.restore_from(a.nemo) if a.nemo.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(a.nemo)).cuda().train()
    if a.tok:
        m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
        m = m.cuda().train()
    ctx = [-1, -1] if a.ctx == "full" else [int(x) for x in a.ctx.split(",")]
    m.encoder.att_context_size_all = [ctx]
    m.encoder.att_context_probs = [1.0]
    m.encoder.set_default_att_context_size(ctx)
    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=a.buckets,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=12, shuffle_buffer_size=10000, seed=23)
    m.setup_training_data(tr)
    if a.fused:
        import fused_joint
        fused_joint.enable(m)
    if a.fused_layer:
        import fused_layer
        fused_layer.enable(m)
    if a.fused_attn:
        import fused_attn
        fused_attn.enable(m)
    if a.fused_conv:
        import fused_conv
        fused_conv.enable(m)
    if a.fused_ctc:
        import fused_ctc
        fused_ctc.enable(m)
    dec = None
    if a.ctc_head:
        import selfcond_head
        dec = selfcond_head.enable(m, a.ctc_head)
    if a.ctc_only:
        for mod in (m.decoder, m.joint):
            mod.requires_grad_(False)
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=1e-5, betas=(0.9, 0.98), weight_decay=1e-3, fused=True)
    if a.bf16_master:
        import bf16_master
        bf16_master.to_bf16(m)
        bf16_master.wrap(opt, clip=1.0)
        amp = contextlib.nullcontext
    else:
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    w = m.ctc_loss_weight
    it = iter(m._train_dl)
    sync = torch.cuda.synchronize
    tag = (f"{'ctc-only' if a.ctc_only else 'hybrid'} head {a.ctc_head or 'stock'} ctx {a.ctx} | ") + ("fused" if a.fused else "nemo") + (" +layer" if a.fused_layer else "") + (" +attn" if a.fused_attn else "") + (" +conv" if a.fused_conv else "") + (" bf16-master" if a.bf16_master else " autocast") + f" {a.batch_sec:.0f}s"

    def step(batch, mark=None):
        mark = mark or (lambda i: None)
        sig, sig_len, y, y_len = (x.cuda(non_blocking=True) for x in batch[:4])
        mark(0)
        with amp():
            enc, enc_len = m.forward(input_signal=sig, input_signal_length=sig_len)
            mark(1)
            if not a.ctc_only:
                pred, _, _ = m.decoder(targets=y, target_length=y_len)
                rnnt, _, _, _ = m.joint(encoder_outputs=enc, decoder_outputs=pred, encoder_lengths=enc_len,
                                        transcripts=y, transcript_lengths=y_len)
            mark(2)
            if dec is not None:                    # the self-conditioned head's fused loss, as in training
                dec.stash = [y, y_len, enc_len, lambda nll, tl: m.ctc_loss.reduce(nll, tl)]
            ctc = m.ctc_loss(log_probs=m.ctc_decoder(encoder_output=enc), targets=y, input_lengths=enc_len,
                             target_lengths=y_len)
            if dec is not None:
                dec.stash, dec.loss = None, None
            loss = ctc if a.ctc_only else (1 - w) * rnnt + w * ctc
        mark(3)
        loss.backward()
        mark(4)
        if not a.bf16_master:
            torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.0, foreach=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        mark(5)
        return sig_len

    batches = [next(it) for _ in range(a.steps)]                   # data off the clock (loader measured separately)
    audio = [float(b[1].sum()) / 16000 for b in batches]
    pad = [float(b[0].shape[0] * b[0].shape[1]) / 16000 for b in batches]
    print(f"  batches: {sum(audio) / len(audio):.0f} s real audio, {sum(pad) / len(pad):.0f} s incl. padding "
          f"({100 * (1 - sum(audio) / sum(pad)):.1f}% padding), {sum(b[0].shape[0] for b in batches) / len(batches):.0f} utts/batch", flush=True)
    for b in batches:                     # warm-up on EVERY batch: Triton compiles a variant per context / shape class
        step(b)
    sync()
    t0 = time.perf_counter()
    for b in batches[3:]:
        step(b)
    sync()
    wall = time.perf_counter() - t0
    n = a.steps - 3
    print(f"== {tag}: {1000 * wall / n:.0f} ms/step, {sum(audio[3:]) / wall:.0f} audio-s/s "
          f"(steps back to back, one sync)", flush=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for b in batches[3:7]:
            step(b)
        sync()
    kern = {}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            kern[e.key] = kern.get(e.key, 0) + e.self_device_time_total / 4 / 1000
    gt = sum(kern.values())
    print(f"  GPU kernel time {gt:.0f} ms/step = {100 * gt / (1000 * wall / n):.0f}% of the step "
          f"(the rest: GPU idle, waiting on the CPU)")
    gk = collections.Counter()
    for k, v in kern.items():
        gk[group(k)] += v
    for g, v in gk.most_common():
        print(f"    {g:20s} {v:7.1f} ms  {100 * v / gt:5.1f}%")
    print("  top kernels:")
    for k, v in sorted(kern.items(), key=lambda x: -x[1])[:15]:
        print(f"    {v:7.2f} ms  {100 * v / gt:5.1f}%  [{group(k)}]  {k[:80]}")

    if a.ops:                             # which aten op launched the elementwise / copy kernels (CPU + CUDA trace)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            for b in batches[3:5]:
                step(b)
            sync()
        print("  aten ops by GPU time (incl. children), ms/step [input shapes]:")
        rows = [e for e in prof.key_averages(group_by_input_shape=True)
                if e.key.startswith("aten::") and e.device_time_total > 0]
        for e in sorted(rows, key=lambda e: -e.device_time_total)[:40]:
            print(f"    {e.device_time_total / 2 / 1000:7.2f}  x{e.count // 2:<4d} {e.key:28s} {str(e.input_shapes)[:90]}")

    if a.phases:
        names = ["h2d", "preprocessor+encoder fwd", "decoder+joint+rnnt fwd", "ctc head fwd", "backward", "clip+optimizer"]
        acc = [0.0] * 6
        for b in batches[3:]:
            ts = []
            sync()
            t = time.perf_counter()

            def mark(i):
                sync()
                ts.append(time.perf_counter())
            step(b, mark)
            for i in range(6):
                acc[i] += ts[i] - (ts[i - 1] if i else t)
        tot = sum(acc)
        print(f"  phases (a sync after each, so CPU cannot run ahead: {1000 * tot / n:.0f} ms/step):")
        for nm, v in zip(names, acc):
            print(f"    {nm:24s} {1000 * v / n:7.1f} ms  {100 * v / tot:5.1f}%")

    if a.sync_debug:
        sites = collections.Counter()

        def show(msg, cat, fn, ln, file=None, line=None):
            for fr in reversed(traceback.extract_stack()[:-1]):
                if "/torch/" not in fr.filename and "warnings" not in fr.filename:
                    sites[f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"] += 1
                    return
        old = warnings.showwarning
        warnings.showwarning = show
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        step(batches[-1])
        torch.cuda.set_sync_debug_mode(0)
        warnings.showwarning = old
        print(f"  host syncs in one step: {sum(sites.values())}")
        for s_, c in sites.most_common(25):
            print(f"    {c:4d}x  {s_}")


if __name__ == "__main__":
    main()
