"""Speed / accuracy of a model in the RT Captions engine (parakeet.cpp, CPU, cache-aware streaming) per quantization.

For each look-ahead R: re-save the .nemo with [left, R] as its FIRST att_context preset (the converter / engine stream
with preset [0]), convert to F32 GGUF (our converter), quantize with the engine's own `parakeet-cli quantize` (q4_0 /
q5_0 / q8_0 / q4_k / q5_k / q6_k; f16 through the converter), then `parakeet-cli transcribe --stream` the meeting clips:
pooled WER (score.normalize_v2) and real-time factor (wall / audio, model load included). The NeMo cache-aware streaming
WER of the same .nemo is the reference (a GGUF far above it = a conversion bug, not quantization).

    python voice/asr/gguf/bench_gguf.py --nemo X.nemo --name nt-l19 --las 6 --quants f32 q8_0 q4_k --cli build/.../parakeet-cli
"""
import argparse
import os
import re
import subprocess
import sys
import time

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from score import edits, normalize_v2  # noqa: E402

MEET = (("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt"))


def la_nemo(src, r, out):
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    m = nemo_asr.models.ASRModel.restore_from(src, map_location="cpu")
    ctx = [list(c) for c in m.cfg.encoder.att_context_size]
    left = max(c[0] for c in ctx)
    want = [-1, -1] if r < 0 else [left - left % (r + 1), r]
    with open_dict(m.cfg):
        m.cfg.encoder.att_context_size = [want] + [c for c in ctx if c != want]
    m.encoder.att_context_size_all = [want] + [c for c in m.encoder.att_context_size_all if list(c) != want]
    m.encoder.set_default_att_context_size(want)
    m.save_to(out)
    return want


def nemo_stream_wer(path, clips, bps):
    """NeMo cache-aware streaming (the engine's algorithm) on the GPU, per RNN-T blank penalty: {bp: (WER, D, I)}."""
    import torch
    import blank_penalty
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from omegaconf import open_dict
    torch.set_grad_enabled(False)
    m = nemo_asr.models.ASRModel.restore_from(path).cuda().eval()
    dec = m.cfg.decoding
    with open_dict(dec):
        dec.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(dec)
    m.encoder.setup_streaming_params()
    res = {}
    for bp in bps:
      blank_penalty.apply(bp, head="rnnt")
      s = d = i = n = 0
      for wav, ref in clips:
        buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
        buf.append_audio(wav, stream_id=-1)
        cc, ct, cl = m.encoder.get_initial_cache_state(batch_size=1)
        prev = pred = None
        for step, (chunk, clen) in enumerate(iter(buf)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, texts, cc, ct, cl, prev = m.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=clen, cache_last_channel=cc, cache_last_time=ct,
                    cache_last_channel_len=cl, keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=prev,
                    previous_pred_out=pred,
                    drop_extra_pre_encoded=m.encoder.streaming_cfg.drop_extra_pre_encoded if step > 0 else 0,
                    return_transcription=True)
        hyp = texts[0].text if hasattr(texts[0], "text") else texts[0]
        x = edits(ref, normalize_v2(hyp))
        s, d, i, n = s + x[0], d + x[1], i + x[2], n + len(ref)
      res[bp] = (100 * (s + d + i) / n, 100 * d / n, 100 * i / n)
    blank_penalty.apply(0.0, head="rnnt")
    del m
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--las", type=int, nargs="+", default=[6])
    ap.add_argument("--quants", nargs="+", default=["f32", "q8_0", "q6_k", "q5_k", "q4_k"], help="at --main_la")
    ap.add_argument("--main_la", type=int, default=3, help="look-ahead that gets every --quants (the app's 240 ms)")
    ap.add_argument("--quants_other", nargs="+", default=["q8_0", "q4_k"], help="at the other look-aheads")
    ap.add_argument("--bps", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5], help="NeMo reference blank penalties")
    ap.add_argument("--cli", required=True, help="parakeet-cli binary")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--hf_dir", default=None, help="upload the GGUFs to fhai50032/bibo-asr-ckpt:<hf_dir>/")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = [(sf.read(os.path.join(a.meeting, w), dtype="float32")[0],
              normalize_v2(open(os.path.join(a.meeting, r), encoding="utf-8").read())) for w, r in MEET]
    secs = sum(len(w) for w, _ in clips) / 16000
    up = []
    for r in a.las:
        tag = f"{a.name}_la{r}"
        src = os.path.join(a.out, tag + ".nemo")
        ctx = la_nemo(a.nemo, r, src)
        res = nemo_stream_wer(src, clips, a.bps)
        best = min(res, key=lambda b: res[b][0])
        print(f"GG {tag} nemo-stream ctx {ctx} | " + " ".join(f"bp{b:g} {res[b][0]:.2f}" for b in a.bps)
              + f" | best bp {best:g}: {res[best][0]:.2f} (D {res[best][1]:.2f} I {res[best][2]:.2f})", flush=True)
        f32 = os.path.join(a.out, tag + "_f32.gguf")
        subprocess.run([sys.executable, os.path.join(HERE, "convert_parakeet_to_gguf.py"), "--model", src,
                        "--dtype", "f32", "--output", f32], check=True, capture_output=True)
        for q in (a.quants if r == a.main_la else a.quants_other):
            g = os.path.join(a.out, f"{tag}_{q}.gguf")
            if q == "f16":
                subprocess.run([sys.executable, os.path.join(HERE, "convert_parakeet_to_gguf.py"), "--model", src,
                                "--dtype", "f16", "--output", g], check=True, capture_output=True)
            elif q != "f32":
                subprocess.run([a.cli, "quantize", f32, g, q], check=True, capture_output=True)
            else:
                g = f32
            s = dd = ii = n = 0
            wall = 0.0
            for (wav, ref), (wn, _) in zip(clips, MEET):
                t0 = time.time()
                p = subprocess.run([a.cli, "transcribe", "--model", g, "--input", os.path.join(a.meeting, wn),
                                    "--stream", "--threads", str(a.threads)], capture_output=True, text=True)
                wall += time.time() - t0
                fin = re.findall(r"^\[stream:final\] (.*)$", p.stdout, re.M)
                if p.returncode or not fin:
                    print(f"GG {tag} {q} FAILED rc {p.returncode}: {p.stderr[-300:]}", flush=True)
                    break
                x = edits(ref, normalize_v2(fin[-1]))
                s, dd, ii, n = s + x[0], dd + x[1], ii + x[2], n + len(ref)
            else:
                mb = os.path.getsize(g) / 1e6
                print(f"GG {tag} {q:5s} | WER {100 * (s + dd + ii) / n:5.2f} (D {100 * dd / n:5.2f} I {100 * ii / n:4.2f}) | "
                      f"RTF {wall / secs:.3f} ({a.threads} threads, {secs / 60:.1f} min audio) | {mb:.0f} MB", flush=True)
                up.append(g)
    if a.hf_dir:
        from huggingface_hub import HfApi, CommitOperationAdd
        HfApi().create_commit("fhai50032/bibo-asr-ckpt",
                              operations=[CommitOperationAdd(f"{a.hf_dir}/{os.path.basename(g)}", g) for g in up],
                              commit_message=f"{a.name} GGUFs (bench_gguf)")
        print(f"GG uploaded {len(up)} GGUFs to {a.hf_dir}/", flush=True)
    print("GG_END", flush=True)


if __name__ == "__main__":
    main()
