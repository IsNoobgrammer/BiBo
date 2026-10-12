"""Benchmark external ASR models on our meeting clips + en3 validation sets, scored exactly like our own models
(score_ctc_decode.py: meeting sets with normalize_v2, validation sets with normalize; pooled WER per set).

    python voice/asr/bench_asr.py --model parakeet --out /home/marimo/work/asr/bench/parakeet
    python voice/asr/bench_asr.py --model nemotron --la 6 --out .../bench/nemotron_la6     # right context in 80 ms frames
    PYTHONPATH=/home/marimo/hfpkgs python voice/asr/bench_asr.py --model granite --out .../bench/granite   # transformers>=5.16
    PYTHONPATH=/home/marimo/hfpkgs python voice/asr/bench_asr.py --model qwen17 --out .../bench/qwen17

Writes OUT/refs.jsonl ({"file", "set", "ref"}) and OUT/hyp.tsv (file<TAB>text); score with
    python voice/asr/score_ctc_decode.py OUT/refs.jsonl OUT/hyp.tsv --tag NAME
Meeting clips are transcribed whole (one call per 5-min recording); validation utterances in length-sorted batches.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

A = "/home/marimo/work/asr"
MEET = (("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt"))
HF = {"granite": "ibm-granite/granite-speech-5.0-470m-turboctc", "qwen17": "Qwen/Qwen3-ASR-1.7B-hf",
      "qwen06": "Qwen/Qwen3-ASR-0.6B-hf"}
NEMO = {"parakeet": "nvidia/parakeet-tdt-0.6b-v2", "nemotron": "nvidia/nemotron-speech-streaming-en-0.6b"}


def items(meeting_set, val_glob):
    """[(path, set, ref)]: the 2 meeting recordings, then every validation utterance."""
    out = [(f"{A}/eval_meeting/{w}", meeting_set, open(f"{A}/eval_meeting/{r}", encoding="utf-8").read()) for w, r in MEET]
    for f in sorted(glob.glob(val_glob)):
        s = os.path.basename(f)[4:-6]
        out += [(r["audio_filepath"], s, r["text"]) for r in map(json.loads, open(f, encoding="utf-8")) if r.get("text")]
    return out


def load(p):
    x, sr = sf.read(p, dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    assert sr == 16000, (p, sr)
    return x


def batches(paths, sec):
    """Length-sorted batches of at most `sec` seconds of audio (a 5-min recording goes alone)."""
    order = sorted(range(len(paths)), key=lambda i: sf.info(paths[i]).duration)
    cur, tot = [], 0.0
    for i in order:
        d = sf.info(paths[i]).duration
        if cur and tot + d > sec:
            yield cur; cur, tot = [], 0.0
        cur.append(i); tot += d
    if cur:
        yield cur


def run_nemo(name, la, paths, sec):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(NEMO[name]).cuda().eval()
    if la is not None:
        ctx = [-1, -1] if la < 0 else [70, la]
        m.encoder.set_default_att_context_size(ctx)
        print(f"[bench] {name} att_context {ctx}", flush=True)
    hyp = [None] * len(paths)
    for b in batches(paths, sec):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.transcribe([paths[i] for i in b], batch_size=len(b), verbose=False)
        out = out[0] if isinstance(out, tuple) else out
        for i, o in zip(b, out):
            hyp[i] = o.text if hasattr(o, "text") else str(o)
    return hyp


def run_hf(name, paths, sec):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(HF[name])
    hyp = [None] * len(paths)
    if name == "granite":
        from transformers import AutoModelForCTC
        m = AutoModelForCTC.from_pretrained(HF[name], device_map="cuda", torch_dtype=torch.bfloat16).eval()
        for b in batches(paths, sec):
            inp = proc([load(paths[i]) for i in b], sampling_rate=16000, device=m.device)
            inp.to(m.device, dtype=m.dtype)
            with torch.inference_mode():
                out = m.generate(**inp)
            for i, t in zip(b, proc.batch_decode(out, skip_special_tokens=True)):
                hyp[i] = t
    else:
        from transformers import AutoModelForMultimodalLM
        m = AutoModelForMultimodalLM.from_pretrained(HF[name], device_map="cuda", torch_dtype=torch.bfloat16).eval()
        for b in batches(paths, sec):
            long = any(sf.info(paths[i]).duration > 60 for i in b)
            inp = proc.apply_transcription_request(audio=[paths[i] for i in b], language="English").to(m.device, m.dtype)
            with torch.inference_mode():
                ids = m.generate(**inp, max_new_tokens=4096 if long else 512)
            gen = ids[:, inp["input_ids"].shape[1]:]
            for i, t in zip(b, proc.decode(gen, return_format="transcription_only")):
                hyp[i] = t
    return hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(NEMO) + list(HF))
    ap.add_argument("--la", type=int, default=None, help="nemotron: right context in 80 ms frames (0 1 6 13), -1 full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default=f"{A}/en3/val_*.jsonl")
    ap.add_argument("--batch_sec", type=float, default=400.0)
    ap.add_argument("--meeting_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mset = "meeting_full" if a.la is None or a.la < 0 else f"meeting_la{a.la * 80}"
    its = items(mset, "/nonexistent" if a.meeting_only else a.val)
    paths = [p for p, _, _ in its]
    t0 = time.time()
    hyp = run_nemo(a.model, a.la, paths, a.batch_sec) if a.model in NEMO else run_hf(a.model, paths, a.batch_sec)
    dt = time.time() - t0
    audio_h = sum(sf.info(p).duration for p in paths) / 3600
    with open(f"{a.out}/refs.jsonl", "w", encoding="utf-8") as f:
        for p, s, r in its:
            f.write(json.dumps({"file": p, "set": s, "ref": r}, ensure_ascii=False) + "\n")
    with open(f"{a.out}/hyp.tsv", "w", encoding="utf-8") as f:
        for p, h in zip(paths, hyp):
            f.write(f"{p}\t{(h or '').replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    print(f"BENCH {a.model} la {a.la} | {len(paths)} files, {audio_h:.2f} h audio in {dt / 60:.1f} min "
          f"(RTFx {audio_h * 3600 / max(dt, 1e-9):.0f}) | empty {sum(not h for h in hyp)}", flush=True)


if __name__ == "__main__":
    main()
