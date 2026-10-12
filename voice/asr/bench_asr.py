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
NEMO = {"parakeet": "nvidia/parakeet-tdt-0.6b-v2", "nemotron": "nvidia/nemotron-speech-streaming-en-0.6b",
        "stock": "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"}   # our base model, before our training


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


def run_nemo(name, la, paths, sec, bps, head="rnnt", sc_head="none"):
    """{bp: hyps}: one model load, one decode per blank penalty (blank_penalty.py patches the RNN-T / TDT joint)."""
    import nemo.collections.asr as nemo_asr
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import blank_penalty
    if name.endswith(".nemo"):          # one of our trained runs (our self-conditioned CTC head: --ctc_head glu_glu)
        from ab_meeting import load
        m = load(name, sc_head)
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(NEMO[name]).cuda().eval()
    if hasattr(m, "joint"):             # RNN-T/TDT greedy: a captured CUDA graph would keep the first blank penalty
        from omegaconf import open_dict
        dec = m.cfg.decoding
        with open_dict(dec):
            dec.greedy.use_cuda_graph_decoder = False
        m.change_decoding_strategy(dec, decoder_type="rnnt")
    if la is not None:
        ctx = [-1, -1] if la < 0 else [70 - 70 % (la + 1), la]    # the training mask (ctc_heads.set_lookahead): 240 ms = [68, 3]
        try:
            m.encoder.set_default_att_context_size(ctx)
        except Exception as e:           # a full-attention model (Parakeet) has no streaming presets: force the mask
            print(f"[bench] preset refused ({e}); forcing chunked_limited {ctx}", flush=True)
            m.encoder.att_context_style = "chunked_limited"
            m.encoder.att_context_size_all = [ctx]
            m.encoder.att_context_size = ctx
            for layer in m.encoder.layers:
                if hasattr(layer.self_attn, "att_context_size"):
                    layer.self_attn.att_context_size = ctx
        print(f"[bench] {name} att_context {m.encoder.att_context_size} style {m.encoder.att_context_style}", flush=True)
    if head == "ctc":                    # hybrid RNN-T/CTC model: decode with its CTC head
        m.change_decoding_strategy(decoder_type="ctc")
    res = {}
    for bp in bps:
        blank_penalty.apply(bp, head=head)
        hyp = [None] * len(paths)
        for b in batches(paths, sec):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = m.transcribe([paths[i] for i in b], batch_size=len(b), verbose=False)
            out = out[0] if isinstance(out, tuple) else out
            for i, o in zip(b, out):
                hyp[i] = o.text if hasattr(o, "text") else str(o)
        res[bp] = hyp
        print(f"[bench] {name} bp {bp} done", flush=True)
    blank_penalty.apply(0.0, head=head)
    return res


def run_granite_bp(paths, bps):
    """{bp: hyps}: Granite TurboCTC frame log-probs once per utterance (batch 1: no padded frames), then for each
    penalty: subtract it from blank (id 0), greedy argmax, collapse repeats, drop blanks, detokenise."""
    from transformers import AutoModelForCTC, AutoProcessor
    proc = AutoProcessor.from_pretrained(HF["granite"])
    m = AutoModelForCTC.from_pretrained(HF["granite"], device_map="cuda", torch_dtype=torch.bfloat16).eval()
    res = {bp: [None] * len(paths) for bp in bps}
    for i, p in enumerate(paths):
        inp = proc([load(p)], sampling_rate=16000, device=m.device)
        inp.to(m.device, dtype=m.dtype)
        with torch.inference_mode():
            lp = m(**inp).logits.float().log_softmax(-1)[0]
        for bp in bps:
            x = lp.clone(); x[:, 0] -= bp
            ids = x.argmax(-1).tolist()
            keep = [t for j, t in enumerate(ids) if t != 0 and (j == 0 or t != ids[j - 1])]
            res[bp][i] = proc.tokenizer.decode(keep, skip_special_tokens=True)
    return res


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
    ap.add_argument("--model", required=True, help=f"{list(NEMO) + list(HF)} or a trained .nemo")
    ap.add_argument("--la", type=int, default=None, help="nemotron: right context in 80 ms frames (0 1 6 13), -1 full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default=f"{A}/en3/val_*.jsonl")
    ap.add_argument("--batch_sec", type=float, default=400.0)
    ap.add_argument("--meeting_only", action="store_true")
    ap.add_argument("--ctc_head", default="none", help="a trained .nemo with our self-conditioned CTC head: glu_glu")
    ap.add_argument("--head", choices=["rnnt", "ctc"], default="rnnt", help="NeMo hybrid models: which decoder")
    ap.add_argument("--bps", type=float, nargs="*", default=None,
                    help="blank penalties: NeMo RNN-T/TDT (blank_penalty.py) and Granite CTC; one OUT_bp<x> dir each")
    a = ap.parse_args()
    mset = "meeting_full" if a.la is None or a.la < 0 else f"meeting_la{a.la * 80}"
    its = items(mset, "/nonexistent" if a.meeting_only else a.val)
    paths = [p for p, _, _ in its]
    t0 = time.time()
    if a.bps is not None and (a.model in NEMO or a.model.endswith('.nemo')):
        res = {f"{a.out}_bp{bp:g}": h for bp, h in run_nemo(a.model, a.la, paths, a.batch_sec, a.bps, a.head, a.ctc_head).items()}
    elif a.bps is not None and a.model == "granite":
        res = {f"{a.out}_bp{bp:g}": h for bp, h in run_granite_bp(paths, a.bps).items()}
    elif (a.model in NEMO or a.model.endswith('.nemo')):
        res = {a.out: run_nemo(a.model, a.la, paths, a.batch_sec, [0.0], a.head, a.ctc_head)[0.0]}
    else:
        res = {a.out: run_hf(a.model, paths, a.batch_sec)}
    dt = time.time() - t0
    audio_h = sum(sf.info(p).duration for p in paths) / 3600
    for out, hyp in res.items():
        os.makedirs(out, exist_ok=True)
        with open(f"{out}/refs.jsonl", "w", encoding="utf-8") as f:
            for p, s, r in its:
                f.write(json.dumps({"file": p, "set": s, "ref": r}, ensure_ascii=False) + "\n")
        with open(f"{out}/hyp.tsv", "w", encoding="utf-8") as f:
            for p, h in zip(paths, hyp):
                f.write(f"{p}\t{(h or '').replace(chr(9), ' ').replace(chr(10), ' ')}\n")
        print(f"BENCH {os.path.basename(out)} | {len(paths)} files, {audio_h:.2f} h audio | empty {sum(not h for h in hyp)}", flush=True)
    print(f"BENCH_TIME {a.model} la {a.la} bps {a.bps}: {dt / 60:.1f} min for {len(res)} decode(s) "
          f"(RTFx {len(res) * audio_h * 3600 / max(dt, 1e-9):.0f})", flush=True)


if __name__ == "__main__":
    main()
