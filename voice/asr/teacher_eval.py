"""Offline teacher candidates on the eval clips: which models should pseudo-label our training audio?

    python voice/asr/teacher_eval.py --eval /home/marimo/work/asr/eval --out /home/marimo/work/asr/teachers

Teachers run OFFLINE (no streaming constraint, they label data once on a GPU). Every clip is cut into fixed 25 s chunks
so all models see identical input (long-form handling differs per model and would confound the ranking); chunk
transcripts are joined in order. One model failing does not stop the others. Outputs <out>/<model>_<clip>.txt; score
with voice/asr/score.py.
"""
import argparse
import functools
import os
import traceback

import soundfile as sf

CHUNK_S = 25.0
CLIPS = {"clip5m": ("clip16k.wav", "en"), "clip2m": ("c2m.wav", "en"), "hindi": ("hindi16k.wav", "hi")}


def chunks(wav, out_dir, tag):
    x, sr = sf.read(wav)
    n = int(CHUNK_S * sr)
    paths = []
    starts = list(range(0, len(x), n))
    if len(starts) > 1 and len(x) - starts[-1] < sr:   # a sub-second tail joins the previous chunk (ARK crashed on 29 samples)
        starts.pop()
    for k, i in enumerate(starts):
        p = os.path.join(out_dir, "chunks", f"{tag}_{k:03d}.wav")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sf.write(p, x[i:starts[k + 1]] if k + 1 < len(starts) else x[i:], sr)
        paths.append(p)
    return paths


def text_of(h):
    return h.text if hasattr(h, "text") else (h[0] if isinstance(h, (list, tuple)) else str(h))


@functools.lru_cache(None)                      # each model is loaded once, then reused for every clip
def load(name):
    if name == "parakeet_v3":
        import nemo.collections.asr as nemo_asr
        return nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").eval()
    if name == "canary1b_v2":
        from nemo.collections.asr.models import EncDecMultiTaskModel
        return EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2").eval()
    if name == "canary_qwen":
        from nemo.collections.speechlm2.models import SALM
        return SALM.from_pretrained("nvidia/canary-qwen-2.5b").eval().to("cuda")
    if name == "whisper_v3":
        import torch
        from transformers import pipeline
        return pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                        device="cuda")


def parakeet(paths, lang):
    m = load("parakeet_v3")
    return [text_of(h) for h in m.transcribe(paths, batch_size=8)]


def canary1b(paths, lang):
    if lang != "en":
        raise ValueError("canary-1b-v2 has no Hindi")
    m = load("canary1b_v2")
    return [text_of(h) for h in m.transcribe(paths, batch_size=8, source_lang="en", target_lang="en")]


def canary_qwen(paths, lang):
    if lang != "en":
        raise ValueError("canary-qwen-2.5b is English-only")
    m = load("canary_qwen")
    out = []
    for p in paths:
        ids = m.generate(prompts=[[{"role": "user", "content": f"Transcribe the following: {m.audio_locator_tag}",
                                    "audio": [p]}]], max_new_tokens=256)
        out.append(m.tokenizer.ids_to_text(ids[0].cpu()))
    return out


def whisper(paths, lang):
    asr = load("whisper_v3")
    return [asr(p, generate_kwargs={"language": lang, "task": "transcribe"})["text"] for p in paths]


TEACHERS = {"parakeet_v3": parakeet, "canary1b_v2": canary1b, "canary_qwen": canary_qwen, "whisper_v3": whisper}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(TEACHERS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clip_paths = {c: chunks(os.path.join(a.eval, w), a.out, c) for c, (w, _) in CLIPS.items()}
    for name in a.models.split(","):
        for clip, (_, lang) in CLIPS.items():
            try:
                texts = TEACHERS[name](clip_paths[clip], lang)
                open(os.path.join(a.out, f"{name}_{clip}.txt"), "w", encoding="utf-8").write(" ".join(texts))
                print(f"TEACHER {name} {clip} ok", flush=True)
            except Exception as ex:
                print(f"TEACHER {name} {clip} FAILED {type(ex).__name__}: {str(ex)[:160]}", flush=True)
                traceback.print_exc()
        load.cache_clear()                     # free this teacher before loading the next
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
