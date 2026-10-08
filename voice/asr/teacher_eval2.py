"""More teacher candidates (HF ASR models up to ~8B), same protocol as teacher_eval.py: 25 s chunks, joined transcript.

    python voice/asr/teacher_eval2.py --eval EVAL --out OUT --models qwen3asr,voxtral_rt,granite8b,ark3b,whisper_turbo
    python voice/asr/teacher_eval2.py --eval EVAL --out OUT --models phi4mm          (transformers 4.48 venv)

Each model has its own dependency set, so run them from the venv that matches (see k103.sh). English-only models skip
the Hindi clip. VibeVoice-ASR (long-form, built-in diarization) runs separately through its repo's demo script.
"""
import argparse
import functools
import gc
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_eval import CLIPS, chunks  # noqa: E402

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

HI_OK = {"qwen3asr", "voxtral_rt", "whisper_turbo"}
QWEN_LANG = {"en": "English", "hi": "Hindi"}


@functools.lru_cache(None)
def load(name):
    if name == "qwen3asr":
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        mid = "Qwen/Qwen3-ASR-1.7B-hf"
        return AutoProcessor.from_pretrained(mid), AutoModelForMultimodalLM.from_pretrained(mid, device_map="cuda")
    if name == "voxtral_rt":
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
        mid = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        return AutoProcessor.from_pretrained(mid), VoxtralRealtimeForConditionalGeneration.from_pretrained(mid, device_map="cuda")
    if name == "granite8b":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        mid = "ibm-granite/granite-speech-3.3-8b"
        return AutoProcessor.from_pretrained(mid), AutoModelForSpeechSeq2Seq.from_pretrained(mid, device_map="cuda", torch_dtype=torch.bfloat16)
    if name == "ark3b":
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        mid = "Edge0/ARK-ASR-3B"
        return (AutoProcessor.from_pretrained(mid, trust_remote_code=True), AutoTokenizer.from_pretrained(mid, trust_remote_code=True),
                AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval())
    if name == "whisper_turbo":
        from transformers import pipeline
        return pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda")
    if name == "phi4mm":
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        mid = "microsoft/Phi-4-multimodal-instruct"
        return (AutoProcessor.from_pretrained(mid, trust_remote_code=True),
                AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, torch_dtype="auto", _attn_implementation="sdpa").cuda().eval(),
                GenerationConfig.from_pretrained(mid))
    raise KeyError(name)


@torch.inference_mode()
def transcribe(name, path, lang):
    if name == "qwen3asr":
        proc, m = load(name)
        inputs = proc.apply_transcription_request(audio=path, language=QWEN_LANG[lang]).to(m.device, m.dtype)
        out = m.generate(**inputs, max_new_tokens=512)
        return proc.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0]
    if name == "voxtral_rt":
        proc, m = load(name)
        x, sr = sf.read(path)
        inputs = proc(x, return_tensors="pt").to(m.device, dtype=m.dtype)
        return proc.batch_decode(m.generate(**inputs), skip_special_tokens=True)[0]
    if name == "granite8b":
        proc, m = load(name)
        x, sr = sf.read(path, dtype="float32")
        sysmsg = ("Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\n"
                  "You are Granite, developed by IBM. You are a helpful AI assistant")
        chat = [{"role": "system", "content": sysmsg},
                {"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}]
        prompt = proc.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = proc(prompt, torch.tensor(x)[None], device="cuda", return_tensors="pt").to("cuda")
        out = m.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
        return proc.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    if name == "ark3b":
        proc, tok, m = load(name)
        conv = [{"role": "user", "content": [{"type": "audio", "path": path}, {"type": "text", "text": "Please transcribe this audio."}]}]
        inputs = proc.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt", sampling_rate=16000,
                                          audio_padding="longest", text_kwargs={"padding": "longest"},
                                          audio_max_length=30 * 16000).to("cuda")
        if "audios" in inputs:
            inputs["audios"] = inputs["audios"].to(dtype=torch.bfloat16)
        keep = {tok.eos_token_id} if isinstance(tok.eos_token_id, int) else set(tok.eos_token_id or [])
        bad = (set(tok.all_special_ids) - keep) | {i for t, i in tok.get_added_vocab().items()
                                                   if t.startswith("<") and t.endswith(">") and i not in keep}
        out = m.generate(**inputs, do_sample=False, max_new_tokens=512, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, bad_words_ids=[[i] for i in sorted(bad)])
        return tok.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    if name == "whisper_turbo":
        return load(name)(path, generate_kwargs={"language": lang, "task": "transcribe"})["text"]
    if name == "phi4mm":
        proc, m, gc_ = load(name)
        x, sr = sf.read(path)
        prompt = "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>"
        inputs = proc(text=prompt, audios=[(x, sr)], return_tensors="pt").to("cuda")
        out = m.generate(**inputs, max_new_tokens=512, generation_config=gc_)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clip_paths = {c: chunks(os.path.join(a.eval, w), a.out, c) for c, (w, _) in CLIPS.items()}
    for name in a.models.split(","):
        for clip, (_, lang) in CLIPS.items():
            if lang == "hi" and name not in HI_OK:
                continue
            try:
                texts = [transcribe(name, p, lang) for p in clip_paths[clip]]
                open(os.path.join(a.out, f"{name}_{clip}.txt"), "w", encoding="utf-8").write(" ".join(texts))
                print(f"TEACHER {name} {clip} ok", flush=True)
            except Exception as ex:
                print(f"TEACHER {name} {clip} FAILED {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                traceback.print_exc()
        load.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
