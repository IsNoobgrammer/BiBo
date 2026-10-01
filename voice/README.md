# voice/

Starting point for BiBo speech models: English + Hindi + Hinglish ASR and TTS, hard budget **<= 500M parameters**,
ideally one backbone that does text/speech -> text/speech.

- `datasets.md` -- every verified ASR / TTS dataset (license, commercial use, access, pros / cons), eval sets, how labs
  select data, a permissive starting recipe and an 8-step data pipeline.
- `landscape.md` -- state of the art Oct 2026: Open ASR Leaderboard, Hindi ASR (Vaani benchmark), TTS <= 1B, unified
  speech-text models, speech codecs.

## Where the field is (what matters for us)
- **ASR is a solved shape at our size.** FastConformer-TDT/CTC at 0.18-0.6B is within ~0.7 WER of 2B LLM-decoder
  models in English and 10-20x faster. Hindi open SOTA is IndicConformer-600M (13.1% Vaani) and SraVaani-1.0
  (430M, 14.0% 8-set avg); closed Saaras v3 ties IndicConformer on Vaani. Everyone collapses on **Hinglish**
  (Shrutam 12.8% -> 28.3% on code-switched tokens). That is the open gap.
- **TTS at 0.3-0.6B reaches en WER ~1.7-2.6.** Open Hindi TTS <= 500M is just IndicF5 (0.4B, MIT). Code-mixed
  Hindi-English TTS exists only at 3B (Veena) or closed (Bulbul). Second gap.
- **Unified <= 500M is possible but costs quality.** UniVoice (0.4B, continuous mel in, flow-matching mel out,
  causal mask for ASR / bidirectional for TTS) is the only credible small design. Fully discrete delay-interleaved
  designs (OpusLM) are unusable for TTS below 1.7B (TTS WER 38.7 @ 135M, 19.8 @ 360M).

## Plan (proposed)
1. **ASR first** (fastest to a strong, measurable result): FastConformer hybrid TDT-CTC, 120-430M. Start from
   SraVaani-1.0 (430M, Indic-native) or distil from IndicConformer-600M. Fine-tune on IndicVoices + Kathbath + Vaani
   (pseudo-labelled) + LibriHeavy / YODAS2-en + Hinglish. Target: Hindi ~13-15% Vaani, English ~6.5-7.5 Open ASR
   avg, and a **real Hinglish win** (our 81k tokenizer already handles mixed script).
2. **TTS second**: flow matching (F5 / IndicF5 style, 0.3-0.4B, mel + Vocos) for the fastest good result; or
   AR LM over 25 Hz semantic tokens + flow decoder (CosyVoice 2 style) if streaming and later unification matter.
   Hinglish TTS data via our text LM + the ASR model (TTS <-> ASR flywheel, arxiv 2605.03073).
3. **Unified as an ablation**, only after 1 and 2 exist as baselines: UniVoice-style, backbone cut from the BiBo text
   MoE (<= 350M), ~80-100M FastConformer encoder (from step 1), ~80M flow head, small vocoder; keep ~50% text-only
   data. Expect ~1.5-2x the WER of the dedicated ASR and ~2x the TTS WER; the gain is one model.

## Open decisions
- Commercial vs research use (decides Emilia / GigaSpeech / Gram Vaani; see datasets.md).
- One Hindi / Hinglish text normalization convention (script, numerals) -- must be fixed before any training.
- Codec choice for TTS / unified: measure Hindi resynthesis WER for Mimi, DualCodec, X-codec2 first.
