# voice/

Starting point for BiBo speech models: English + Hindi + Hinglish ASR and TTS, hard budget **<= 500M parameters**,
ideally one backbone that does text/speech -> text/speech.

- `datasets.md` -- every verified ASR / TTS dataset (license, commercial use, access, pros / cons), eval sets, how labs
  select data, a permissive starting recipe and an 8-step data pipeline.
- `realtime-asr-lessons.md` -- measured constraints from building RT Captions (live captions on a laptop CPU):
  streaming vs rolling window, latency/CPU/size budgets, QAT, VAD, runtime lessons, requirements for our model.
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

## Chosen design: real-time trilingual ASR with built-in speaker turns (Oct 8 2026)
For the RT Captions app (constraints: `realtime-asr-lessons.md`). Decisions: English + Hindi + Hinglish from the
start; research / internal use for now; the internal meeting clip may be used for EVAL ONLY (never training, never
pushed to HF or git).

- **One streaming model, ~150-200M.** Cache-aware FastConformer encoder (8x subsampling, 80 ms frames), multi-lookahead
  training (0 / 80 / 480 / 1040 ms) so one model gives fast previews and a better commit pass; TDT/RNN-T head with a
  small prediction net + CTC head. Start from the 114M `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`
  encoder (CC-BY-4.0); grow toward 200M only if accuracy plateaus (CPU budget favours smaller).
- **Tokenizer: ONE joint SentencePiece-BPE, ~4k, NO language tags.** Trained on a BALANCED 50/50 Hindi + English text
  sample, `character_coverage=1.0`, `byte_fallback`; cased + punctuated. Hindi words in Devanagari, English words in
  Latin, so Hinglish = natural code-switched output ("कल की meeting cancel हो गई"). Language tags were dropped: hard
  `<hi>`/`<en>` masks break code-switching, and with romanized Hinglish removed there is no script ambiguity left.
  Specials: `<spk1>`..`<spk4>`, `<eou>` only. Target convention: English loanwords ALWAYS in Latin (pipeline
  normalises Devanagari-written English loanwords). Drift guard, only if needed: an optional DECODE-TIME script mask
  in the app ("English only" / "Hindi only"), never in training; default auto mode keeps code-switching free.
  Vocab size confirmed by tokens / s of speech per language at 2k / 4k / 8k on the eval data.
- **Diarization as tokens, not a separate model (SOT / speaker-turn tokens).** Speakers are numbered by FIRST
  APPEARANCE in the sample (first voice = `<spk1>`), so the model learns "same voice or new voice", never identity.
  Identity across a long meeting (beyond the ~10-30 s cache) comes from a runtime speaker-profile table: the encoder's
  pooled vector at each speaker token is matched by cosine to running profiles -- no extra network. Trade-off vs a
  frame-level head: overlapping speech is serialized; speaker changes land on word boundaries (fine for captions).
- **Training:** stage 1 plain trilingual ASR (pseudo-labels from Nemotron-3.5-ASR 0.6B, IndicConformer-600M,
  Parakeet as teachers); stage 2 fine-tune with speaker tokens on AMI / Fisher / Switchboard, Sortformer- or
  pyannote-labelled meeting audio, and SIMULATED conversations (concatenated / lightly overlapped single-speaker
  Hindi and Indian-English utterances -- exact labels, the only scalable source of Hindi / Hinglish turn data).
- **Then** 4-bit QAT (first/last layers + joint at 8-bit) and the CPU runtime with the C API from the lessons doc.
- **First step:** eval set (meeting clips + Kathbath / IndicVoices test + MUCS-2021 Hinglish) and baselines for the
  114M English model and Nemotron 0.6B, scored with RT Captions `bench` (WER, word latency p50/p95, engine busy %).
