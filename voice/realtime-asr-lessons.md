# Real-time ASR: lessons from building RT Captions (Oct 2026)

What we measured while building an on-device live-caption app (Tauri + Rust, `S:\projects\RT-Captions`), written
as requirements for a BiBo real-time ASR model. Every number below was measured on **one laptop: Intel
i5-12450H (4 P-cores + 4 E-cores, 12 threads, AVX2, no AVX-512), Windows 11, CPU only**, on one 5-minute
stretch of a real internal meeting (accented Indian English, medical + product jargon, 3 speakers, 764 words).
Reference = Gemini 3.8 Flash transcript; one shared normalizer (lowercase, no punctuation, fillers dropped).

## 1. What we tried (live = simulated real-time with the app's own pipeline)

| Engine | Params / file | How it runs live | WER | Word on screen after (p50 / p95) | Engine busy |
|---|---|---|---|---|---|
| Fathom (cloud, after the meeting) | ? | not live | 12.8% | - | - |
| Parakeet Ultra (TDT 0.6B v3, q8) | 0.6B / 942 MB | offline only (0.6-1.1 s per 3-5 s clip) | 11.3% offline | too slow live | - |
| **Nemotron 3.5 ASR streaming q8** | 0.6B / 984 MB | cache-aware streaming, 160 ms feed | 14.6% | 0.58 / 1.37 s (4 threads) | 77% |
| **Nemotron 3.5 ASR streaming q4_k** | 0.6B / 718 MB | cache-aware streaming | 16.7% | 0.58 / 1.44 s (4 threads) | 72% |
| Parakeet TDT 0.6B v2 q4_k (English-only) | 0.6B / 638 MB | rolling 3 s window, 0.5 s hop | 16.0% | 1.87 / 3.89 s | 90% (falls behind) |
| Parakeet Redux (ternary TDT 0.6B v3) | 0.6B / 213 MB | rolling 3 s window, 0.5 s hop | 19.9% | ~1.5 / 3.6 s | 75% |
| Parakeet realtime EOU | 120M / 176 MB | cache-aware streaming | 20.8% | 0.37 / 1.32 s | 78% (!) |
| Cactus Whistle | ~17 MB file | rolling 3 s window, 0.25 s hop | 32.0% (27-30% offline) | 0.35 / 1.0 s | ~40% |

## 2. Hard lessons

1. **Accent + jargon, not streaming, is the accuracy problem.** Whistle was 27-30% WER even fully offline, so its
   ~2-4 point streaming overhead was irrelevant. Errors were acoustic ("general examination" -> "general gemnison",
   "eye jaundice" -> "I jumped this"). Indian-English + medical vocabulary must be in training data. Clean
   LibriSpeech-style benchmarks hid this completely (Whistle scored 4% WER on a clean TTS clip).
2. **Rolling window costs ~window/hop x compute.** A 3 s window every 0.5 s encodes each second of audio ~6 times.
   A 0.6B offline model could not keep up that way on this CPU; the same-size cache-aware streaming model could.
   Streaming = each audio frame encoded once against a cache of past keys/values and conv states.
3. **But the rolling window's re-reading buys accuracy and early previews.** Offline Ultra (full bidirectional
   context) 11.3% vs streaming Nemotron 14.6%. The rolling window can also show a grey guess for the last 0.5 s
   and correct it; a streaming model only emits finalized words (~0.6 s late). Best of both, if affordable: a
   cheap first pass for preview + a better pass that commits (two-speed design).
4. **Fixed per-call overhead dominates small models.** Redux took ~230 ms even for a 1.5 s clip; the 120M EOU model
   used as much engine time (78%) as the 0.6B Nemotron q4. Graph rebuild / thread-team wakeups / decoder loops cost
   more than the matmuls at this size. A real-time model must be co-designed with its runtime: static graphs,
   persistent buffers, no per-chunk allocation, batched decoder steps.
5. **Threads: match physical performance cores.** 4 threads beat the default 8 by a wide margin (p50 lag 0.98 s
   -> 0.58 s, p95 2.99 s -> 1.44 s). Extra threads land on E-cores and spin. The runtime must not busy-spin between
   calls: an idle app burned 1-2.5 cores purely from VAD calls waking spinning thread pools.
6. **Bandwidth, not FLOPs, bounds CPU decoding.** Ternary Redux (213 MB) ran ~2.5x faster than 8-bit at equal
   size per Moondream; q4 vs q8 Nemotron: q4 used ~5 points less CPU for ~2 points more WER. Kernels matter as
   much as bits: an MSVC build ran the ternary kernel 17x slower than clang (scalar fallback).
7. **Multilingual models drift.** Parakeet v3 (25 EU languages, no language prompt) sometimes emitted French or
   Spanish on accented English. Either train with a language prompt/token (Nemotron lets you pin "en") or
   restrict the output vocabulary at decode time. Hindi is absent from all Parakeet models.
8. **Voice activity detection is part of the product.** A loudness threshold fails in real rooms (43% of quiet-room
   frames exceeded 0.008 RMS), so the model ran nonstop. A learned VAD fixes accuracy but costs CPU if called
   every 32 ms through a heavy runtime. Build VAD into the ASR model (an end-of-speech/VAD head is cheap) or make
   it a few-kFLOP standalone net that runs on a single thread.
9. **Punctuation + casing matter for readability.** The EOU 120M model was as accurate as Redux but emits lowercase
   without punctuation; captions are much harder to read. Train with punctuated, cased targets.
10. **Backlogs.** A streaming model at ~75% load occasionally ran slower than real time for ~10 s (other system
    load), and words arrived 3-5 s late until it recovered. Budget for <= 50% sustained load on the target CPU.
11. **Keyword biasing is wanted** (names, product terms like Claude/Fathom/Intelehealth). Only Whistle supported
    it; Parakeet's C API has none. Plan for shallow-fusion / boosting-tree hotwords in the decoder.

## 3. Requirements for our real-time model

- **Latency:** first word on screen <= 0.5 s p50, <= 1.0 s p95; finalized text <= 1.0 s p50.
- **Compute:** <= 50% of 4 P-cores of a 2022 laptop i5 (AVX2 only) while speech is present; ~0% in silence;
  Apple Silicon (NEON / Metal) as a second target.
- **Size:** <= 150-250M params for the always-on streaming path (0.6B is at this laptop's ceiling even
  streaming); a larger 0.6B "commit" model is fine if it runs only once per finished 6-10 s phrase.
- **Accuracy target:** <= 12% WER on accented Indian English meetings (Fathom's offline 12.8% is the bar to beat),
  and a real Hindi + Hinglish result (see README.md plan).
- **Outputs:** cased + punctuated text, word timestamps, per-word confidence, end-of-utterance events.

## 4. Design recommendations

- **Architecture: cache-aware streaming FastConformer, not a full-attention Transformer.** Full self-attention
  over a growing context is O(T^2) and re-encodes; use chunked attention with limited left context (cached) and
  a small right lookahead (e.g. 80-560 ms), causal depthwise convolutions with cached states, 8x subsampling
  (80 ms frames). Decoder: TDT or RNN-T (stateless / 1-layer prediction net); avoid an autoregressive Transformer
  decoder on the live path. Optional CTC head for cheap previews and alignment.
- **Train for both modes at once: dynamic-chunk / multi-lookahead training.** Randomize chunk size and right
  context during training (as Nemotron's multi-lookahead and U2/U2++ dynamic chunk training do), so one model
  serves a low-latency streaming setting and a high-accuracy full-context "commit" pass. This is the trained
  version of our "rolling window looks twice" idea: train on overlapping windows / chunk masks so the model is
  robust to cut words at window edges, and so a second look with more right context improves the same words.
- **Quantization: plan QAT from the start.** 4-bit or ternary weights (Redux shows ternary can keep within ~0.3
  WER of the base in English) give the bandwidth win that decides CPU speed. Post-training q4 cost ~2 WER points
  here; QAT should recover most of it. Keep the first/last layers and the joint network at 8-bit.
- **Runtime co-design:** fixed-shape chunk graphs compiled once, a persistent cache, a non-spinning thread pool
  sized to P-cores, VAD inside the same graph or a tiny single-thread VAD, and a C API: `stream_begin`,
  `stream_feed(pcm) -> words json`, `set_threads`, `set_hotwords`, `reset`.
- **Data:** add accented Indian-English conversational/meeting speech and medical vocabulary; keep a held-out
  real-meeting eval (not just LibriSpeech/FLEURS); include code-switched Hinglish.
- **Eval protocol (reuse):** RT Captions has `rt-captions bench <clip.wav> --ref @ref.txt` (WER, word latency
  spoken->on-screen p50/p95, engine busy %), `speed` (per-call ms by clip length), and a 5-min + 2-min densest-speech
  clip of a real meeting with Gemini references (scratchpad `eval/`, re-create from the Fathom recording).
