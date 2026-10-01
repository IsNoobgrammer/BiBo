# Speech model landscape (ASR, TTS, unified), Oct 1 2026

Sourced from fetched pages (links inline). `(mem)` = recalled by the research agent, not re-fetched this session ->
UNVERIFIED. Re-check numbers before quoting them externally.

## ASR

### English: HF Open ASR Leaderboard (avg WER, lower is better)
Source: benchmarklist.com/benchmarks/open_asr_leaderboard, RTFx from arxiv 2510.06961 / model cards.

| Model | Params | Architecture | Avg WER | RTFx | Notes |
|---|---|---|---|---|---|
| IBM Granite Speech 4.1 2B | 2B | - | 5.33 | ? | top as of mid-2026 |
| Cohere Transcribe 03-2026 | 2B | big Conformer enc + light Transformer AED | 5.42 | 525 | 14 langs, no Hindi, Apache |
| Granite-4.0-1B-speech | ~2B listed | CTC Conformer enc + window-Q projector + 1B LLM | 5.52 | 280 | >100k h, no Hindi |
| Canary-Qwen-2.5B | 2.5B | FastConformer + Qwen LLM (SALM) | 5.63 | 418 | |
| Qwen3-ASR-1.7B | 1.7B | Qwen3-Omni audio enc + LLM | 5.76 | - | 30 langs incl. Hindi |
| Parakeet-TDT-0.6B-v2 | 0.6B | FastConformer-TDT | 6.05 | 3390 | English only |
| Parakeet-TDT-0.6B-v3 | 0.6B | FastConformer-TDT | 6.32 | 3330 | 25 EU langs, no Hindi |
| Canary-1B-flash | 883M | FastConformer 32L enc + 4L dec | 6.35 | 1046 | 85k h, CC-BY-4.0 |
| Qwen3-ASR-0.6B | 0.9B listed | as above | 6.42 | 166 | Hindi supported |
| Moonshine-streaming-medium | 245M | sliding-window enc + AR dec | 6.65 | - | 300k h, English only, MIT |
| Canary-180M-flash | 180M | FastConformer AED | 7.12 | - | |
| Whisper large-v3 / turbo | 1.55B / 809M | AED | 7.44 / 7.83 | 146 / 200 | |

**Takeaway:** below 600M, attention-free FastConformer-TDT gives ~6.0-6.3 WER at >3000x RTFx. LLM decoders buy only
~0.5-0.7 WER at 10x lower speed.

### Hindi / Indic
| Model | Params | Architecture | Data | Hindi WER |
|---|---|---|---|---|
| IndicConformer-600M-multi (AI4Bharat) | 600M | Conformer, hybrid CTC + RNNT | 22 langs | 13.1 on Vaani (best open); 13.8 avg over 8 sets (SraVaani's comparison); MIT |
| SraVaani-1.0 (IISc / ARTPARK) | ~430M | FastConformer hybrid TDT-CTC, 17L, d=1024 | 31k h SSL from scratch + 31k h labelled, 65 langs | 14.0 avg over 8 sets (arxiv 2608.08235) |
| Sarvam Saaras v3 (closed) | ? | streaming | ? | 13.1 Vaani; ~19.3 IndicVoices top-10 |
| Nemotron-3.5-ASR-streaming-0.6B | 600M | cache-aware FastConformer-RNNT | 40 locales | 6.81 FLEURS-hi; 20.2 Vaani |
| IndicWhisper (Vistaar 2023) | 1.55B | Whisper-large fine-tune | ~10.7k h (mem) | Kathbath 10.3, Kathbath-Hard 12.0, FLEURS 11.4, CV 15.0, 7-set avg 13.6 (arxiv 2305.15386) |
| Omnilingual ASR (Meta, Nov 2025) | 300M-7B | w2v2 enc + CTC / LLM-style dec | 4.3M h SSL, 1600+ langs | OmniASR_LLM_1B 25.6 Vaani |
| Whisper-large-v3 | 1.55B | AED | - | 26.4 Vaani |
| Gemini-3.1-Pro / Google Chirp 3 | closed | - | - | 12.3 / 15.6 Vaani |

Vaani numbers: Vaani Benchmark v1.0 (arxiv 2606.21408). **Hinglish (code-switched) tokens hurt every system**:
Shrutam 12.8% -> 28.3%, Chirp 3 10.2% -> 12.8%. MonsoonASR (Hindi + Indian English) became an Open ASR
Leaderboard track on 2026-08-28; its per-model WERs are UNVERIFIED.

## TTS (<= ~1B focus)
Seed-TTS-eval test-en (WER %, speaker SIM). Sources: Spark-TTS arxiv 2503.01710, dots.tts arxiv 2606.07080,
CosyVoice 3 arxiv 2505.17589.

| Model | Params | Family | Codec / latent | Data | en WER / SIM | License | Hindi |
|---|---|---|---|---|---|---|---|
| Kokoro-82M | 82M | StyleTTS2 + ISTFTNet, no cloning | mel -> iSTFT | few hundred h | n/a | Apache | 4 weak voices (<10 h) |
| F5-TTS | 0.3B | flow-matching DiT, NAR | mel + Vocos | ~95k h Emilia (mem) | 1.83 / 0.647 | weights CC-BY-NC (mem) | via IndicF5 |
| **IndicF5** | 0.4B | F5 fine-tune | mel | 1,417 h, 11 Indic langs | - | MIT | **yes** |
| Spark-TTS | 0.5B | AR LM (Qwen2.5-0.5B) | BiCodec 50 TPS + global speaker tokens | VoxBox 102k h | 1.98 / 0.584 | CC-BY-NC-SA (mem) | no |
| CosyVoice 2 | 0.5B LM + flow | AR LM -> flow to mel | S3 25 Hz FSQ 6561 (mem) | ~170k h (mem) | 2.57 / 0.652 | Apache | no |
| CosyVoice 3-0.5B (RL) | 0.5B LM + 0.1B flow | same | 25 Hz MinMo-derived | 1M h | 1.76 / 0.774 | - | no |
| VibeVoice-Realtime-0.5B | 0.5B | LLM + diffusion head, continuous | 7.5 Hz acoustic VAE | - | 2.05 / 0.633 | MIT | no |
| MegaTTS 3 | 0.5B | DiT + WaveVAE | latent | - | 2.79 / 0.771 | - | no |
| Qwen3-TTS-12Hz 0.6B / 1.7B | 0.6 / 1.7B | AR multi-codebook | 12 Hz | - | 1.7B: 1.24 | Apache | no (10 langs) |
| **Chatterbox (multilingual)** | 0.5B Llama | AR -> flow | S3-style (mem) | 0.5M h | - | MIT | **yes** (+ Hindi fine-tune) |
| MaskGCT | ~1B (mem) | masked NAR | w2v-BERT semantic + DAC-style | Emilia 100k h | 2.62 / 0.714 | NC (mem) | no |
| Llasa-1B / 3B / 8B | 1-8B | AR | X-codec2 | 250k h | 3.22 / 3.14 / 2.97 | NC (mem) | no |
| Kyutai TTS-1.6B | 1.8B | delayed streams | Mimi 12.5 Hz, 32 cb | 2.5M h | - | CC-BY-4.0 | no |
| **Indic Parler-TTS** | 0.9B | description-prompted AR over DAC (mem) | DAC | 1,806 h, 21 langs, 69 voices | - | Apache | **yes** |
| **Veena** (Maya Research) | 3B Llama | AR | SNAC 24 kHz | - | - | Apache | **Hindi + English, code-mixed** |
| Sarvam Bulbul v3 | closed API | - | 48 kHz | 11 langs | - | closed | yes |

**Takeaway:** at 0.3-0.6B both families reach en WER ~1.7-2.6. Flow matching (F5 / IndicF5) is the cheapest route,
no tokenizer needed. Codec-LM + flow (CosyVoice 2/3, Chatterbox) gives the best SIM and streaming. The only open
Hindi TTS under 500M are IndicF5 (0.4B, MIT) and Kokoro (weak Hindi).

## Unified text/speech -> text/speech
| Model | Params | Speech in | Speech out | Layout | Data | Small-scale result |
|---|---|---|---|---|---|---|
| **UniVoice** | **0.4B** (SmolLM2-360M) | continuous mel | flow matching on mel, same backbone | causal mask for ASR, bidirectional for TTS | 50k h LibriHeavy | ASR 3.0 / 6.3; TTS WER 4.06, SIM 0.56, UTMOS 3.72 (arxiv 2510.04593) |
| **OpusLM** (CMU / ESPnet) | 135M / 360M / 1.7B / 7B | 1 semantic + 8 acoustic tokens @ 50 Hz, delay interleave, summed embeddings | same | task tokens; loss text : semantic : acoustic = 1 : 1/2 : 1/8 | 213k h + 292B text tokens | ASR clean/other 135M 6.9/11.1, 360M 4.2/8.7, 1.7B 2.5/5.7; **TTS WER 38.7 / 19.8 / 6.0**; 7B MMLU 59 |
| Mini-Omni / Mini-Omni2 | 0.5B (Qwen2-0.5B) | Whisper enc | SNAC 7 layers, parallel + delay, text-led | parallel streams | 9k h | ASR 4.8 / 9.8 |
| SLAM-Omni | 0.5B | Whisper | grouped 50 Hz semantic tokens | single-stage | small | competitive at 0.5B (arxiv 2412.15649) |
| LLaMA-Omni 2 | 0.5-14B | Whisper enc | AR streaming over CosyVoice 2 tokens -> flow | read R text, write W speech tokens | 200k dialogues | beats GLM-4-Voice on SpokenQA |
| Moshi | 7B | Mimi 12.5 Hz, 8 cb | same | parallel user/model streams + inner monologue | 7M h | ASR 5.7 |
| GLM-4-Voice | 9B | 12.5 Hz Whisper-VQ | same + flow | interleaved text:speech 13:26 (mem) | ~1T tokens | ASR 2.8 / 7.7 |
| Kimi-Audio | 7B | 12.5 Hz semantic VQ + continuous Whisper features | semantic -> flow + BigVGAN | parallel heads | 13M h | LibriSpeech 1.28 / 2.42 |
| MiMo-Audio | 7B + 1.2B tokenizer | 25 Hz RVQ-8, patched to 6.25 Hz | patch decoder + delay | - | >100M h | few-shot emergence |

### Speech tokenizers / codecs
| Codec | Rate | Codebooks | Bitrate | Semantic |
|---|---|---|---|---|
| Mimi | 12.5 Hz | 8 (up to 32) x 2048 | ~1.1 kbps | 1st codebook distilled from WavLM (mem) |
| X-codec2 | 50 Hz | 1 FSQ 65,536 | 0.8 kbps | fused w2v-BERT 2.0 (mem) |
| WavTokenizer | 40 / 75 Hz | 1 x 4096 | 0.48 / 0.9 kbps | none (mem) |
| BiCodec | 50 Hz + global speaker tokens | 1 x 8192 | 0.65 kbps | wav2vec2-based (mem) |
| CosyVoice S3 v2 | 25 Hz | 1 FSQ 6561 | ~0.3 kbps | ASR-supervised (mem) |
| XY-Tokenizer | 12.5 Hz | RVQ-8 | 1 kbps | dual tower |
| DualCodec | 12.5 / 25 Hz | RVQ, layer 1 from w2v-BERT-2 L16 | 0.80-0.93 kbps | yes |

No codec reports Hindi reconstruction quality: measure Hindi resynthesis WER before committing to one.
