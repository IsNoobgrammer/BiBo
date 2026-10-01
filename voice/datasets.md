# Speech datasets: ASR + TTS (English, Hindi, Hinglish)

Verified Oct 1 2026 against HF API tags / dataset cards, arXiv HTML and lab sites. `[v]` = checked on the source page;
UNVERIFIED = figure from a paper or card that was not re-fetched. Re-check the license on the source page before any
product use -- licenses change (Common Voice left HF in Oct 2025).

Access: **D** direct download | **G** HF gate (share contact info, instant) | **F** form / terms | **R** request / permission only.
Commercial: **yes** = permissive (CC0 / CC-BY / Apache / MIT; CC-BY-SA = yes, but share-alike on the data itself) | **NO** = NC / academic only.

## Headline caveats
- **Hindi coverage is thinner than it looks.** Granary (643k h) has NO Hindi; Emilia has NO Hindi; MLS has no Hindi.
  The Hindi pool is IndicVoices, Vaani, Shrutilipi, Kathbath, SPRING-INX, MUCS, Gram Vaani, plus TTS sets.
- **Hinglish is almost absent.** MUCS 2021 has a Hindi-English code-switch subset; IndicVoices / Vaani have natural
  code-mixing inside Hindi speech. There is no large clean code-switched TTS corpus -- that is the gap to fill.
- **Non-commercial traps:** Emilia (not Emilia-YODAS), GigaSpeech, SPGISpeech, TED-LIUM 3, Expresso, EARS. Gram Vaani:
  commercial only with permission. IndicTTS (IIT Madras): custom license, commercial use UNVERIFIED.
- **Script normalization decides Hindi / Hinglish WER.** The same word appears in Devanagari or Latin script, numbers as
  digits or words. Fix one convention per dataset before mixing, or WER is inflated 5-10 points by spelling alone.
- **ASR and TTS want different audio.** TTS: studio-clean (SNR >= 25 dB, DNSMOS > 3). ASR: diversity (noise,
  accents, telephony). A clean-only ASR model breaks on real audio; augment or mix noisy data.

## ASR -- English
| Dataset / id | Hours | Domain | License | Comm. | Access | Pros | Cons |
|---|---|---|---|---|---|---|---|
| LibriSpeech `openslr/librispeech_asr` | 960 | read audiobooks | CC-BY-4.0 [v] | yes | D | standard, clean, every paper reports it | narrow domain, no punctuation |
| LibriHeavy `pkufool/libriheavy` | 50k [v] | read | Apache-2.0 [v] | yes | D | punctuation + casing, huge | still audiobooks only |
| People's Speech `MLCommons/peoples_speech` | 30k+ [v] | gov / legal / speeches | CC-BY / CC-BY-SA [v] | yes (SA) | D | diverse real speech | card: some transcripts are ASR output -> filter |
| MLS `facebook/multilingual_librispeech` | ~44.5k en | read | CC-BY-4.0 [v] | yes | D | big, 8 langs | no Hindi |
| VoxPopuli `facebook/voxpopuli` | ~1.8k labelled | EU parliament | CC0 [v] | yes | D | accented English, CC0 | formal register |
| YODAS2 `espnet/yodas2` | ~370k raw, 149 langs | YouTube | CC-BY-3.0 [v] | yes | D | massive, real-world, has some Hindi | noisy labels; must clean (OWSM recipe -> 166k h) |
| AMI `edinburghcstr/ami` | ~100 | meetings | CC-BY-4.0 [v] | yes | D | far-field, overlap | small |
| Earnings-21/22 `Revai/earnings21` | 39 / 119 | calls | CC-BY-SA-4.0 [v] | yes (SA) | D | long-form eval | eval-sized |
| Granary `nvidia/Granary` | 643k filtered, 25 EU langs | pseudo-labelled web | CC-BY-4.0 [v] | yes | D (manifests) | best-documented pseudo-label pipeline | **no Hindi**; audio from source corpora |
| GigaSpeech `speechcolab/gigaspeech` | 10k [v] | podcasts / YouTube / books | non-commercial [v] | **NO** | F | high quality, diverse | research only |
| SPGISpeech `kensho/spgispeech` | 5k [v] | earnings calls | academic only [v] | **NO** | F | punctuated, 50k speakers | eval only for a product |
| TED-LIUM 3 | 452 | talks | CC-BY-NC-ND | **NO** | D | classic benchmark | NC-ND; HF id UNVERIFIED |
| Common Voice | per language | crowd read | CC0 | yes | F (Mozilla Data Collective only) | CC0, many accents, has Hindi | left HF Oct 2025; variable quality |

## ASR -- Hindi / Indic
| Dataset | Publisher | Hours | Speakers | Domain | License | Access | Pros | Cons |
|---|---|---|---|---|---|---|---|---|
| **IndicVoices** `ai4bharat/IndicVoices` | AI4Bharat (IIT Madras) | 23.7k collected / 11.2k transcribed, 22 langs [v] | 51k, 400+ districts [v] | 8% read / 76% extempore / 15% conversational [v] | CC-BY-4.0 [v] | G | best Indic ASR set, spontaneous, demographic balance | Hindi-only hours UNVERIFIED; big download |
| **Vaani** `ARTPARK-IISc/Vaani` | IISc + ARTPARK | 31.3k total / 2.1k transcribed; **Hindi 14.9k h audio** [v] | 156k [v] | spontaneous, image-prompted | CC-BY-4.0 [v] | G | largest Hindi audio pool, huge speaker diversity -> SSL / pseudo-labelling | only ~7% transcribed; image-prompted captions are low-entropy (HF discussion #11) |
| Shrutilipi `ai4bharat/Shrutilipi` | AI4Bharat | 6.4k+, 12 langs [v] | - | All India Radio news | CC-BY-4.0 [v] | G | lots of Hindi broadcast | document-level mined alignment -> noisy text, needs filtering |
| Kathbath `ai4bharat/Kathbath` | AI4Bharat | 1,684, 12 langs [v] | 1,218, 203 districts [v] | read | CC0 [v] | G | clean, CC0, IndicSUPERB benchmark | read speech only |
| SPRING-INX | IIT Madras SPRING lab | ~2k, 10 langs (arxiv 2310.14654) | ? | mixed | CC-BY-4.0 | UNVERIFIED | manually transcribed | access route unverified |
| MUCS 2021 | openslr 103/104 | Hindi 95 h + hi-en code-switch | ? | read / lecture | UNVERIFIED | D | **real Hinglish** labels | small; license to confirm |
| Gram Vaani | openslr 118 | 1,108 (100 labelled + 1,000 unlabelled + 8 dev/eval) | ? | 8 kHz telephony, regional Hindi | academic free; commercial by permission | D / R | only real phone-quality Hindi | mostly unlabelled; commercial needs permission |
| Bhashini / ULCA, AI Kosh | MeitY | ? | ? | ? | per dataset | R / portal | government Indic data (IndicConformer used AI Kosh) | UNVERIFIED; access per dataset |

## TTS -- English
| Dataset | Hours / speakers | License | Comm. | Pros | Cons |
|---|---|---|---|---|---|
| LJSpeech `keithito/lj_speech` | 24 / 1 | public domain [v] | yes | single-speaker baseline, everyone uses it | one voice, 22 kHz |
| LibriTTS-R `mythicinfinity/libritts_r` | 585 / 2,456 | CC-BY-4.0 [v] | yes | Miipher-restored, multi-speaker | read audiobooks |
| Hi-Fi TTS `MikhailT/hifi-tts` | 292 / 10 | CC-BY-4.0 [v] | yes | 44.1 kHz, high fidelity | 10 speakers |
| GLOBE `MushanW/GLOBE` | 535 / 23,519 | CC0 [v] | yes | worldwide accents, zero-shot cloning | derived from Common Voice (variable mic quality) |
| VCTK `CSTR-Edinburgh/vctk` | ~44 / 110 | CC-BY-4.0 [v] | yes | accents, classic | old, short sentences |
| Emilia-YODAS (part of `amphion/Emilia-Dataset`) | large share of 215.6k total [v] | CC-BY-4.0 [v] | yes | spontaneous, in-the-wild, Emilia-Pipe cleaned | **main Emilia part is CC-BY-NC**; no Hindi |
| Emilia (main) | en ~139k of 215.6k [v] | CC-BY-NC [v] | **NO** | what F5-TTS etc. train on | non-commercial |
| Expresso `ylacombe/expresso` | 40 / 4 [v] | CC-BY-NC [v] | **NO** | expressive styles, 48 kHz | NC |
| EARS (facebookresearch GitHub) | 100 / 107 [v] | CC-BY-NC [v] | **NO** | anechoic, 22 emotions | NC |

## TTS -- Hindi / Indic
| Dataset | Publisher | Hours / speakers | License | Access | Pros | Cons |
|---|---|---|---|---|---|---|
| **SYSPIN** (vaani.iisc.ac.in/dataset/syspindataset) | IISc SPIRE lab | 920 h / 18 speakers, 9 langs incl. Hindi, Bhojpuri, Magahi [v] | CC-BY-4.0 [v] | D | studio 48 kHz / 24-bit, LIMMITS challenge source | few speakers; agriculture / finance text domain |
| **IndicVoices-R** `ai4bharat/indicvoices_r` | AI4Bharat | 1,704 h / 10,496, 22 langs [v] | CC-BY-4.0 [v] | G | restored to 48 kHz, many speakers -> zero-shot | restored audio (artifacts possible), not studio |
| **Rasa** `ai4bharat/Rasa` | AI4Bharat | 1,145 h, 22 langs; Hindi 50.8 h, 2 speakers [v] | CC-BY-4.0 [v] | G | studio, 6 emotions | only 2 Hindi speakers |
| IndicTTS `SPRINGLab/IndicTTS-Hindi` | IIT Madras | ~20 h native + 20 h Indian English / lang, 13 langs | IITM license (commercial UNVERIFIED) | F | classic studio Hindi + Indian English | license; few speakers |

## Evaluation sets (never train on these)
- **ASR, English:** HF Open ASR Leaderboard `hf-audio/open-asr-leaderboard` (AMI, Common Voice, Earnings22, GigaSpeech,
  LibriSpeech, SPGI, TED-LIUM, VoxPopuli; WER + RTFx). Long-form: Earnings21/22, CORAAL.
- **ASR, Hindi:** `VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN` (5.8 h spontaneous; license UNVERIFIED), FLEURS hi
  (`google/fleurs`), Vistaar benchmark (Kathbath / Kathbath-Hard, FLEURS, CV, IndicTTS, MUCS, Gram Vaani), Lahaja
  `ai4bharat/Lahaja` (12.5 h, accents), Svarah `ai4bharat/Svarah` (9.6 h Indian English), `ARTPARK-IISc/Vaani-Benchmark-V1.0`.
- **TTS:** LibriSpeech test-clean zero-shot protocol (WER via ASR + speaker similarity), IndicVoices-R benchmark split;
  metrics UTMOS / DNSMOS, ASR CER / WER on the generated audio, speaker-embedding similarity.

## How labs select data (copy the filters, not the scale)
| Lab / model | Scale | Selection |
|---|---|---|
| OpenAI Whisper | 680k h, 100% weak labels | drop machine-made transcripts (all caps / no punctuation); audio language ID must match text; drop sources with bad initial-model WER; fuzzy dedup; remove eval overlap |
| CMU OWSM v4 | YODAS 370k -> 166k h | CTC forced-alignment re-segmentation <= 30 s; language ID agreement (text fastText + audio ECAPA + label); per-language CTC-confidence quantile 0.10 |
| NVIDIA Granary / Parakeet | 1.06M -> 643k h (60.7% kept) | Whisper-large-v3 two-pass pseudo-labels; lid_prob < 0.8, repeated n-grams, char rate / char set, hallucination list; LLM punctuation kept only if CER <= 5%. Parakeet-0.6B-v2 = 10k h human + 110k h pseudo |
| Meta MMS | 44.7k h, 1,107 langs | two-round forced alignment, keep length-normalized score > -0.2, cross-validated ASR drops mismatches |
| Google USM | 12M h unlabelled | self-supervised pretraining, small labelled fine-tune |
| AI4Bharat | IndicVoices; IndicWhisper 10.7k h | district / demographic-balanced collection, 3 styles, native transcribers; IndicWhisper = Whisper fine-tuned on public labelled data |
| IISc SraVaani 1.0 (arxiv 2608.08235) | 28.4k h SSL + 30.6k h supervised | SSL on Vaani (0.5-25 s), supervised on 24 public sets (0.1-40 s), 65 langs |
| Sarvam Saaras v3 | "1M+ h curated" | pretrain -> SFT -> RL; pipeline unpublished |
| Emilia-Pipe (TTS) | keeps 29.4% of raw audio | 24 kHz -20 dBFS; UVR-MDX-Net separation; pyannote 3.1 diarization; Silero VAD 3-30 s; WhisperX; LID > 0.8, DNSMOS > 3.0, phone-duration IQR outliers dropped |
| IndicVoices-R (TTS) | 1.7k h from IndicVoices | HTDemucs -> VoiceFixer -> DeepFilterNet3; C50 >= 30 dB, SNR >= 25 dB, 0.2-30 s, pitch mean <= 350 Hz / std <= 150 Hz, <= 30 chars/s |

## Starting recipe (permissive only)
- **ASR Hindi:** IndicVoices + Vaani-transcribed + Kathbath + SPRING-INX + Shrutilipi (filtered); pseudo-label Vaani's
  14.9k h unlabelled Hindi. **ASR English:** LibriHeavy + People's Speech (filtered) + YODAS2-en (OWSM-cleaned).
  **Hinglish:** MUCS 2021 + code-mixed segments mined from IndicVoices / Vaani.
- **TTS Hindi:** SYSPIN Hindi + Rasa Hindi (studio voices) + IndicVoices-R (multi-speaker). **TTS English:** LibriTTS-R
  + Hi-Fi TTS + GLOBE + Emilia-YODAS.
- **Pipeline:**
  1. standardize (16 kHz ASR / 24-48 kHz TTS), loudness-normalize, keep license + source per row
  2. VAD segments 1-30 s (TTS 3-20 s); diarize long-form audio
  3. audio + text language ID, keep confidence >= 0.8, TAG Hinglish instead of dropping it
  4. pseudo-label with IndicConformer / IndicWhisper + Whisper-large-v3; keep if they agree (CER <= 10-15%, tune)
     or CTC confidence is above a per-language quantile
  5. hallucination filters (repeated n-grams, char rate, char set); script + numeral normalization
  6. TTS only: separation if needed, DNSMOS > 3.0, SNR >= 25 dB, C50 / speaking-rate cuts, re-transcribe
  7. fuzzy dedup + remove eval overlap (Vistaar, Lahaja, Svarah, FLEURS, Open ASR sets)
  8. human : pseudo about 1:5-1:10 with human data oversampled (Parakeet 1:11); evaluate every round

Source report with all links: `~/researcher/asr-tts-datasets_01_OCT_2026/README.md`.
