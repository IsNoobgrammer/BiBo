"""Figures that explain ASR front ends with REAL clips and CONTRASTS (what each representation keeps / discards).

    python voice/viz/frontends_viz.py --run1 /home/marimo/work/asr/run1 --out /home/marimo/work/viz [--push fhai50032/bibo-viz]

fig1_resample      48 kHz sweep -> 16 kHz: naive decimation aliases, a proper resampler low-passes first
fig2_pitch         low- vs high-pitch speaker: waveform / linear spectrogram / 80-band log-mel / 13 MFCC
fig3_noise         same clip clean vs 4-talker babble at 5 dB SNR: waveform / log-mel / wav2vec2 features + similarity
fig4_rates         one clip through every front end on one time axis: log-mel 100/s, HuBERT k-means 50/s,
                   EnCodec 75/s, Mimi 12.5/s, our FastConformer encoder 12.5/s
fig5_langs         English vs Hindi clip: log-mel + Mimi tokens
fig6_ctc           our pretrained 114M model's CTC head per 80 ms frame: blank vs token, collapsed to text
CPU only (training owns the GPU).
"""
import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
import torchaudio.transforms as T  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "asr"))
import augment  # noqa: E402

SR = 16000
torch.set_num_threads(8)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.titlesize": 10})
MEL = T.MelSpectrogram(SR, n_fft=512, win_length=400, hop_length=160, n_mels=80)
MFCC = T.MFCC(SR, n_mfcc=13, melkwargs=dict(n_fft=512, win_length=400, hop_length=160, n_mels=80))


def logmel(x):
    return torch.log(MEL(torch.from_numpy(x)) + 1e-6).numpy()


def spec(x):
    return 20 * np.log10(np.abs(torch.stft(torch.from_numpy(x), 512, 160, 400, torch.hann_window(400),
                                           return_complex=True).numpy()) + 1e-6)


def load(r, secs=4.0):
    x, _ = sf.read(r["audio_filepath"], dtype="float32")
    return x[: int(secs * SR)]


def f0(x):
    p = AF.detect_pitch_frequency(torch.from_numpy(x), SR).numpy()
    p = p[(p > 60) & (p < 400)]
    return float(np.median(p)) if len(p) else 0.0


def img(ax, m, title, sr_axis=None, cmap="magma", ylabel=None):
    t = m.shape[1] / 100 if sr_axis is None else m.shape[1] / sr_axis
    ax.imshow(m, origin="lower", aspect="auto", cmap=cmap, extent=[0, t, 0, m.shape[0]])
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def fig1(out):
    sr_hi = 48000
    t = np.arange(int(2.0 * sr_hi)) / sr_hi
    sweep = np.sin(2 * np.pi * (200 * t + (20000 - 200) / (2 * 2.0) * t ** 2)).astype(np.float32) * 0.5
    naive = sweep[::3]
    proper = AF.resample(torch.from_numpy(sweep), sr_hi, SR).numpy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a, x, sr, title in [(ax[0], sweep, sr_hi, "48 kHz original: one tone sweeping 200 Hz -> 20 kHz"),
                            (ax[1], naive, SR, "16 kHz, NAIVE (keep every 3rd sample): above 8 kHz folds back = fake tones"),
                            (ax[2], proper, SR, "16 kHz, PROPER resample (low-pass at 8 kHz first): clean cut")]:
        S = 20 * np.log10(np.abs(torch.stft(torch.from_numpy(x), 1024, 256, window=torch.hann_window(1024),
                                            return_complex=True).numpy()) + 1e-6)
        a.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, len(x) / sr, 0, sr / 2000], vmin=-60)
        a.set_title(title, fontsize=8.5)
        a.set_xlabel("time (s)")
        a.set_ylabel("frequency (kHz)")
    fig.tight_layout()
    fig.savefig(f"{out}/fig1_resample.png")


def fig2(rows, out):
    cand = [(f0(load(r)), r) for r in rows[:300]]
    cand = [c for c in cand if c[0] > 0]
    lo, hi = min(cand, key=lambda c: c[0]), max(cand, key=lambda c: c[0])
    fig, ax = plt.subplots(4, 2, figsize=(13, 10))
    for j, (p, r) in enumerate([lo, hi]):
        x = load(r)
        ax[0, j].plot(np.arange(len(x)) / SR, x, lw=0.4)
        ax[0, j].set_title(f"{'LOW' if j == 0 else 'HIGH'} pitch speaker (median F0 {p:.0f} Hz): \"{r['text'][:60]}\"")
        ax[0, j].set_xlim(0, len(x) / SR)
        S = spec(x)[:160]
        img(ax[1, j], S, "linear spectrogram 0-5 kHz: horizontal stripes = pitch harmonics (spacing = F0)",
            ylabel="FFT bin (31 Hz each)")
        img(ax[2, j], logmel(x), "80-band log-mel (OURS): harmonics blurred, formants (vowel shape) kept",
            ylabel="mel band")
        img(ax[3, j], MFCC(torch.from_numpy(x)).numpy(), "13 MFCC: smooth envelope only, pitch largely gone",
            cmap="coolwarm", ylabel="coefficient")
        ax[3, j].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_pitch.png")


def w2v_feats(x, model, fe):
    with torch.no_grad():
        return model(**fe(x, sampling_rate=SR, return_tensors="pt")).last_hidden_state[0].numpy()


def fig3(r, out, w2v, fe):
    x = load(r)
    babble = sum(np.resize(sf.read(p, dtype="float32")[0], len(x)) for p in BABBLE[:4])   # 4 other talkers
    y = augment._at_snr(x, babble, 5.0).astype(np.float32)   # noise only (no speed change: keeps frames aligned)
    fx, fy = w2v_feats(x, w2v, fe), w2v_feats(y, w2v, fe)
    n = min(len(fx), len(fy))
    cos = (fx[:n] * fy[:n]).sum(1) / (np.linalg.norm(fx[:n], axis=1) * np.linalg.norm(fy[:n], axis=1))
    mx, my = logmel(x), logmel(y)
    m = min(mx.shape[1], my.shape[1])
    mel_cos = (mx[:, :m] * my[:, :m]).sum(0) / (np.linalg.norm(mx[:, :m], axis=0) * np.linalg.norm(my[:, :m], axis=0))
    wav_corr = np.corrcoef(x[:len(y)], y[:len(x)])[0, 1]
    fig, ax = plt.subplots(4, 2, figsize=(13, 9.5))
    for j, (s, name) in enumerate([(x, "CLEAN"), (y, "SAME CLIP + babble of 4 other talkers at 5 dB SNR")]):
        ax[0, j].plot(np.arange(len(s)) / SR, s, lw=0.4)
        ax[0, j].set_title(name)
        img(ax[1, j], logmel(s), "80-band log-mel", ylabel="mel band")
        img(ax[2, j], (fx if j == 0 else fy).T[:128], "wav2vec2-base features (first 128 of 768 dims, 50 / s)",
            sr_axis=50, cmap="viridis", ylabel="dim")
    gs = ax[3, 0].get_gridspec()
    for a in ax[3]:
        a.remove()
    a = fig.add_subplot(gs[3, :])
    a.plot(np.arange(n) / 50, cos, label=f"wav2vec2 frame cosine (mean {cos.mean():.2f})")
    a.plot(np.arange(m) / 100, mel_cos, label=f"log-mel frame cosine (mean {mel_cos.mean():.2f})", alpha=0.7)
    a.set_title(f"How similar is clean vs noisy, frame by frame? (raw waveform correlation: {wav_corr:.2f})")
    a.set_xlabel("time (s)")
    a.set_ylim(-0.2, 1.05)
    a.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_noise.png")


def tokens_strip(ax, ids, rate, title, k=None):
    ids = np.asarray(ids)
    ax.imshow(ids[None, :] % 20, aspect="auto", cmap="tab20", extent=[0, len(ids) / rate, 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.set_title(f"{title}: {len(ids)} tokens for {len(ids) / rate:.1f} s = {rate:g} / s"
                 + (f", vocabulary {k}" if k else ""), fontsize=9)


def codec_codes(x, name):
    from transformers import AutoFeatureExtractor, EncodecModel, MimiModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    m = (MimiModel if "mimi" in name else EncodecModel).from_pretrained(name).eval()
    x24 = AF.resample(torch.from_numpy(x), SR, fe.sampling_rate).numpy()
    with torch.no_grad():
        enc = m.encode(**fe(raw_audio=x24, sampling_rate=fe.sampling_rate, return_tensors="pt"))
    return enc.audio_codes.reshape(-1, enc.audio_codes.shape[-2], enc.audio_codes.shape[-1])[0, 0].numpy()


def our_encoder(x, asr):
    with torch.no_grad():
        sig = torch.from_numpy(x)[None]
        mel, ml = asr.preprocessor(input_signal=sig, length=torch.tensor([len(x)]))
        enc, el = asr.encoder(audio_signal=mel, length=ml)
        logp = asr.ctc_decoder(encoder_output=enc)
    return mel[0].numpy(), enc[0].numpy(), logp[0].numpy()


def fig4(r, out, hub, hfe, km, asr):
    x = load(r, 5.0)
    mel, enc, _ = our_encoder(x, asr)
    hu = w2v_feats(x, hub, hfe)
    fig, ax = plt.subplots(7, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [1.2, 2, 2, 0.5, 0.5, 0.5, 2]})
    ax[0].plot(np.arange(len(x)) / SR, x, lw=0.4)
    ax[0].set_xlim(0, len(x) / SR)
    ax[0].set_title(f"waveform: {len(x)} numbers for {len(x) / SR:.0f} s (16,000 / s)   \"{r['text'][:80]}\"")
    img(ax[1], mel, f"log-mel, NeMo's own preprocessor (what OUR model sees): {mel.shape[0]} bands x {mel.shape[1]} frames = 100 / s",
        ylabel="mel band")
    img(ax[2], hu.T[:128], f"HuBERT-base features (self-supervised): 768 dims x {len(hu)} frames = 50 / s", sr_axis=50,
        cmap="viridis", ylabel="dim")
    tokens_strip(ax[3], km.predict(hu), 50, "HuBERT k-means 'semantic' tokens", k=km.n_clusters)
    tokens_strip(ax[4], codec_codes(x, "facebook/encodec_24khz"), 75, "EnCodec (acoustic codec), codebook 1", k=1024)
    tokens_strip(ax[5], codec_codes(x, "kyutai/mimi"), 12.5, "Mimi (low-rate codec), codebook 1", k=2048)
    img(ax[6], enc[:128], f"OUR FastConformer encoder output (after 8x subsampling + 17 layers): 512 dims x {enc.shape[1]} "
        f"frames = 12.5 / s", sr_axis=12.5, cmap="viridis", ylabel="dim")
    ax[6].set_xlabel("time (s) -- every row is the same 5 s of audio")
    fig.tight_layout()
    fig.savefig(f"{out}/fig4_rates.png")


def fig5(en, hi, out):
    fig, ax = plt.subplots(3, 2, figsize=(13, 7))
    for j, (r, name) in enumerate([(en, "ENGLISH (Svarah, Indian accent)"), (hi, "HINDI (Kathbath)")]):
        x = load(r)
        ax[0, j].plot(np.arange(len(x)) / SR, x, lw=0.4)
        ax[0, j].set_xlim(0, len(x) / SR)
        ax[0, j].set_title(name + ("  \"" + r["text"][:50] + "\"" if r["lang"] == "en" else ""))
        img(ax[1, j], logmel(x), "80-band log-mel", ylabel="mel band")
        tokens_strip(ax[2, j], codec_codes(x, "kyutai/mimi"), 12.5, "Mimi tokens", k=2048)
    fig.tight_layout()
    fig.savefig(f"{out}/fig5_langs.png")


def fig6(r, out, asr):
    x = load(r, 4.0)
    mel, enc, logp = our_encoder(x, asr)
    blank = logp.shape[1] - 1
    ids = logp.argmax(1)
    prob = np.exp(logp.max(1))
    vocab = asr.tokenizer
    fig, ax = plt.subplots(3, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2, 1.4, 2.2]})
    img(ax[0], mel, "input: log-mel (100 frames / s)", ylabel="mel band")
    ax[1].bar(np.arange(len(ids)) * 0.08, prob, width=0.07, color=["#bbbbbb" if i == blank else "#d1495b" for i in ids],
              align="edge")
    ax[1].set_xlim(0, len(ids) * 0.08)
    ax[1].set_title("CTC head, one decision per 80 ms frame: grey = BLANK (nothing new), red = emits a token "
                    "(bar = confidence)")
    for k, i in enumerate(ids):
        if i != blank and (k == 0 or ids[k - 1] != i):
            ax[1].text(k * 0.08 + 0.035, 1.02, vocab.ids_to_tokens([int(i)])[0], rotation=90, fontsize=7, ha="center",
                       va="bottom")
    ax[1].set_ylim(0, 1.45)
    collapsed, prev = [], None
    for i in ids:
        if i != blank and i != prev:
            collapsed.append(int(i))
        prev = i
    ax[2].axis("off")
    ax[2].text(0, 0.85, f"frames: {len(ids)}  (4 s / 80 ms)   blank frames: {(ids == blank).sum()}   "
               f"token frames: {(ids != blank).sum()}", fontsize=10)
    ax[2].text(0, 0.6, "collapse repeats + drop blanks ->  " + vocab.ids_to_text(collapsed), fontsize=11, color="#d1495b")
    ax[2].text(0, 0.35, "reference ->  " + r["text"][:110], fontsize=11)
    ax[2].text(0, 0.1, "(pretrained NVIDIA English model, before our fine-tune)", fontsize=9, color="#666666")
    fig.tight_layout()
    fig.savefig(f"{out}/fig6_ctc.png")


BABBLE = []


def main():
    global BABBLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--push", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rd = lambda n: [json.loads(l) for l in open(os.path.join(a.run1, n), encoding="utf-8")]
    sv, kb, ps = rd("val_svarah.jsonl"), rd("val_kathbath_hi.jsonl"), rd("val_peoples_speech.jsonl")
    random.Random(1).shuffle(sv)
    long_en = [r for r in sv if r["duration"] >= 5 and len(r["text"].split()) >= 8]
    BABBLE = [r["audio_filepath"] for r in ps[:40]]
    fig1(a.out); print("fig1", flush=True)
    fig2([r for r in sv if r["duration"] >= 4], a.out); print("fig2", flush=True)
    from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2Model
    w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
    wfe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    fig3(long_en[0], a.out, w2v, wfe); print("fig3", flush=True)
    hub = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval()
    from sklearn.cluster import KMeans
    feats = np.concatenate([w2v_feats(load(r), hub, wfe) for r in sv[:40]])
    km = KMeans(100, n_init=3, random_state=0).fit(feats)
    import nemo.collections.asr as nemo_asr
    asr = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_hybrid_large_streaming_multi",
                                                   map_location="cpu").eval()
    fig4(long_en[1], a.out, hub, wfe, km, asr); print("fig4", flush=True)
    fig5(long_en[2], next(r for r in kb if r["duration"] >= 4), a.out); print("fig5", flush=True)
    fig6(long_en[3], a.out, asr); print("fig6", flush=True)
    if a.push:
        from huggingface_hub import HfApi
        HfApi().create_repo(a.push, repo_type="dataset", private=True, exist_ok=True)
        HfApi().upload_folder(repo_id=a.push, repo_type="dataset", folder_path=a.out, path_in_repo="frontends")
        print("pushed", a.push, flush=True)


if __name__ == "__main__":
    main()
