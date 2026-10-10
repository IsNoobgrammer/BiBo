"""Perturbed copies for repeated rows: a repeat should be NEW audio of the same words, not a byte-identical duplicate.

Per copy (seeded per row + copy index): speed 0.9x / 1.1x (Kaldi-style resample: tempo + pitch; text unchanged) ->
synthetic room reverb (30%) -> noise at 5-20 dB SNR (half babble = 3-5 other training utterances summed, half
white / pink / brown) -> gain -6..+3 dB. No external noise corpus needed.
ponytail: synthetic RIR + coloured noise; swap in MUSAN / real RIRs if the noisy-meeting eval says it matters.
"""
import os
import zlib

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

SR = 16000
torch.set_num_threads(1)                                           # runs inside a 16-process Pool


def _colored(n, kind, rng):
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1] if n > 1 else 1.0
    shape = 1 / np.sqrt(f) if kind == "pink" else 1 / f          # pink 1/f power, brown 1/f^2 power
    return np.fft.irfft(np.fft.rfft(w) * shape, n).astype(np.float32)


def _rir(rng):
    t60 = rng.uniform(0.2, 0.7)
    n = int(t60 * SR)
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
    h[0] = 1.0                                                     # direct path
    return h / np.abs(h).sum() * 4


def _at_snr(x, noise, snr_db):
    ps, pn = (x ** 2).mean() + 1e-10, (noise ** 2).mean() + 1e-10
    return x + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def perturb(x, babble_paths, seed):
    rng = np.random.default_rng(seed)
    speed = rng.choice([0.9, 1.1])
    x = AF.resample(torch.from_numpy(x), int(SR * speed), SR).numpy() if speed != 1.0 else x
    if rng.random() < 0.3:
        x = np.convolve(x, _rir(rng))[: len(x)]
    if rng.random() < 0.5 and babble_paths:
        noise = np.zeros_like(x)
        for p in rng.choice(babble_paths, size=int(rng.integers(3, 6))):
            b, _ = sf.read(p, dtype="float32")
            b = np.resize(b, len(x)) if len(b) else np.zeros_like(x)
            noise += b / (np.abs(b).max() + 1e-6)
    else:
        noise = _colored(len(x), rng.choice(["white", "pink", "brown"]), rng)
    x = _at_snr(x, noise, rng.uniform(5, 20)) * 10 ** (rng.uniform(-6, 3) / 20)
    peak = np.abs(x).max()
    return (x / peak * 0.95 if peak > 1 else x).astype(np.float32)


def _gaps(x, rng, n, lo_ms, hi_ms):
    """n transmission dropouts: lo..hi ms of silence, 2 ms fades so the cut is not a click."""
    x = x.copy()
    f = int(0.002 * SR)
    ramp = np.linspace(1, 0, f, dtype=np.float32)
    for _ in range(n):
        w = int(rng.uniform(lo_ms, hi_ms) / 1000 * SR)
        if len(x) <= w + 2 * f:
            break
        i = int(rng.integers(f, len(x) - w - f))
        x[i - f:i] *= ramp
        x[i:i + w] = 0
        x[i + w:i + w + f] *= ramp[::-1]
    return x


def phone(x, seed):
    """Telephone channel: 300-3400 Hz (8 kHz round trip + biquads), 8-bit mu-law codec noise, faint line hiss,
    an occasional lost packet, gain."""
    rng = np.random.default_rng(seed)
    x = _at_snr(x, _colored(len(x), "white", rng), rng.uniform(25, 35))     # line hiss, band-limited with the speech
    t = AF.resample(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), SR, 8000)   # the line runs at 8 kHz
    t = AF.lowpass_biquad(AF.highpass_biquad(t, 8000, 300.0), 8000, 3400.0)
    t = t / (t.abs().max() + 1e-6) * 0.9
    t = AF.mu_law_decoding(AF.mu_law_encoding(t, 256), 256)                       # G.711 8-bit codec, at 8 kHz
    y = AF.resample(t, 8000, SR).numpy()[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    if rng.random() < 0.5:
        y = _gaps(y, rng, int(rng.integers(1, 3)), 20, 60)
    y = y * 10 ** (rng.uniform(-6, 0) / 20)
    peak = np.abs(y).max()
    return (y / peak * 0.95 if peak > 1 else y).astype(np.float32)


def room(x, babble_paths, seed):
    """Far-field + noise + dropout: a long weak-direct-path room response, babble or coloured noise at 0-15 dB SNR,
    2-6 transmission dropouts (30-150 ms) per 10 s."""
    rng = np.random.default_rng(seed)
    t60 = rng.uniform(0.4, 0.9)
    n = int(t60 * SR)
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
    h[0] = rng.uniform(0.3, 0.6) * np.abs(h).max() * 4    # distant talker: direct sound weak vs the reverb tail
    y = np.convolve(x, h / np.abs(h).sum() * 4)[: len(x)].astype(np.float32)
    if babble_paths and rng.random() < 0.5:
        noise = np.zeros_like(y)
        for p in rng.choice(babble_paths, size=int(rng.integers(3, 7))):
            b, _ = sf.read(p, dtype="float32")
            noise += (np.resize(b, len(y)) if len(b) else np.zeros_like(y)) / (np.abs(b).max() + 1e-6 if len(b) else 1)
    else:
        noise = _colored(len(y), rng.choice(["white", "pink", "brown"]), rng)
    y = _at_snr(y, noise, rng.uniform(0, 15))
    y = _gaps(y, rng, int(rng.integers(2, 7) * max(len(y) / SR / 10, 0.3)), 30, 150)
    peak = np.abs(y).max()
    return (y / peak * 0.95 if peak > 1 else y).astype(np.float32)


def make_copy(row, k, out_dir, babble_paths):
    """Copy k (>= 1) of a row -> new row pointing at the perturbed audio."""
    x, _ = sf.read(row["audio_filepath"], dtype="float32")
    seed = zlib.crc32(f"{row['audio_filepath']}|{k}".encode())   # reproducible across processes
    y = perturb(x, babble_paths, seed)
    name = f"{os.path.splitext(os.path.basename(row['audio_filepath']))[0]}_{row['source']}_aug{k}.flac"
    p = os.path.join(out_dir, name)
    sf.write(p, y, SR)
    return {**row, "audio_filepath": p, "duration": round(len(y) / SR, 3), "aug": k}


if __name__ == "__main__":
    x = np.sin(np.arange(SR * 2) / 10).astype(np.float32) * 0.3
    y = perturb(x, [], 1)
    assert abs(len(y) / len(x) - 1) > 0.05 and np.isfinite(y).all() and np.abs(y).max() <= 1.0
    snr = 10 * np.log10((x ** 2).mean() / ((_at_snr(x, _colored(len(x), "pink", np.random.default_rng(0)), 10) - x) ** 2).mean())
    assert abs(snr - 10) < 0.1
    p = phone(np.sin(2 * np.pi * 1000 * np.arange(SR * 2) / SR).astype(np.float32) * 0.3, 2)
    spec = np.abs(np.fft.rfft(p)) ** 2
    f = np.fft.rfftfreq(len(p), 1 / SR)
    assert len(p) == len(x) and spec[f > 4000].sum() < 1e-3 * spec.sum()      # nothing above the phone band
    r = room(x, [], 3)
    assert len(r) == len(x) and np.isfinite(r).all() and np.abs(r).max() <= 1.0 and (r == 0).sum() > SR * 0.03
    print("augment ok")
