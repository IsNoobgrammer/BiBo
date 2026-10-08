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
    print("augment ok")
