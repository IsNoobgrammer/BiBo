"""On-the-fly acoustic augmentation for train_asr.py --aug_schedule: each epoch a fraction p(epoch) of the utterances
is re-rendered (fresh every epoch, nothing stored): speed 0.9 / 1.0 / 1.1, synthetic room reverb (30 %), noise at
5-20 dB SNR (half babble = 2-4 OTHER utterances of the same batch, half white / pink / brown), a laptop-mic band
limit (50 %: high-pass 100-300 Hz, low-pass 3.4-7.5 kHz) and gain -6..+3 dB. The effects are augment.py's.

Hook: the dataset's load_audio_with_cuts (padded audio + lengths + cuts, before the transcripts are attached), so
nothing downstream changes. Seeded by (cut id, epoch): same seed -> same batches, a new rendering every epoch.
p is set in the MAIN process at each epoch start (STATE, by train_asr's EpochShuffle); the loader workers are forked
when the epoch's iterator is created, so they inherit it.
"""
import zlib

import numpy as np
import torch

STATE = {"p": 0.0, "epoch": 0}


def _mic(x, rng):
    import torchaudio.functional as AF
    t = torch.from_numpy(x)
    t = AF.highpass_biquad(t, 16000, float(rng.uniform(100, 300)))
    t = AF.lowpass_biquad(t, 16000, float(rng.uniform(3400, 7500)))
    return t.numpy()


def augment_one(x, others, rng):
    import augment as A
    import torchaudio.functional as AF
    speed = rng.choice([0.9, 1.0, 1.1])
    if speed != 1.0:
        x = AF.resample(torch.from_numpy(x), int(16000 * speed), 16000).numpy()
    if rng.random() < 0.3:
        x = np.convolve(x, A._rir(rng))[: len(x)]
    if rng.random() < 0.5 and len(others) >= 2:
        noise = np.zeros_like(x)
        for j in rng.choice(len(others), size=min(len(others), int(rng.integers(2, 5))), replace=False):
            b = others[j]
            b = np.resize(b, len(x)) if len(b) else np.zeros_like(x)
            noise += b / (np.abs(b).max() + 1e-6)
    else:
        noise = A._colored(len(x), rng.choice(["white", "pink", "brown"]), rng)
    x = A._at_snr(x, noise, rng.uniform(5, 20))
    if rng.random() < 0.5:
        x = _mic(x.astype(np.float32), rng)
    x = x * 10 ** (rng.uniform(-6, 3) / 20)
    peak = np.abs(x).max()
    return (x / peak * 0.95 if peak > 1 else x).astype(np.float32)


def wrap(dataset):
    orig = dataset.load_audio_with_cuts

    def load_audio_with_cuts(cuts):
        audio, lens, cuts = orig(cuts)
        p, ep = STATE["p"], STATE["epoch"]
        if p <= 0:
            return audio, lens, cuts
        torch.set_num_threads(1)                           # inside a loader worker
        xs = [audio[i, : int(lens[i])].numpy().copy() for i in range(len(lens))]
        out = list(xs)
        for i, c in enumerate(cuts):
            rng = np.random.default_rng(zlib.crc32(f"{c.id}|{ep}".encode()))
            if rng.random() < p:
                out[i] = augment_one(xs[i], [xs[j] for j in range(len(xs)) if j != i], rng)
        lens = torch.tensor([len(x) for x in out], dtype=lens.dtype)
        audio = torch.zeros(len(out), int(lens.max()), dtype=audio.dtype)
        for i, x in enumerate(out):
            audio[i, : len(x)] = torch.from_numpy(x)
        return audio, lens, cuts

    dataset.load_audio_with_cuts = load_audio_with_cuts


def p_for(schedule, epoch):
    return schedule[min(epoch, len(schedule) - 1)]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rng = np.random.default_rng(0)
    x = (np.sin(np.arange(16000 * 2) / 10) * 0.3).astype(np.float32)
    others = [rng.standard_normal(16000).astype(np.float32) * 0.1 for _ in range(4)]
    a = augment_one(x, others, np.random.default_rng(5))
    b = augment_one(x, others, np.random.default_rng(5))
    assert np.array_equal(a, b) and np.isfinite(a).all() and np.abs(a).max() <= 1.0
    assert p_for([0, 0.5, 0.5, 0.5, 0.2], 0) == 0 and p_for([0, 0.5, 0.5, 0.5, 0.2], 9) == 0.2
    print("aug_online ok", len(x), "->", len(a))
