"""Verbatim fine-tune set (user, Oct 12 2026): the en3 sources whose transcripts keep disfluencies (fillers, repeats,
backchannels -- measured: AMI / NOTSOFAR / Earnings22 / phone 12-18 repeats and 16-54 fillers per 1k words vs 0-2 for
the rest) plus Svarah (Indian English), each with an acoustic copy. Spotify is excluded (it holds Spanish rows).

    source         copies
    earnings22     orig + 1 aug: 50% far field + noise (ffn), 50% echo + reverb + obstruction (ero)
    multispk_ami   orig + 1 aug: same 50 / 50 (en3's AMI = multi-speaker meeting windows)
    notsofar       orig + 1 aug: reverb + obstruction (ro; already far field)
    phone          orig + 1 aug: far field + noise (ffn)
    svarah         its 3 existing copies (svarah, svarah_phone, svarah_room)

Rows come from en3/train.jsonl (validation sentences already removed). Augmented rows get source <src>_<kind> so
--source_weight can address them. Effects follow Voices-in-the-Wild's categories (its simulation code is not
published): synthetic RIRs, babble from LibriSpeech / TED-LIUM rows or coloured noise, fixed per row (crc32 seed).

    python voice/asr/build_vft.py [--en3 /home/marimo/work/asr/en3] [--out /home/marimo/work/asr/vft] [--workers 18]
"""
import argparse
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import SR, _at_snr, _colored  # noqa: E402

torch.set_num_threads(1)
PLAN = {"earnings22": "ffn|ero", "multispk_ami": "ffn|ero", "notsofar": "ro", "phone": "ffn"}
KEEP = ("svarah", "svarah_phone", "svarah_room")
BABBLE = []


def _rir(rng, t60_lo, t60_hi, direct_lo, direct_hi):
    t60 = rng.uniform(t60_lo, t60_hi)
    n = int(t60 * SR)
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
    h[0] = rng.uniform(direct_lo, direct_hi) * np.abs(h).max() * 4      # direct sound vs reverb tail
    return h / np.abs(h).sum() * 4


def _noise(n, rng):
    if BABBLE and rng.random() < 0.5:
        out = np.zeros(n, np.float32)
        for p in rng.choice(BABBLE, size=int(rng.integers(3, 7))):
            b, _ = sf.read(p, dtype="float32")
            if len(b):
                out += np.resize(b, n) / (np.abs(b).max() + 1e-6)
        return out
    return _colored(n, rng.choice(["white", "pink", "brown"]), rng)


def _obstruct(x, rng):
    """Talker behind an object / mask / laptop lid: low-pass 800-2500 Hz, 3-10 dB quieter."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    t = AF.lowpass_biquad(t, SR, float(rng.uniform(800, 2500)))
    if rng.random() < 0.5:                                        # second pole: a thicker obstruction
        t = AF.lowpass_biquad(t, SR, float(rng.uniform(1500, 3500)))
    return t.numpy() * 10 ** (-rng.uniform(3, 10) / 20)


def _echo(x, rng):
    """1-2 discrete reflections (speakerphone / hard wall): 80-350 ms, -4..-12 dB, each later one weaker."""
    y = x.copy()
    g = 1.0
    for _ in range(int(rng.integers(1, 3))):
        d = int(rng.uniform(0.08, 0.35) * SR)
        g *= 10 ** (-rng.uniform(4, 12) / 20)
        if d < len(x):
            y[d:] += g * x[: len(x) - d]
    return y


def ffn(x, rng):                                                   # far field + noise
    y = fftconvolve(x, _rir(rng, 0.4, 0.9, 0.3, 0.6))[: len(x)].astype(np.float32)
    return _at_snr(y, _noise(len(y), rng), rng.uniform(0, 15))


def ero(x, rng):                                                   # echo + reverb + obstruction
    y = _obstruct(x, rng)
    y = fftconvolve(y, _rir(rng, 0.3, 0.8, 0.5, 1.0))[: len(x)].astype(np.float32)
    return _echo(y, rng)


def ro(x, rng):                                                    # reverb + obstruction (NOTSOFAR: already far)
    y = _obstruct(x, rng)
    return fftconvolve(y, _rir(rng, 0.2, 0.6, 0.5, 1.0))[: len(x)].astype(np.float32)


FX = {"ffn": ffn, "ero": ero, "ro": ro}


def _init(babble):
    BABBLE[:] = babble


def _job(a):
    row, kinds, out = a
    rng = np.random.default_rng(zlib.crc32(row["audio_filepath"].encode()))
    kind = kinds[0] if len(kinds) == 1 else kinds[int(rng.random() < 0.5)]
    x, sr = sf.read(row["audio_filepath"], dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    assert sr == SR, (row["audio_filepath"], sr)
    y = FX[kind](x, rng)
    y = y * np.sqrt(((x ** 2).mean() + 1e-10) / ((y ** 2).mean() + 1e-10))   # the RIR normalisation drops the level
    y = y * 10 ** (rng.uniform(-6, 0) / 20)                                 # 20-70x: back to the original loudness
    peak = np.abs(y).max()
    y = (y / peak * 0.95 if peak > 1 else y).astype(np.float32)
    d = os.path.join(out, "audio", f"{row['source']}_{kind}")
    p = os.path.join(d, os.path.splitext(os.path.basename(row["audio_filepath"]))[0] + f"_{kind}.flac")
    sf.write(p, y, SR)
    return {**row, "audio_filepath": p, "duration": round(len(y) / SR, 3), "source": f"{row['source']}_{kind}",
            "aug": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en3", default="/home/marimo/work/asr/en3")
    ap.add_argument("--out", default="/home/marimo/work/asr/vft")
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.en3, "train.jsonl"), encoding="utf-8")]
    babble = [r["audio_filepath"] for r in rows if r["source"] in ("librispeech", "tedlium")][::20]
    keep = [r for r in rows if r["source"] in PLAN or r["source"] in KEEP]
    jobs = [(r, PLAN[r["source"]].split("|"), a.out) for r in keep if r["source"] in PLAN]
    for s in PLAN:
        for k in PLAN[s].split("|"):
            os.makedirs(os.path.join(a.out, "audio", f"{s}_{k}"), exist_ok=True)
    print(f"[vft] {len(keep)} original rows kept, {len(jobs)} augmented copies to write, babble pool {len(babble)}",
          flush=True)
    aug = []
    with Pool(a.workers, initializer=_init, initargs=(babble,)) as p:
        for i, r in enumerate(p.imap(_job, jobs, chunksize=16)):
            aug.append(r)
            if (i + 1) % 10000 == 0:
                print(f"[vft] {i + 1}/{len(jobs)}", flush=True)
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in keep + aug:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = {}
    for r in keep + aug:
        h[r["source"]] = h.get(r["source"], 0) + r["duration"] / 3600
    for s in sorted(h):
        print(f"[vft] {s:24s} {h[s]:7.1f} h", flush=True)
    print(f"[vft] total {sum(h.values()):.1f} h, {len(keep) + len(aug)} rows -> {a.out}/train.jsonl", flush=True)
    print("VFT_DONE", flush=True)


if __name__ == "__main__":
    main()
