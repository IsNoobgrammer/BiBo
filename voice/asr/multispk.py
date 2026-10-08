"""Multi-speaker training windows with speaker-turn tokens: "<spk1> text <spk2> text <spk1> text".

Speakers are numbered by FIRST APPEARANCE in the window (the model learns "same voice / new voice", not identity), max 4.
Three kinds, a third of the hours each (all built from TRAIN rows only, texts already cleaned):
  ami  -- real meetings: consecutive AMI utterances of one meeting placed at their real start times (real turns and
          overlap; ihm = close-talk mics summed, so cross-talk bleed is real too)
  hi   -- simulated Hindi conversations: 2-4 different speakers, 1-2 utterances per turn, 0.1-0.8 s gaps,
          10% of turns overlap the previous one by up to 0.4 s
  cs   -- simulated code-switched conversations: the same, alternating a Hindi and an English speaker
Windows are capped at MAX_S. Written as <out>/<n>.flac + rows with lang "mix".
"""
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf

MAX_S, SR = 25.0, 16000
TOK = ["<spk1>", "<spk2>", "<spk3>", "<spk4>"]


def _text(turns):
    """turns: [(speaker, text)] in time order -> one string with a token at every speaker change."""
    ids, out, prev = {}, [], None
    for s, t in turns:
        ids.setdefault(s, len(ids))
        if s != prev:
            out.append(TOK[ids[s]])
        out.append(t)
        prev = s
    return " ".join(out)


def _render(pieces, path):
    """pieces: [(offset_s, row)] -> mixed 16 kHz waveform written to path; returns duration."""
    end = max(o + r["duration"] for o, r in pieces)
    y = np.zeros(int(end * SR) + 1, np.float32)
    for o, r in pieces:
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        i = int(o * SR)
        y[i:i + len(x)] += x[: len(y) - i]
    peak = np.abs(y).max()
    sf.write(path, y / peak * 0.9 if peak > 1 else y, SR)
    return len(y) / SR


def _ami_windows(rows, rng):
    by_m = defaultdict(list)
    for r in rows:
        if r["source"] == "ami_ihm" and r.get("meeting") is not None:
            by_m[r["meeting"]].append(r)
    wins = []
    for m in by_m.values():
        m.sort(key=lambda r: r["begin"])
        i = 0
        while i < len(m):
            j, t0 = i, m[i]["begin"]
            while j + 1 < len(m) and m[j + 1]["begin"] + m[j + 1]["duration"] - t0 <= MAX_S:
                j += 1
            win = m[i:j + 1]
            if len({r["speaker"] for r in win}) >= 2:
                wins.append([(r["begin"] - t0, r) for r in win])
            i = j + 1
    rng.shuffle(wins)
    return wins


def _sim_window(pools, rng):
    """pools: list of speaker->rows dicts to alternate between (one pool = monolingual, two = code-switch)."""
    n_spk = rng.choice([2, 2, 3, 4])
    spk = []
    for k in range(n_spk):
        pool = pools[k % len(pools)]
        spk.append((pool, rng.choice(list(pool))))
    pieces, t, last, used = [], 0.0, None, set()
    for _ in range(12):
        choices = [s for s in range(n_spk) if s != last]
        s = rng.choice(choices)
        pool, name = spk[s]
        left = [r for r in pool[name] if r["audio_filepath"] not in used]   # never repeat an utterance in a window
        if not left:
            return pieces
        for r in rng.sample(left, min(rng.choice([1, 1, 2]), len(left))):
            used.add(r["audio_filepath"])
            if t + r["duration"] > MAX_S:
                return pieces
            gap = -rng.uniform(0, 0.4) if (pieces and rng.random() < 0.1) else rng.uniform(0.1, 0.8)
            t = max(0.0, t + (gap if pieces else 0.0))
            pieces.append((t, r))
            t += r["duration"]
        last = s
    return pieces


def make(rows, out_dir, hours, rng):
    os.makedirs(out_dir, exist_ok=True)
    by_lang = {"hi": defaultdict(list), "en": defaultdict(list)}
    for r in rows:
        if r["duration"] <= 12 and r["source"] != "ami_ihm":
            by_lang[r["lang"]][r["speaker"]].append(r)
    for d in by_lang.values():                                       # need >= 2 speakers with audio
        for k in [k for k, v in d.items() if not v]:
            del d[k]
    ami = _ami_windows(rows, rng)
    budget = {"ami": hours / 3, "hi": hours / 3, "cs": hours / 3}
    done, out, n = defaultdict(float), [], 0
    for kind in ("ami", "hi", "cs"):
        while done[kind] < budget[kind] * 3600:
            if kind == "ami":
                if not ami:
                    break
                pieces = ami.pop()
            else:
                pieces = _sim_window([by_lang["hi"]] if kind == "hi" else [by_lang["hi"], by_lang["en"]], rng)
            if len({r["speaker"] for _, r in pieces}) < 2:
                continue
            path = os.path.join(out_dir, f"{n:07d}.flac")
            dur = _render(pieces, path)
            text = _text([(r["speaker"], r["text"]) for _, r in sorted(pieces, key=lambda p: p[0])])
            out.append({"audio_filepath": path, "duration": round(dur, 3), "text": text, "lang": "mix",
                        "source": f"multispk_{kind}", "speaker": "multi"})
            done[kind] += dur
            n += 1
        print(f"multispk {kind}: {done[kind] / 3600:.1f} / {budget[kind]:.1f} h", flush=True)
    return out


if __name__ == "__main__":
    assert _text([("a", "hi"), ("b", "yo"), ("b", "there"), ("a", "ok")]) == "<spk1> hi <spk2> yo there <spk1> ok"
    assert _text([("x", "1"), ("y", "2"), ("z", "3"), ("w", "4")]).split()[::2] == TOK
