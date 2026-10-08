"""Multi-speaker training windows with speaker-turn tokens: "<spk1> text <spk2> text <spk1> text".

Speakers are numbered by FIRST APPEARANCE in the window (the model learns "same voice / new voice", not identity), max 4.
Kinds (all from TRAIN rows only, texts already cleaned), hours per kind from `budgets` (None = as many as exist):
  ami  -- real meetings: consecutive AMI utterances of one meeting at their real start times (real turns + overlap;
          ihm = close-talk mics summed, so cross-talk bleed is real too). EVERY speaker's words are labelled.
  hi   -- simulated Hindi conversations: 2-4 different speakers, 0.1-0.8 s gaps, 10% of turns overlap by <= 0.4 s
  cs   -- simulated code-switched conversations: the same, alternating a Hindi and an English speaker
  bc   -- backchannels: speaker A says two utterances; a DIFFERENT speaker's short reply ("okay", "yeah", "haan", "जी")
          lands at the end of A's first one, 6-15 dB quieter and overlapping, and is labelled with its own token:
          "<spk1> A1 <spk2> okay <spk1> A2". run1 dropped exactly these words on the meeting clip.
Windows are capped at MAX_S. Written as <out>/<n>.flac + rows with lang "mix".
"""
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

MAX_S, SR = 25.0, 16000
TOK = ["<spk1>", "<spk2>", "<spk3>", "<spk4>"]
BACKCHANNELS = {"okay", "ok", "yeah", "yes", "yep", "right", "sure", "hmm", "mhm", "uh huh", "exactly", "correct",
                "haan", "han", "achha", "accha", "theek hai", "हाँ", "हां", "जी", "हाँ जी", "जी हाँ", "अच्छा", "ठीक है",
                "हम्म", "सही", "बिल्कुल"}


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
    """pieces: [(offset_s, row) or (offset_s, row, gain)] -> mixed 16 kHz waveform written to path; returns duration."""
    end = max(p[0] + p[1]["duration"] for p in pieces)
    y = np.zeros(int(end * SR) + 1, np.float32)
    for p in pieces:
        o, r, g = p[0], p[1], (p[2] if len(p) > 2 else 1.0)
        x, _ = sf.read(r["audio_filepath"], dtype="float32")
        i = int(o * SR)
        y[i:i + len(x)] += g * x[: len(y) - i]
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


def _bc_window(main_pool, bcs, rng):
    """A1 + (other speaker's short reply, quieter, overlapping A1's end) + A2."""
    name = rng.choice(list(main_pool))
    if len(main_pool[name]) < 2:
        return []
    a1, a2 = rng.sample(main_pool[name], 2)
    bc = rng.choice(bcs)
    if bc["speaker"] == a1["speaker"] or a1["duration"] + a2["duration"] + bc["duration"] > MAX_S:
        return []
    t_bc = max(0.0, a1["duration"] - rng.uniform(0.0, 0.4))
    t_a2 = t_bc + bc["duration"] + rng.uniform(-0.2, 0.4)
    gain = 10 ** (-rng.uniform(6, 15) / 20)
    return [(0.0, a1), (t_bc, bc, gain), (max(t_a2, a1["duration"]), a2)]


def make(rows, out_dir, budgets, rng):
    os.makedirs(out_dir, exist_ok=True)
    by_lang = {"hi": defaultdict(list), "en": defaultdict(list)}
    for r in rows:
        if r["duration"] <= 12 and r["source"] != "ami_ihm":
            by_lang[r["lang"]][r["speaker"]].append(r)
    bcs = [r for r in rows if r["duration"] <= 2.0 and r["text"] in BACKCHANNELS]
    ami = _ami_windows(rows, rng)
    main = {**by_lang["en"], **by_lang["hi"]}
    done, out, n = defaultdict(float), [], 0
    for kind in ("ami", "hi", "cs", "bc"):
        limit = float("inf") if budgets.get(kind) is None else budgets[kind] * 3600
        tries = 0
        while done[kind] < limit and tries < 200000:
            tries += 1
            if kind == "ami":
                if not ami:
                    break
                pieces = ami.pop()
            elif kind == "bc":
                if not bcs:
                    break
                pieces = _bc_window(main, bcs, rng)
            else:
                pieces = _sim_window([by_lang["hi"]] if kind == "hi" else [by_lang["hi"], by_lang["en"]], rng)
            if len({p[1]["speaker"] for p in pieces}) < 2:
                continue
            path = os.path.join(out_dir, f"{n:07d}.flac")
            dur = _render(pieces, path)
            text = _text([(p[1]["speaker"], p[1]["text"]) for p in sorted(pieces, key=lambda p: p[0])])
            out.append({"audio_filepath": path, "duration": round(dur, 3), "text": text, "lang": "mix",
                        "source": f"multispk_{kind}", "speaker": "multi"})
            done[kind] += dur
            n += 1
        print(f"multispk {kind}: {done[kind] / 3600:.1f} h (budget {budgets.get(kind)})"
              + (f", {len(bcs)} backchannel clips" if kind == "bc" else ""), flush=True)
    return out


if __name__ == "__main__":
    assert _text([("a", "hi"), ("b", "yo"), ("b", "there"), ("a", "ok")]) == "<spk1> hi <spk2> yo there <spk1> ok"
    assert _text([("x", "1"), ("y", "2"), ("z", "3"), ("w", "4")]).split()[::2] == TOK
    import random
    rng = random.Random(0)
    main = {"s1": [{"audio_filepath": f"a{i}", "duration": 3.0, "speaker": "s1", "text": f"a{i}"} for i in range(3)]}
    bc = [{"audio_filepath": "b", "duration": 0.5, "speaker": "s2", "text": "okay"}]
    p = next(w for w in (_bc_window(main, bc, rng) for _ in range(20)) if w)
    assert len(p) == 3 and p[1][1]["text"] == "okay" and p[1][2] < 0.51 and p[2][0] >= p[0][1]["duration"]
    assert _text([(q[1]["speaker"], q[1]["text"]) for q in sorted(p, key=lambda q: q[0])]).count("<spk") == 3
    print("multispk ok")
