"""Stored 3x of a small source: the original + <src>_phone (augment.phone: telephone channel) + <src>_room
(augment.room: far-field + noise + dropout), each a build_mix-style source of its own (manifest + 16 kHz FLAC) so it
is pushed / restored / weighted like any other. Speaker ids keep the original id (prep_en.SAME_SPEAKERS puts all
three in one val speaker set, so no voice is in both val and train).

    python voice/asr/aug_sources.py --mix /home/marimo/work/asr/mix_en --sources medical spotify svarah
"""
import argparse
import json
import os
import random
import sys
import zlib
from functools import partial
from multiprocessing import Pool

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment  # noqa: E402

SR = 16000


def one(row, kind, out_dir, babble):
    x, _ = sf.read(row["audio_filepath"], dtype="float32")
    seed = zlib.crc32(f"{row['audio_filepath']}|{kind}".encode())
    y = augment.phone(x, seed) if kind == "phone" else augment.room(x, babble, seed)
    p = os.path.join(out_dir, os.path.basename(row["audio_filepath"]))
    sf.write(p, y, SR)
    src = f"{row['source']}_{kind}"
    return {**row, "audio_filepath": p, "duration": round(len(y) / SR, 3), "source": src,
            "speaker": f"{src}:{str(row['speaker']).split(':', 1)[-1]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for s in a.sources:
        rows = [json.loads(l) for l in open(os.path.join(a.mix, f"{s}.jsonl"), encoding="utf-8")]
        babble = [r["audio_filepath"] for r in random.Random(23).sample(rows, min(500, len(rows)))]
        for kind in ("phone", "room"):
            out_dir = os.path.join(a.mix, "audio", f"{s}_{kind}")
            os.makedirs(out_dir, exist_ok=True)
            with Pool(a.workers) as p:
                out = p.map(partial(one, kind=kind, out_dir=out_dir, babble=babble), rows, chunksize=32)
            with open(os.path.join(a.mix, f"{s}_{kind}.jsonl"), "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
            print(f"AUG {s}_{kind}: {len(out)} rows {sum(r['duration'] for r in out) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
