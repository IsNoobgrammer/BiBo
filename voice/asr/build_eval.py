"""Public eval sets for rough comparison with published models: FLEURS en_us + hi_in test, same text convention as
training (prep_train.clean: lowercase, no punctuation, Latin English / Devanagari Hindi).

    python voice/asr/build_eval.py --out /home/marimo/work/asr/eval_sets

Writes <out>/fleurs_en.jsonl, <out>/fleurs_hi.jsonl (+ audio). The manifest file name becomes the W&B metric prefix
(fleurs_en_val_wer ...). FLEURS is read speech; it is a sanity anchor, the speaker-held-out val splits and the
internal meeting clips are the real targets.
"""
import argparse
import io
import json
import os
import sys

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_train import clean  # noqa: E402

SETS = {"fleurs_en": ("en_us", "en"), "fleurs_hi": ("hi_in", "hi")}
REV = "refs/convert/parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for name, (cfg, lang) in SETS.items():
        files = [f for f in HfApi().list_repo_files("google/fleurs", repo_type="dataset", revision=REV)
                 if f.startswith(f"{cfg}/test/") and f.endswith(".parquet")]
        adir = os.path.join(a.out, "audio", name)
        os.makedirs(adir, exist_ok=True)
        rows = []
        for f in files:
            for r in pq.read_table(hf_hub_download("google/fleurs", f, repo_type="dataset", revision=REV),
                                   columns=["audio", "transcription"]).to_pylist():
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                text = clean(r["transcription"] or "")
                if sr != 16000 or not text:
                    continue
                p = os.path.join(adir, f"{len(rows):05d}.flac")
                sf.write(p, x, sr)
                rows.append({"audio_filepath": p, "duration": round(len(x) / sr, 3), "text": text, "lang": lang,
                             "source": name})
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as fo:
            fo.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"{name}: {len(rows)} utts, {sum(r['duration'] for r in rows) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
