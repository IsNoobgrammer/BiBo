"""What is in each candidate Hindi / code-switch / diarization source? Reads metadata + a sample (never the whole set).

    python voice/asr/profile_sources.py > profile.log

Per source: shard count and bytes, row count of the first shard, columns, audio duration from a 200-row sample
(-> estimated total hours), category columns (scenario, speaker, ...), share of rows with Latin letters / Devanagari,
and 3 example transcripts. A source that fails is reported and skipped.
"""
import collections
import io
import re

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem

CONV = "refs/convert/parquet"
# name, repo, revision, path prefix of the parquet shards
SOURCES = [
    ("indicvoices_hi", "ai4bharat/IndicVoices", "main", "hindi/train-"),
    ("vaani_hi", "ARTPARK-IISc/Vaani", "main", "audio/Hindi/train-"),
    ("shrutilipi_hi", "amithm3/shrutilipi", CONV, "hi/train/"),
    ("indictts_hi", "SPRINGLab/IndicTTS-Hindi", CONV, "default/train/"),
    ("cv17_hi", "fsicoli/common_voice_17_0", CONV, "hi/train/"),
    ("fleurs_hi", "google/fleurs", CONV, "hi_in/train/"),
    ("diarbench_hi", "sarvamai/indic-diarbench", "main", "Hindi/"),
    ("codemixed_new", "RidheshBhati/Codemixed_New", "main", ""),
    ("hinglish_ayushi", "agarwalayushi/hinglish", "main", ""),
    ("kaira_elise_hi", "projectkaira/Pretraining-V1", "main", "elise_hindi/"),
    ("svq", "google/svq", "main", ""),
]
CAT_HINT = re.compile(r"scenario|task|gender|age|area|state|district|speaker|lang|accent|domain|source|label|split", re.I)
TEXT_HINT = re.compile(r"text|transcri|sentence|normalized|verbatim|caption|utterance", re.I)


def audio_col(schema):
    for f in schema:
        if "bytes" in str(f.type) or f.name in ("audio", "audio_filepath"):
            return f.name
    return None


def profile(name, repo, rev, prefix):
    api, fs = HfApi(), HfFileSystem()
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=rev)
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        print(f"== {name}: no parquet under '{prefix}' ({repo}@{rev})")
        return
    rev_q = rev.replace("/", "%2F")
    sizes = [fs.info(f"datasets/{repo}@{rev_q}/{f}")["size"] for f in files[:50]]
    pf = pq.ParquetFile(fs.open(f"datasets/{repo}@{rev_q}/{files[0]}"))
    schema = pf.schema_arrow
    acol = audio_col(schema)
    meta_cols = [f.name for f in schema if f.name != acol and "binary" not in str(f.type)]
    t = pf.read_row_group(0, columns=meta_cols + ([acol] if acol else [])).slice(0, 200).to_pylist()
    durs = []
    for r in t:
        a = r.get(acol)
        if isinstance(a, dict) and a.get("bytes"):
            try:
                x, sr = sf.read(io.BytesIO(a["bytes"]))
                durs.append(len(x) / sr)
            except Exception:
                pass
    rows0 = pf.metadata.num_rows
    mean_d = sum(durs) / len(durs) if durs else 0
    est_rows = rows0 / max(sizes[0], 1) * sum(sizes) * (len(files) / len(sizes))
    print(f"== {name} ({repo}): {len(files)} shards, ~{sum(sizes) / len(sizes) * len(files) / 1e9:.1f} GB, "
          f"{rows0} rows in shard 0, mean dur {mean_d:.1f}s -> est {est_rows * mean_d / 3600:.0f} h")
    print("   columns:", meta_cols, "audio:", acol)
    for c in meta_cols:
        vals = [r.get(c) for r in t if isinstance(r.get(c), (str, int, bool))]
        if vals and CAT_HINT.search(c) and len(set(vals)) < 60:
            print(f"   {c}: {collections.Counter(vals).most_common(8)}")
    tc = next((c for c in meta_cols if TEXT_HINT.search(c)), None)
    if tc:
        texts = [str(r.get(tc) or "") for r in t]
        lat = sum(bool(re.search("[A-Za-z]{2,}", s)) for s in texts) / max(len(texts), 1)
        dev = sum(bool(re.search("[ऀ-ॿ]", s)) for s in texts) / max(len(texts), 1)
        print(f"   text col '{tc}': Latin-word rows {100 * lat:.0f}%, Devanagari rows {100 * dev:.0f}%")
        for s in texts[:3]:
            print("     >", s[:140])


if __name__ == "__main__":
    for s in SOURCES:
        try:
            profile(*s)
        except Exception as e:
            print(f"== {s[0]}: FAILED {type(e).__name__}: {str(e)[:150]}")
