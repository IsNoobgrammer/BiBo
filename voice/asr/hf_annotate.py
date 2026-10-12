"""Publish the teacher transcripts + cleaning flags as a sidecar per HF dataset repo (no audio re-upload).

For each repo, annotations/<name>.parquet: one row per clip of that repo found in the transcribed manifest, keyed like
the repo's own parquet / manifests ("source" + "file" = the audio file name):
  parakeet_transcription   stock Parakeet TDT 0.6B v2, full context, greedy (teacher_transcribe.py)
  parakeet_wer             WER % of the reference against it (score.py normalize_v2)
  missing_speech           teacher hears >= 1.5x and >= 5 more words than the reference (clean_by_teacher.py)
  garbage                  WER > 60 % with >= 4 reference words
Clips built locally (not in any repo, e.g. prep_en's AMI multi-speaker windows) go to --derived_repo under
annotations/<name>_derived.parquet with their full relative audio path.

    python voice/asr/hf_annotate.py --manifest en3/train.jsonl --hyp en3/teacher_pk --hf_mix asr/hf_mix \
        --repos fhai50032/asr-english fhai50032/asr-english-v2 fhai50032/asr-english-nc [--upload]
"""
import argparse
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_by_teacher import wer_counts  # noqa: E402
from score import normalize_v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--hf_mix", required=True, help="dir holding one snapshot folder per repo (fetch_mix.py)")
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--derived_repo", default=None, help="default: the first --repos")
    ap.add_argument("--name", default="parakeet_tdt_0.6b_v2")
    ap.add_argument("--out", default="hf_annotations")
    ap.add_argument("--ratio", type=float, default=1.5)
    ap.add_argument("--extra", type=int, default=5)
    ap.add_argument("--max_wer", type=float, default=0.6)
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    owner = {}                                                  # "<source>/<file>" -> repo
    for repo in a.repos:
        dirs = [d for d in glob.glob(os.path.join(a.hf_mix, "*")) if os.path.basename(d).replace("___", "/").endswith(repo.split("/")[-1])]
        mans = [m for d in dirs for m in glob.glob(os.path.join(d, "**", "manifests", "*.jsonl"), recursive=True)]
        n = 0
        for m in mans:
            for l in open(m, encoding="utf-8"):
                r = json.loads(l)
                owner.setdefault(f'{r["source"]}/{os.path.basename(r["audio_filepath"])}', repo)
                n += 1
        print(f"[annotate] {repo}: {len(mans)} manifests, {n} rows", flush=True)
    hyp = {}
    for p in glob.glob(os.path.join(a.hyp, "hyp_*.tsv")):
        for l in open(p, encoding="utf-8"):
            i, _, t = l.rstrip("\n").partition("\t")
            hyp[int(i)] = t
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    cols = {}
    derived = a.derived_repo or a.repos[0]
    for i, r in enumerate(rows):
        if i not in hyp:
            continue
        key = f'{r.get("source")}/{os.path.basename(r["audio_filepath"])}'
        repo = owner.get(key)
        ref, h = normalize_v2(r["text"]).split(), normalize_v2(hyp[i]).split()
        e, n = wer_counts(ref, h)
        missing = len(h) >= a.ratio * max(len(ref), 1) and len(h) - len(ref) >= a.extra
        garbage = not missing and n >= a.min_words and e / max(n, 1) > a.max_wer
        tgt = (repo, a.name) if repo else (derived, a.name + "_derived")
        c = cols.setdefault(tgt, {"source": [], "file": [], "parakeet_transcription": [], "parakeet_wer": [],
                                  "missing_speech": [], "garbage": []})
        c["source"].append(r.get("source"))
        c["file"].append(os.path.basename(r["audio_filepath"]) if repo else os.path.relpath(r["audio_filepath"], "/home/marimo/work/asr"))
        c["parakeet_transcription"].append(hyp[i])
        c["parakeet_wer"].append(round(100.0 * e / max(n, 1), 2))
        c["missing_speech"].append(bool(missing))
        c["garbage"].append(bool(garbage))
    os.makedirs(a.out, exist_ok=True)
    api = None
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
    for (repo, name), c in sorted(cols.items()):
        path = os.path.join(a.out, repo.replace("/", "__") + f"__{name}.parquet")
        pq.write_table(pa.table(c), path, compression="zstd")
        nm, ng = sum(c["missing_speech"]), sum(c["garbage"])
        print(f"[annotate] {repo} annotations/{name}.parquet: {len(c['file'])} rows, missing_speech {nm}, garbage {ng} -> {path}",
              flush=True)
        if api:
            api.upload_file(path_or_fileobj=path, path_in_repo=f"annotations/{name}.parquet", repo_id=repo,
                            repo_type="dataset", commit_message=f"annotations/{name}.parquet: stock Parakeet TDT 0.6B v2 "
                            f"transcripts + WER + missing_speech / garbage flags ({len(c['file'])} clips)")
            print(f"[annotate] uploaded -> {repo}", flush=True)
    print("ANNOTATE_DONE", flush=True)


if __name__ == "__main__":
    main()
