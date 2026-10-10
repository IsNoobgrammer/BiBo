"""(Re)write the README of a push_src.py repo from the manifests in it: one viewer config per source + `all`, an
hours table, how each source was cut, and every source's origin and license (CC-BY / CC-BY-SA need the attribution).

    python voice/asr/hf_card.py --repo fhai50032/asr-english-v2
"""
import argparse
import json
import os

from huggingface_hub import HfApi, hf_hub_download

INFO = {   # source -> (origin, license, what / how it was cut)
    "ami_ihm": ("edinburghcstr/ami (ihm)", "CC-BY-4.0", "AMI meetings, close-talk headsets; utterances >= 0.2 s (meeting windows built at training-prep time)"),
    "ami_sdm": ("edinburghcstr/ami (sdm)", "CC-BY-4.0", "AMI meetings, single distant mic (far-field); utterances >= 0.2 s"),
    "librispeech": ("openslr/librispeech_asr (train.clean.100)", "CC-BY-4.0", "read audiobooks"),
    "earnings22": ("anton-l/earnings22_baseline_5_gram (Earnings-22)", "CC-BY-SA-4.0", "earnings calls, many accents; rows with <inaudible>/<crosstalk> dropped"),
    "notsofar": ("microsoft/NOTSOFAR (train 240825.1)", "CC-BY-4.0", "real office meetings, every far-field device (ch0); <= 25 s windows cut between words of all speakers, words in start order"),
    "spgi2": ("kensho/SPGISpeech2.0", "Kensho terms: non-commercial research only", "earnings calls; snippets cut at word alignment into single-speaker pieces <= 25 s"),
    "tedlium": ("kfajdsl/tedlium (TED-LIUM release 3, legacy train)", "CC-BY-NC-ND-3.0", "TED talks; segments containing <unk> dropped"),
}
VITW = ("zhifeixie/Voices-in-the-Wild-2M", "Apache-2.0", "LibriSpeech / Common Voice sentences through simulated acoustics ({}); English rows only, resampled 24 -> 16 kHz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    api = HfApi()
    files = api.list_repo_files(a.repo, repo_type="dataset")
    srcs = sorted(f[len("manifests/"):-len(".jsonl")] for f in files if f.startswith("manifests/") and f.endswith(".jsonl"))
    rows = {}
    for s in srcs:
        man = [json.loads(l) for l in open(hf_hub_download(a.repo, f"manifests/{s}.jsonl", repo_type="dataset"), encoding="utf-8")]
        rows[s] = (len(man), sum(r["duration"] for r in man) / 3600)
    info = lambda s: INFO.get(s) or ((VITW[0], VITW[1], VITW[2].format(s[5:].replace("_", " "))) if s.startswith("vitw_") else ("?", "?", ""))
    cfg = "\n".join([f"- config_name: all\n  data_files:\n  - split: train\n    path: data/*/*.parquet"] +
                    [f"- config_name: {s}\n  data_files:\n  - split: train\n    path: data/{s}/*.parquet" for s in srcs])
    table = "\n".join(f"| {s} | {rows[s][1]:.1f} | {rows[s][0]:,} | {info(s)[2]} | {info(s)[0]} | {info(s)[1]} |" for s in srcs)
    total = sum(h for _, h in rows.values())
    card = f"""---
language: [en]
task_categories: [automatic-speech-recognition]
license: other
configs:
{cfg}
---
# {a.repo.split('/')[-1]}

English ASR training audio for the BiBo voice model (16 kHz mono FLAC), built by BiBo `voice/asr/build_mix.py` and
pushed source by source with `voice/asr/push_src.py`. **{total:.1f} h** in {len(srcs)} sources; config `all` = every
source, one config per source.

Columns: `audio`, `text` (the source's own transcript, original casing; training normalisation is in
`voice/asr/prep_en.py`), `lang`, `source`, `speaker` (speaker / call / talk / meeting id, used only to keep
validation speaker-disjoint), `scenario`, `meeting`, `begin`, `duration`. `manifests/<source>.jsonl` holds the build
rows (restore with `voice/asr/fetch_mix.py`).

| source | hours | rows | content / cut | origin | license |
|---|---|---|---|---|---|
{table}

Each row keeps its origin's license; attribution is to the datasets above. Plain ASR text: no speaker tags.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset",
                    commit_message="dataset card: per-source configs, hours, origins, licenses")
    print(f"CARD {a.repo}: {len(srcs)} sources, {total:.1f} h", flush=True)


if __name__ == "__main__":
    main()
