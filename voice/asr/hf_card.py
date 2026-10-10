"""(Re)write the dataset card of a push_src.py repo from what is in it: viewer configs (`all`, one per source, a
`<src>_<H>h` config per core split), sources grouped by domain with hours / rows / mean clip / speakers, processing,
columns, validation notes, origins and licenses (CC-BY / CC-BY-SA need the attribution).

    python voice/asr/hf_card.py --repo fhai50032/asr-english-v2
"""
import argparse
import json

from huggingface_hub import HfApi, hf_hub_download

REPOS = {"fhai50032/asr-english": "conversational / lecture / podcast / phone base set (+ stored augmented copies)",
         "fhai50032/asr-english-v2": "meetings, far-field, calls, read speech and simulated acoustics",
         "fhai50032/asr-english-nc": "PRIVATE: non-commercial sources (SPGISpeech 2.0, TED-LIUM)"}
# source -> (domain, origin repo, license, content / how it was cut)
INFO = {
    "ami_ihm": ("Meetings", "edinburghcstr/ami (ihm)", "CC-BY-4.0", "AMI meetings, close-talk headsets; utterances >= 0.2 s (backchannels kept)"),
    "ami_sdm": ("Meetings", "edinburghcstr/ami (sdm)", "CC-BY-4.0", "the same AMI meetings through ONE distant mic (far-field); utterances >= 0.2 s"),
    "notsofar": ("Meetings", "microsoft/NOTSOFAR (train 240825.1)", "CC-BY-4.0", "real office meetings, every far-field device (sc_* / mc_* ch0); <= 25 s windows cut between the words of all speakers, overlap kept, words in start order"),
    "earnings22": ("Calls", "anton-l/earnings22_baseline_5_gram (Earnings-22)", "CC-BY-SA-4.0", "earnings calls, many accents; rows with <inaudible> / <crosstalk> dropped"),
    "spgi2": ("Calls", "kensho/SPGISpeech2.0", "Kensho terms (non-commercial research)", "earnings calls; ~75 s snippets cut at the word alignment into single-speaker pieces <= 25 s"),
    "phone": ("Calls", "sawradip/phone-asr-data", "see origin", "phone conversations"),
    "tedlium": ("Talks / lectures", "kfajdsl/tedlium (TED-LIUM r3 legacy train)", "CC-BY-NC-ND-3.0", "TED talks (human subtitles, auto-aligned); segments containing <unk> dropped"),
    "nptel": ("Talks / lectures", "skbose/indian-english-nptel-v0", "see origin", "Indian-English university lectures; Qwen3-ASR check dropped rows > 15% WER"),
    "voxpopuli": ("Talks / lectures", "facebook/voxpopuli (en)", "CC0", "European Parliament speeches"),
    "emilia": ("Conversational / web", "MrDragonFox/EN_Emilia_Yodas_616h", "CC-BY-4.0", "spontaneous YouTube English; rows where two independent transcripts agree (<= 5% WER) and audio PQ >= 6.5"),
    "spotify": ("Conversational / web", "SALT-NLP/spotify_podcast_ASR", "see origin", "podcasts, human verbatim, 2-3 speakers"),
    "peoples_speech": ("Conversational / web", "MLCommons/peoples_speech (clean)", "CC-BY / CC-BY-SA", "mixed web audio"),
    "svarah": ("Accents / domain", "ai4bharat/Svarah", "see origin", "Indian-accented English. A public TEST set: models trained on it cannot report official Svarah numbers"),
    "medical": ("Accents / domain", "yashtiwari/PaulMooney-Medical-ASR-Data", "see origin", "medical dictation / dialogue"),
    "librispeech": ("Read speech", "openslr/librispeech_asr (train.clean.100)", "CC-BY-4.0", "read audiobooks"),
}
VITW = ("Simulated acoustics", "zhifeixie/Voices-in-the-Wild-2M", "Apache-2.0",
        "LibriSpeech-train / Common Voice sentences through simulated {}; English rows only (Chinese rows dropped), 24 -> 16 kHz")
AUG = {"phone": "stored copy through a telephone channel (8 kHz line, 300-3400 Hz, G.711 mu-law, line hiss, lost packets)",
       "room": "stored copy through far-field reverb + babble / coloured noise at 0-15 dB SNR + transmission dropouts"}
ORDER = ["Meetings", "Calls", "Talks / lectures", "Conversational / web", "Accents / domain", "Read speech",
         "Simulated acoustics", "Stored augmented copies"]


def info(s):
    if s in INFO:
        return INFO[s]
    if s.startswith("vitw_"):
        return (VITW[0], VITW[1], VITW[2], VITW[3].format(s[5:].replace("_", " ")))
    for k, txt in AUG.items():
        if s.endswith("_" + k) and s[: -len(k) - 1] in INFO:
            b = INFO[s[: -len(k) - 1]]
            return ("Stored augmented copies", b[1], b[2], f"{s[: -len(k) - 1]}: {txt}")
    return ("Other", "?", "?", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    api = HfApi()
    files = api.list_repo_files(a.repo, repo_type="dataset")
    srcs = sorted(f[len("manifests/"):-len(".jsonl")] for f in files if f.startswith("manifests/") and f.endswith(".jsonl"))
    srcs = [s for s in srcs if any(f.startswith(f"data/{s}/") for f in files)]        # a manifest without data: skip
    st = {}
    for s in srcs:
        man = [json.loads(l) for l in open(hf_hub_download(a.repo, f"manifests/{s}.jsonl", repo_type="dataset"), encoding="utf-8")]
        h = sum(r["duration"] for r in man) / 3600
        st[s] = dict(rows=len(man), h=h, mean=h * 3600 / max(len(man), 1), spk=len({r.get("speaker") for r in man}))
    cfg = ["- config_name: all\n  data_files:\n  - split: train\n    path: data/**/*.parquet"]
    cores = {}
    for s in srcs:
        cfg.append(f"- config_name: {s}\n  data_files:\n  - split: train\n    path: data/{s}/**/*.parquet")
        for sub in sorted({f.split("/")[2] for f in files if f.startswith(f"data/{s}/core") and f.count("/") == 3}):
            cores[f"{s}_{sub[4:]}h"] = s
            cfg.append(f"- config_name: {s}_{sub[4:]}h\n  data_files:\n  - split: train\n    path: data/{s}/{sub}/*.parquet")
    total = sum(v["h"] for v in st.values())
    by_dom = {}
    for s in srcs:
        by_dom.setdefault(info(s)[0], []).append(s)
    sections = []
    for d in ORDER + sorted(set(by_dom) - set(ORDER)):
        if d not in by_dom:
            continue
        hrs = sum(st[s]["h"] for s in by_dom[d])
        rows = "\n".join(f"| `{s}` | {st[s]['h']:.1f} | {st[s]['rows']:,} | {st[s]['mean']:.1f} s | {st[s]['spk']:,} | "
                         f"{info(s)[3]} | [{info(s)[1].split(' ')[0]}](https://huggingface.co/datasets/{info(s)[1].split(' ')[0]}) | {info(s)[2]} |"
                         for s in by_dom[d])
        sections.append(f"### {d} ({hrs:.1f} h)\n\n| config | hours | rows | mean clip | speakers | content / processing | origin | license |\n"
                        f"|---|---:|---:|---:|---:|---|---|---|\n{rows}")
    lic = sorted({info(s)[2] for s in srcs})
    nc = any("non-commercial" in l.lower() or "NC" in l for l in lic)
    others = "\n".join(f"- [`{r}`](https://huggingface.co/datasets/{r}): {d}" for r, d in REPOS.items() if r != a.repo)
    first = srcs[0] if srcs else "all"
    card = f"""---
language: [en]
task_categories: [automatic-speech-recognition]
license: other
pretty_name: {a.repo.split('/')[-1]}
tags: [speech, asr, english, meetings, far-field]
configs:
{chr(10).join(cfg)}
---
# {a.repo.split('/')[-1]}

English speech-recognition training audio for the **BiBo voice** streaming ASR model (real-time captions):
{REPOS.get(a.repo, '')}. **{total:.1f} hours** in **{len(srcs)} sources**, 16 kHz mono FLAC, every clip with its human
transcript.{' **Private and non-commercial: see the licenses below before any use.**' if nc else ''}

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("{a.repo}", "{first}", split="train")            # one source
everything = load_dataset("{a.repo}", "all", split="train", streaming=True)
print(ds[0]["text"], ds[0]["duration"])
```

Configs: `all` (every source), one per source{', and ' + ', '.join('`' + c + '`' for c in cores) + ' (fixed core subsets)' if cores else ''}.
To restore the exact build tree (manifests + audio files) for training, use BiBo `voice/asr/fetch_mix.py --repo {a.repo}`.

## Sources

{chr(10).join(sections)}

## Processing

- Audio: decoded, downmixed to mono, resampled to **16 kHz**, stored as **FLAC** (lossless). Clips are 1-30 s
  (AMI: >= 0.2 s, so short backchannels such as "yeah" / "okay" survive for the meeting windows).
- Bad rows are **removed, never relabelled**: each source keeps only rows its own quality signal trusts (see the table).
- Long recordings were cut at their own segment / word timings (NOTSOFAR, SPGISpeech 2.0, TED-LIUM), never at fixed
  intervals, so no word is split by a cut.
- Text is each source's **original transcript** (casing, punctuation, markup as published). Training normalisation
  (lowercase, punctuation stripped, tags removed) is done later by BiBo `voice/asr/prep_en.py`. No speaker tags.

## Columns

| column | meaning |
|---|---|
| `audio` | 16 kHz FLAC |
| `text` | transcript as published by the source |
| `duration` | seconds |
| `source` | config name |
| `speaker` | speaker / call / talk / meeting id, used only to keep validation speaker-disjoint |
| `lang` | `en` |
| `meeting`, `begin`, `scenario` | meeting id + start time (AMI), else empty |

## Validation and test

There is no validation split in the repo: BiBo `prep_en.py` holds out whole speakers per source (<= 1 h per source,
copies of one recording share one held-out speaker set) and drops any training row whose text is a held-out sentence.
Official test sets of these corpora were not used, except Svarah (itself a test set, noted above).

## Licenses and attribution

Every row keeps the license of its origin; the dataset as a whole is `license: other`. Attribution goes to the origin
datasets linked in the tables. Licenses present: {', '.join(lic)}.{' Non-commercial / no-derivatives terms apply to the sources that carry them.' if nc else ''}

## Companion repos

{others}
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset",
                    commit_message="dataset card: domains, per-source stats, quick start, processing, licenses")
    print(f"CARD {a.repo}: {len(srcs)} sources, {total:.1f} h", flush=True)


if __name__ == "__main__":
    main()
