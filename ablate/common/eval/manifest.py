"""Held-out bits-per-byte text sources — language + domain tagged, swappable.

Verified parallel English/Hindi public HF datasets (disjoint from the QTK-81K instruct training set):
  FLORES-200  : clean parallel sentences, best bpb probe (no chance floor).
  GSM8K       : English original (openai/gsm8k) + Hindi (bingbangboom/gsm8k-hindi), row-parallel math.

HINDI IS MANDATORY: assert_has_hindi() guarantees a lang="hi" source so Hindi bpb is always reported
separately. Swap/extend by editing MANIFEST (e.g. point at held-out shards of your own instruct data).
LL-MCQ benchmarks (Belebele/XNLI/Global-MMLU) live in mcq.py, not here.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Source:
    name: str
    lang: str                        # "en", "hi", ... (also pseudo-langs if needed)
    domain: str
    loader: Callable[[int], list]    # loader(n) -> list[str] raw text
    n: int = 300


def _hf(repo, config, split, to_text, streaming=True):
    cache = {}                                   # {n: [texts]} — periodic eval reuses, no re-download
    def load(n):
        for m, v in cache.items():               # a bigger cached pull already covers a smaller n
            if m >= n:
                return v[:n]
        from datasets import load_dataset
        ds = (load_dataset(repo, config, split=split, streaming=streaming) if config
              else load_dataset(repo, split=split, streaming=streaming))
        out = []
        for ex in ds:
            t = to_text(ex)
            if t and t.strip():
                out.append(t)
            if len(out) >= n:
                break
        if not out:
            raise RuntimeError(f"no samples from {repo}:{split}")
        cache[n] = out
        return out
    return load


_qa = lambda e: (e.get("question", "") + "\n" + e.get("answer", "")).strip()

MANIFEST = [
    Source("flores_en", "en", "text", _hf("facebook/flores", "eng_Latn", "devtest", lambda e: e["sentence"]), n=500),
    Source("flores_hi", "hi", "text", _hf("facebook/flores", "hin_Deva", "devtest", lambda e: e["sentence"]), n=500),
    Source("gsm8k_en",  "en", "math", _hf("openai/gsm8k", "main", "test", _qa), n=300),
    Source("gsm8k_hi",  "hi", "math", _hf("bingbangboom/gsm8k-hindi", None, "test_main", _qa), n=300),
]


def assert_has_hindi(manifest):
    if not any(s.lang == "hi" for s in manifest):
        raise AssertionError("bpb manifest MUST include a Hindi (lang='hi') source")


assert_has_hindi(MANIFEST)

# tiny synthetic manifest for the offline self-check (no downloads)
SELFCHECK_MANIFEST = [
    Source("synth_en", "en", "text", lambda n: ["the quick brown fox " * 40] * 4, n=4),
    Source("synth_hi", "hi", "text", lambda n: ["नमस्ते दुनिया यह एक परीक्षण है " * 40] * 4, n=4),
]
