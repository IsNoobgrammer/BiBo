"""Fast, lean training data path for train_asr.py (--loader fast): numpy manifest index + exact bucketing + int16
pinned batches. Replaces Lhotse's cut objects / buffered sampler / float32 collation on the training side only
(validation keeps NeMo's loader).

  ManifestIndex  the manifest as arrays: duration f32, source id i16, audio paths in ONE byte blob + offsets, and the
                 transcripts tokenized ONCE (the model's SentencePiece) into one flat int32 array + offsets; cached
                 next to the manifest (<manifest>.<tok>.idx.npz). filter() / with_column() take vectorized functions
                 over the arrays (ms over ~700k rows). Workers fork it copy-on-write: numpy pages are never touched
                 by refcounts, unlike 700k Python cut objects.
  BatchPlan      the exact sampler (packed_sampler's rules on arrays): equal-audio duration buckets, greedy packing to
                 <= batch_sec of REAL audio, one partial tail per bucket, order seeded by (seed, epoch); yields int64
                 row-index arrays; set_epoch / len / stats (the preflight).
  FastAudioDataset  __getitem__(row ids) -> (audio int16 (B, Tmax), audio_len int32, tokens int64 (B, Umax),
                 token_len int64) -- NeMo's own dtypes: FLAC decoded straight to int16 into one preallocated buffer, tokens gathered from
                 the flat array. DataLoader pin_memory=True pins the int16 batch; install() converts it to float32 on
                 the GPU after the copy (before aug_online, which wraps the same hook).

    python voice/asr/fast_loader.py                      # self-test (synthetic clips)
    python voice/asr/fast_loader.py --bench --manifest .../en3/train.jsonl --tok .../tokenizer.model --nemo_parity
"""
import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch


class ManifestIndex:
    COLS = ("dur", "src", "path_off", "tok", "tok_off")

    def __init__(self, arrays, sources, path_blob, rows=None):
        self.a, self.sources, self.path_blob = arrays, list(sources), path_blob
        self.rows = np.arange(len(arrays["dur"]), dtype=np.int64) if rows is None else rows

    # ---- build / cache ------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, manifest, tok_model, cache=True):
        key = hashlib.sha1(f"{os.path.abspath(manifest)}|{os.path.getmtime(manifest)}|{os.path.abspath(tok_model)}|"
                           f"{os.path.getmtime(tok_model)}".encode()).hexdigest()[:12]
        cpath = f"{manifest}.{key}.idx.npz"
        if cache and os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=False)
            arrays = {k: z[k] for k in cls.COLS + tuple(c for c in z.files if c.startswith("x_"))}
            return cls(arrays, json.loads(str(z["sources"])), z["path_blob"].tobytes())
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=tok_model)
        dur, src, paths, texts, src_ids = [], [], [], [], {}
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                dur.append(r["duration"])
                src.append(src_ids.setdefault(r.get("source", ""), len(src_ids)))
                paths.append(r["audio_filepath"].encode())
                texts.append(r["text"])
        toks = sp.encode(texts)                                   # one C++ call for every row
        tok_off = np.zeros(len(toks) + 1, np.int64)
        np.cumsum([len(t) for t in toks], out=tok_off[1:])
        path_off = np.zeros(len(paths) + 1, np.int64)
        np.cumsum([len(p) for p in paths], out=path_off[1:])
        arrays = dict(dur=np.asarray(dur, np.float32), src=np.asarray(src, np.int16), path_off=path_off,
                      tok=np.fromiter((t for row in toks for t in row), np.int32, count=int(tok_off[-1])),
                      tok_off=tok_off)
        sources = sorted(src_ids, key=src_ids.get)
        blob = b"".join(paths)
        if cache:
            np.savez(cpath, **arrays, sources=json.dumps(sources), path_blob=np.frombuffer(blob, np.uint8))
        return cls(arrays, sources, blob)

    # ---- view ---------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def col(self, name):
        """A per-row column of THIS view (dur, src, n_tok, or an x_ column added by with_column)."""
        if name == "n_tok":
            return (self.a["tok_off"][self.rows + 1] - self.a["tok_off"][self.rows]).astype(np.int32)
        return self.a[name][self.rows]

    def src_id(self, name):
        return self.sources.index(name)

    def filter(self, fn):
        """fn(view) -> bool mask over this view's rows, vectorized, e.g. lambda v: (v.col("dur") <= 25)."""
        m = np.asarray(fn(self), bool)
        assert m.shape == self.rows.shape, (m.shape, self.rows.shape)
        return ManifestIndex(self.a, self.sources, self.path_blob, self.rows[m])

    def with_column(self, name, fn):
        """A new per-row column x_<name> from fn(view) (vectorized); rows outside this view get 0."""
        full = np.zeros(len(self.a["dur"]), dtype=np.asarray(fn(self)).dtype)
        full[self.rows] = fn(self)
        return ManifestIndex({**self.a, f"x_{name}": full}, self.sources, self.path_blob, self.rows)

    def path(self, i):
        o = self.a["path_off"]
        return self.path_blob[o[i]:o[i + 1]].decode()

    def tokens(self, i):
        o = self.a["tok_off"]
        return self.a["tok"][o[i]:o[i + 1]]


class BatchPlan:
    """Exact bucketing sampler over an index view: yields np.int64 arrays of ROW ids (global, into the index)."""

    def __init__(self, index, batch_sec, buckets=30, seed=23):
        self.index, self.batch_sec, self.seed, self.epoch = index, float(batch_sec), seed, 0
        dur = index.col("dur").astype(np.float64)
        order = np.argsort(dur, kind="stable")
        cum = np.cumsum(dur[order])
        edges = np.searchsorted(cum, cum[-1] * np.arange(1, buckets) / buckets)       # equal audio per bucket
        self.buckets = [index.rows[order[s]] for s in np.split(np.arange(len(order)), edges) if len(s)]
        self._dur = index.a["dur"]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def plan(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        out, tails = [], []
        for b in self.buckets:
            ids = b[rng.permutation(len(b))]
            d = self._dur[ids].astype(np.float64)
            start, acc = 0, 0.0
            for j in range(len(ids)):                      # greedy, never over batch_sec (one clip may exceed alone)
                if j > start and acc + d[j] > self.batch_sec:
                    out.append(ids[start:j])
                    start, acc = j, 0.0
                acc += d[j]
            out.append(ids[start:])
            tails.append(len(out) - 1)
        perm = rng.permutation(len(out))
        self._tails = {int(np.nonzero(perm == t)[0][0]) for t in tails}
        return [out[p] for p in perm]

    def __iter__(self):
        yield from self.plan()

    def __len__(self):
        return len(self.plan())

    def stats(self):
        bs = self.plan()
        real = np.array([self._dur[b].sum() for b in bs])
        padded = np.array([len(b) * self._dur[b].max() for b in bs])
        full = np.array([r for k, r in enumerate(real) if k not in self._tails])
        return dict(batches=len(bs), fill=float(full.mean() / self.batch_sec), min_fill=float(full.min() / self.batch_sec),
                    tails=len(self._tails), padding=float(1 - real.sum() / padded.sum()),
                    utts=float(np.mean([len(b) for b in bs])), real_s=float(real.mean()))


class FastAudioDataset(torch.utils.data.Dataset):
    """Map-style: the DataLoader hands __getitem__ a whole batch of row ids (batch_size=None + BatchPlan)."""

    def __init__(self, index, sr=16000):
        self.index, self.sr = index, sr

    def __getitem__(self, ids):
        ix = self.index
        n = (ix.a["dur"][ids] * self.sr).astype(np.int64) + self.sr            # upper bound per row (+1 s slack)
        audio = np.zeros((len(ids), int(n.max())), np.int16)
        alen = np.zeros(len(ids), np.int32)
        for k, i in enumerate(ids):
            with sf.SoundFile(ix.path(i)) as f:
                m = f.read(frames=audio.shape[1], dtype="int16", out=audio[k])
            alen[k] = len(m) if m.ndim == 1 else m.shape[0]
        audio = audio[:, : int(alen.max())]
        to = ix.a["tok_off"]
        tl = (to[ids + 1] - to[ids]).astype(np.int64)                         # NeMo's dtype
        toks = np.zeros((len(ids), max(int(tl.max()), 1)), np.int64)
        for k, i in enumerate(ids):
            toks[k, : tl[k]] = ix.a["tok"][to[i]:to[i + 1]]
        return torch.from_numpy(audio), torch.from_numpy(alen), torch.from_numpy(toks), torch.from_numpy(tl)

    def __len__(self):
        return len(self.index)


def loader(index, batch_sec, buckets=30, seed=23, workers=None, prefetch=4):
    """(DataLoader, BatchPlan). Workers default to the CPU count - 1 (the main process feeds the GPU)."""
    plan = BatchPlan(index, batch_sec, buckets, seed)
    w = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    dl = torch.utils.data.DataLoader(FastAudioDataset(index), sampler=plan, batch_size=None, num_workers=w,
                                     pin_memory=True, persistent_workers=w > 0, prefetch_factor=prefetch if w else None)
    return dl, plan


def install(model):
    """int16 training batches -> float32 on the GPU right after the host->device copy (before aug_online, which wraps
    the same hook afterwards). Validation batches (NeMo's float loader) pass through untouched."""
    after = model.on_after_batch_transfer

    def on_after_batch_transfer(batch, dataloader_idx):
        batch = after(batch, dataloader_idx)
        if isinstance(batch, (tuple, list)) and batch[0].dtype == torch.int16:
            batch = (batch[0].float().mul_(1.0 / 32768.0), *batch[1:])
        return batch

    model.on_after_batch_transfer = on_after_batch_transfer


# ---------------------------------------------------------------------------------------------------------------------
def _selftest():
    import tempfile
    import sentencepiece as spm
    d = tempfile.mkdtemp()
    rng = random.Random(0)
    words = "the quick brown fox jumps over a lazy dog in meeting room jaundice".split()
    with open(f"{d}/text.txt", "w") as f:
        for _ in range(500):
            f.write(" ".join(rng.choice(words) for _ in range(8)) + "\n")
    spm.SentencePieceTrainer.train(input=f"{d}/text.txt", model_prefix=f"{d}/tok", vocab_size=40, model_type="bpe",
                                   minloglevel=2)
    rows = []
    for i in range(300):
        dur = rng.uniform(0.5, 6.0)
        x = (np.sin(np.arange(int(dur * 16000)) * 0.01 * (i + 1)) * 8000).astype(np.int16)
        p = f"{d}/{i}.flac"
        sf.write(p, x, 16000, subtype="PCM_16")
        rows.append({"audio_filepath": p, "duration": len(x) / 16000, "text": " ".join(rng.choice(words) for _ in range(5)),
                     "source": "a" if i % 3 else "b"})
    with open(f"{d}/m.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    ix = ManifestIndex.load(f"{d}/m.jsonl", f"{d}/tok.model")
    ix2 = ManifestIndex.load(f"{d}/m.jsonl", f"{d}/tok.model")                  # from the cache
    assert all(np.array_equal(ix.a[k], ix2.a[k]) for k in ManifestIndex.COLS) and ix.path(5) == ix2.path(5)
    sp = spm.SentencePieceProcessor(model_file=f"{d}/tok.model")
    assert all(list(ix.tokens(i)) == sp.encode(rows[i]["text"]) for i in range(300))   # same tokens as the model
    short = ix.filter(lambda v: v.col("dur") <= 3.0)
    assert len(short) == sum(r["duration"] <= 3.0 for r in rows) and (short.col("dur") <= 3.0).all()
    b_only = ix.filter(lambda v: v.col("src") == v.src_id("b"))
    assert len(b_only) == 100
    plan = BatchPlan(ix, batch_sec=60, buckets=4, seed=1)
    st = plan.stats()
    assert st["fill"] > 0.9 and st["tails"] == 4, st
    p1 = [b.tolist() for b in plan]
    assert p1 == [b.tolist() for b in plan] and sorted(i for b in p1 for i in b) == list(range(300))
    plan.set_epoch(1)
    assert p1 != [b.tolist() for b in plan]
    ds = FastAudioDataset(ix)
    a, al, t, tl = ds[np.array(p1[0])]
    assert a.dtype == torch.int16 and al.dtype == torch.int32 and t.dtype == torch.int64 and tl.dtype == torch.int64
    for k, i in enumerate(p1[0]):
        ref, _ = sf.read(rows[i]["audio_filepath"], dtype="int16")
        assert int(al[k]) == len(ref) and np.array_equal(a[k, : len(ref)].numpy(), ref) and (a[k, len(ref):] == 0).all()
        assert t[k, : tl[k]].tolist() == sp.encode(rows[i]["text"])
    dl, _ = loader(ix, 60, buckets=4, seed=1, workers=2)
    n = sum(1 for _ in dl)
    assert n == len(p1)
    print(f"fast_loader ok: {len(ix)} rows, {st['batches']} batches, fill {st['fill']:.3f}, padding {st['padding']:.3f}")


def _cpu():
    """(busy, total) jiffies over all CPUs from /proc/stat: the whole box, workers included."""
    v = list(map(int, open("/proc/stat").readline().split()[1:]))
    return sum(v) - v[3] - v[4], sum(v)


def _run(dl, n_max):
    b0, t0j = _cpu()
    t0, n, audio = time.perf_counter(), 0, 0.0
    for b in dl:
        n += 1
        audio += float(b[1].sum()) / 16000
        if n == n_max:
            break
    wall = time.perf_counter() - t0
    b1, t1j = _cpu()
    return n, wall, audio, 100 * (b1 - b0) / max(t1j - t0j, 1) * (os.cpu_count() or 1)


def _bench(a):
    import resource
    t0 = time.perf_counter()
    ix = ManifestIndex.load(a.manifest, a.tok)
    t_idx = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    f = ix.filter(lambda v: (v.col("dur") <= 25.0) & (v.col("n_tok") > 0))
    t_f = (time.perf_counter() - t0) * 1000
    print(f"index: {len(ix)} rows in {t_idx:.1f} s (cached next time), RSS {rss:.0f} MB; vectorized filter over all rows "
          f"{t_f:.1f} ms -> {len(f)} rows", flush=True)
    dl, plan = loader(ix, a.batch_sec, a.buckets, 23, a.workers)
    st = plan.stats()
    print("preflight", {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()}, flush=True)
    n, wall, audio, cpu = _run(dl, a.batches)
    print(f"FAST  {n} batches in {wall:.1f} s = {n / wall:.2f} batches/s, {audio / wall:.0f} audio-s/s delivered, "
          f"box CPU {cpu:.0f}% of {100 * (os.cpu_count() or 1)}% (workers {dl.num_workers})", flush=True)
    if a.nemo_parity:
        _nemo_compare(a, ix, plan)


def _nemo_compare(a, ix, plan):
    """The same clips through NeMo's Lhotse dataset: audio must be equal (int16/32768 == NeMo float), tokens identical;
    then NeMo's own loader speed on the same number of batches."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf, open_dict
    m = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_hybrid_large_streaming_multi", map_location="cpu")
    m.change_vocabulary(new_tokenizer_dir=os.path.dirname(a.tok), new_tokenizer_type="bpe")
    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.manifest, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=a.buckets,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=a.workers or max(1, (os.cpu_count() or 2) - 1), shuffle_buffer_size=10000, seed=23, pin_memory=True)
    m.setup_training_data(tr)
    ndl = m._train_dl
    # parity: NeMo's dataset on the cuts our first batch names
    want = {ix.path(i): i for i in next(iter(plan))[:32]}
    by_path = {}
    for c in ndl.sampler.cuts[0]:                        # find exactly those clips in NeMo's own cut set
        p = c.recording.sources[0].source
        if p in want:
            by_path[p] = c
            if len(by_path) == len(want):
                break
    ids = [i for p, i in want.items() if p in by_path]
    from lhotse import CutSet
    nb = ndl.dataset[CutSet.from_cuts(by_path[ix.path(i)] for i in ids)]
    fb = FastAudioDataset(ix)[np.array(ids)]
    same_len = torch.equal(nb[1].to(torch.int64), fb[1].to(torch.int64))
    audio_eq = same_len and all(torch.equal(nb[0][k, : nb[1][k]].float(), fb[0][k, : fb[1][k]].float() / 32768.0)
                                for k in range(len(ids)))
    tok_eq = all(nb[2][k, : nb[3][k]].tolist() == fb[2][k, : fb[3][k]].tolist() for k in range(len(ids)))
    print(f"PARITY vs NeMo on {len(ids)} clips: lengths equal {same_len}, audio equal {audio_eq}, tokens equal {tok_eq}; "
          f"NeMo dtypes {[str(x.dtype) for x in nb[:4]]}", flush=True)
    n, wall, audio, cpu = _run(ndl, a.batches)
    print(f"NEMO  {n} batches in {wall:.1f} s = {n / wall:.2f} batches/s, {audio / wall:.0f} audio-s/s delivered, "
          f"box CPU {cpu:.0f}% of {100 * (os.cpu_count() or 1)}% (workers {ndl.num_workers})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--tok")
    ap.add_argument("--batch_sec", type=float, default=2400)
    ap.add_argument("--buckets", type=int, default=30)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--nemo_parity", action="store_true")
    a = ap.parse_args()
    _bench(a) if a.bench else _selftest()
