"""Our own bucketing sampler for train_asr.py (replaces Lhotse's DynamicBucketingSampler on the map-style path).

Lhotse's sampler fills buckets from a shuffle buffer: too small a buffer (10,000 clips) for big batches or many
buckets silently gives UNDERFILLED batches (2,400 s asked, ~1,050 s delivered), and its order needed seed /
concurrency fixes to be repeatable. This one is exact and simple:
  - every clip's duration is known up front (the manifest), so buckets are cut at duration quantiles holding EQUAL
    audio each, and each bucket is packed greedily into batches of <= batch_sec of REAL audio (one partial batch per
    bucket per epoch at most);
  - order: clips shuffled inside their bucket, then the whole batch list shuffled, both seeded by (seed, epoch) --
    bitwise the same batches on every run / resume;
  - same interface the train loop already uses: set_epoch(e), iteration yields CutSet batches, len().

    from packed_sampler import PackedBuckets
    s = PackedBuckets(cuts, batch_sec=2400, buckets=30, seed=23); s.set_epoch(0); print(s.stats())
"""
import random

from lhotse import CutSet


class PackedBuckets:
    def __init__(self, cuts, batch_sec, buckets=30, seed=23):
        self.cuts = list(cuts)
        self.batch_sec, self.seed, self.epoch = float(batch_sec), seed, 0
        order = sorted(range(len(self.cuts)), key=lambda i: self.cuts[i].duration)
        total = sum(self.cuts[i].duration for i in order)
        self.buckets, cur, acc = [], [], 0.0
        for i in order:                                   # equal AUDIO per bucket (not equal clip counts)
            cur.append(i)
            acc += self.cuts[i].duration
            if acc >= total * (len(self.buckets) + 1) / buckets and len(self.buckets) < buckets - 1:
                self.buckets.append(cur)
                cur = []
        if cur:
            self.buckets.append(cur)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        self._tails = set()
        out = []
        for b in self.buckets:
            ids = b[:]
            rng.shuffle(ids)
            cur, acc = [], 0.0
            for i in ids:
                d = self.cuts[i].duration
                if cur and acc + d > self.batch_sec:
                    out.append(cur)
                    cur, acc = [], 0.0
                cur.append(i)
                acc += d
            if cur:
                out.append(cur)
                self._tails.add(id(cur))                  # the bucket's remainder: the only batch allowed short
        rng.shuffle(out)
        return out

    def __iter__(self):
        for ids in self._batches():
            yield CutSet.from_cuts(self.cuts[i] for i in ids)

    def __len__(self):
        return len(self._batches())

    def stats(self, n=None):
        """Fill / padding / clips per batch over the epoch (metadata only): the preflight numbers."""
        bs = self._batches()[: n or None]
        real = [sum(self.cuts[i].duration for i in b) for b in bs]
        padded = [len(b) * max(self.cuts[i].duration for i in b) for b in bs]
        full = [r for b, r in zip(bs, real) if id(b) not in self._tails]   # bucket tails reported separately
        return dict(batches=len(bs), fill=sum(full) / (len(full) * self.batch_sec) if full else 0.0,
                    min_fill=min(full) / self.batch_sec if full else 0.0, tails=len(bs) - len(full),
                    padding=1 - sum(real) / sum(padded), utts=sum(map(len, bs)) / len(bs), real_s=sum(real) / len(bs))


if __name__ == "__main__":
    from lhotse import MonoCut, Recording

    def cut(i, d):
        return MonoCut(id=f"c{i}", start=0, duration=d, channel=0,
                       recording=Recording(id=f"r{i}", sources=[], sampling_rate=16000, num_samples=int(d * 16000), duration=d))

    rnd = random.Random(0)
    cs = [cut(i, rnd.uniform(1, 30)) for i in range(5000)]
    s = PackedBuckets(cs, batch_sec=1200, buckets=10, seed=1)
    st = s.stats()
    assert st["fill"] > 0.97 and st["min_fill"] > 0.97 and st["tails"] <= 10, st   # exact packing, <= 1 tail/bucket
    a = [[c.id for c in b] for b in s]
    assert a == [[c.id for c in b] for b in s]                            # same epoch -> same batches
    s.set_epoch(1)
    assert a != [[c.id for c in b] for b in s]                            # new epoch -> new order
    assert sorted(i for b in a for i in b) == sorted(c.id for c in cs)    # every clip exactly once per epoch
    print("packed_sampler ok", {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()})
