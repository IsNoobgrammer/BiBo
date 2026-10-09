"""Box check for train_asr's mid-epoch resume (CPU, metadata only):
    CUDA_VISIBLE_DEVICES= /home/marimo/asrenv/bin/python voice/asr/test_resume.py [train.jsonl]
SkipFirst on the real Lhotse sampler and through the rebuilt DataLoader: the batches after a resume == the
uninterrupted run's; callback state round-trips. (wandb_fork was checked by hand: bibo-asr-test project.)"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_asr as T  # noqa: E402
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


class Ids:
    def __getitem__(self, cuts):
        return [c.id for c in cuts]


def loader(workers):
    cfg = OmegaConf.create(dict(
        manifest_filepath=sys.argv[1] if len(sys.argv) > 1 else "/home/marimo/work/asr/run2/train.jsonl",
        use_lhotse=True, use_bucketing=True, num_buckets=30, batch_duration=1200, batch_size=None, max_duration=30,
        min_duration=0.1, shuffle=True, num_workers=workers, shuffle_buffer_size=10000, seed=23,
        shard_seed="randomized", concurrent_bucketing=False))
    return get_lhotse_dataloader_from_config(cfg, global_rank=0, world_size=1, dataset=Ids())


ids = lambda it, n: [tuple(c.id for c in b) for b in itertools.islice(it, n)]  # noqa: E731
s = T.SkipFirst(loader(0).sampler)
s.set_epoch(2); ref = ids(iter(s), 205)
s.set_epoch(2); s.skip = 200; assert ids(iter(s), 5) == ref[200:205], "skip mismatch"
s.set_epoch(2); assert ids(iter(s), 3) == ref[:3], "skip must apply once"
s.set_epoch(3); assert ids(iter(s), 3) != ref[:3], "epochs must differ"


class M:
    pass


m = M(); m._train_dl = loader(2); T.wrap_train_sampler(m)
m._train_dl.sampler.set_epoch(2); full = [tuple(b) for b in itertools.islice(iter(m._train_dl), 60)]
ep = T.EpochShuffle(); ep.load_state_dict({"done": 50})
ep.on_train_epoch_start(type("TR", (), {"current_epoch": 2})(), m)
assert [tuple(b) for b in itertools.islice(iter(m._train_dl), 10)] == full[50:60], "DataLoader resume mismatch"
a = T.AudioMeter(); a.gn_ema = 4.2; b = T.AudioMeter(); b.load_state_dict(a.state_dict()); assert b.gn_ema == 4.2
print("test_resume ok")
