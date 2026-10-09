"""Fine-tune the 114M cache-aware streaming FastConformer hybrid (RNNT + CTC) on our en/hi mix.

    python voice/asr/train_asr.py --run run1 --train R/train.jsonl --tok R/tok/tokenizer_spe_bpe_v4096 \
        --val R/val_en.jsonl R/val_hi.jsonl E/fleurs_en.jsonl E/fleurs_hi.jsonl [--compile_layers]

vs NeMo's speech_to_text_finetune.py (run0): Lhotse duration-bucketed batches (--batch_sec of real audio, ~no padding;
profile_train.py: padding was 50%), optional per-conformer-layer torch.compile, evaluation on every --val set each
--eval_hours of audio seen (Lhotse is an infinite iterator, so "epoch" = one eval interval), W&B logging of
train loss / WER / lr / audio hours seen / audio-s per s, top-2 + last checkpoints (.ckpt) and a final .nemo.
"""
import argparse
import json
import math
import os
import sys

# batch shapes change every step (Lhotse buckets): growable segments instead of cudaFree + re-malloc stalls
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_layout import define_metrics, wb_keys  # noqa: E402
import hf_sync  # noqa: E402


class RemapLogger(WandbLogger):
    """Every key goes through wb_layout.wb_key (core/ val/<source>/ eval/ speed/ optim/ misc/)."""

    def log_metrics(self, metrics, step=None):
        super().log_metrics(wb_keys(metrics), step)


class AudioMeter(Callback):
    """Audio hours seen, audio-seconds per wall-second (the throughput we optimise), ms/step, peak memory, grad norm."""

    def __init__(self, every=50):
        self.every, self.secs, self.t0, self.win, self.win_steps = every, 0.0, None, None, 0
        self.lsum, self.tsum, self.gn_ema = None, None, None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.every == 0:
            g = [p.grad.norm() for p in pl_module.parameters() if p.grad is not None]
            gn = torch.stack(g).norm().item() if g else 0.0
            pl_module.log("grad_norm", gn)
            # divergence alarm (run4: median 49 -> 132 -> 720 -> 3445 over ~1,500 steps, unnoticed until an eval).
            # Running mean in log space over the logged points; a sustained >5x jump prints GRAD_ALERT.
            if gn > 0:
                lg = math.log(gn)
                if self.gn_ema is not None and gn > 5 * math.exp(self.gn_ema) and trainer.global_step > 2000:
                    print(f"[GRAD_ALERT] step {trainer.global_step}: grad norm {gn:.1f} = "
                          f"{gn / math.exp(self.gn_ema):.1f}x its running level {math.exp(self.gn_ema):.1f}", flush=True)
                self.gn_ema = lg if self.gn_ema is None else 0.98 * self.gn_ema + 0.02 * lg
                pl_module.log("grad_norm_level", math.exp(self.gn_ema))

    # validation wall time is not training time (the step after each eval read ~650 audio-s/s)
    def on_validation_start(self, trainer, pl_module):
        self.vt = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if self.t0 is not None:
            self.t0 += time.perf_counter() - self.vt

    # the running level survives a resume (it restarted from one noisy step: a fake hump on the chart, alert blind)
    def state_dict(self):
        return {"gn_ema": self.gn_ema}

    def load_state_dict(self, state):
        self.gn_ema = state.get("gn_ema")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # samples summed ON the device: a .item() here every step drained the GPU queue (prof_step --sync_debug)
        s = batch[1].sum()
        self.win = s if self.win is None else self.win + s
        # per-token loss (the logged train loss is a per-utterance SUM, so it swings with utterance length)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None:
            ls = loss.detach() * batch[3].numel()
            tk = batch[3].sum()
            self.lsum = ls if self.lsum is None else self.lsum + ls
            self.tsum = tk if self.tsum is None else self.tsum + tk
        self.win_steps += 1
        if self.t0 is None:
            self.t0 = time.perf_counter()
        if trainer.global_step % self.every == 0:
            win_secs = self.win.item() / 16000
            self.secs += win_secs
            dt = max(time.perf_counter() - self.t0, 1e-6)
            pl_module.log_dict({"audio_hours_seen": self.secs / 3600, "audio_s_per_s": win_secs / dt,
                                "ms_per_step": 1000 * dt / max(self.win_steps, 1),
                                "mem_gb": torch.cuda.max_memory_allocated() / 2**30})
            if self.lsum is not None:
                pl_module.log("loss_per_token", (self.lsum / self.tsum.clamp(min=1)).item())
            self.t0, self.win, self.win_steps, self.lsum, self.tsum = time.perf_counter(), None, 0, None, None


class SkipFirst:
    """Train sampler wrapper: the next iteration drops its first `skip` batches (mid-epoch resume, see EpochShuffle).
    The sampler runs in the main process (map-style Lhotse: workers only load audio), so skipping reads metadata only."""

    def __init__(self, sampler):
        self.sampler, self.skip = sampler, 0

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        # next() only: Lhotse's sampler IS its own iterator and iter() on it restarts the epoch (so no `yield from`,
        # no returning it to code that may call iter() again)
        it, k, self.skip = iter(self.sampler), self.skip, 0
        for _ in range(k):
            next(it, None)
        while True:
            try:
                yield next(it)
            except StopIteration:
                return


def wrap_train_sampler(m):
    """Rebuild NeMo's train DataLoader around SkipFirst(sampler); every other DataLoader setting is kept."""
    from torch.utils.data import DataLoader
    old = m._train_dl
    assert not hasattr(old.dataset, "sampler"), "iterable (tarred / Shar) Lhotse data: sampler lives in the workers"
    m._train_dl = DataLoader(old.dataset, sampler=SkipFirst(old.sampler), batch_size=None, num_workers=old.num_workers,
                             collate_fn=old.collate_fn, pin_memory=old.pin_memory, worker_init_fn=old.worker_init_fn,
                             prefetch_factor=old.prefetch_factor, persistent_workers=old.persistent_workers)


class EpochShuffle(Callback):
    """A new (seeded) Lhotse order every data epoch. NeMo hands the sampler to a plain DataLoader and nobody calls
    set_epoch: with a fixed shard_seed every new iterator replays the same batches (run2 v1 trained on its first
    ~280 h eleven times). batch_log: ASR_BATCH_LOG=path writes 'step epoch lens-hash' per batch (replay checks)."""

    def __init__(self, aug_schedule=None, model=None):
        self.blog = open(os.environ["ASR_BATCH_LOG"], "a") if os.environ.get("ASR_BATCH_LOG") else None
        self.aug_schedule, self.m = aug_schedule, model
        self.done, self.epoch, self.restored = 0, 0, False  # batches trained in the current epoch (saved in the ckpt)

    def state_dict(self):
        return {"done": self.done, "epoch": self.epoch}

    def load_state_dict(self, state):
        # Applied HERE, at checkpoint restore: on a mid-epoch resume Lightning builds the epoch's iterator (and forks
        # the loader workers) BEFORE on_train_epoch_start -- run5's resumed epoch printed "p = 0.5" but trained
        # un-augmented (6,000 audio-s/s and clean-audio loss until the next epoch boundary, where it fell to 4,450)
        if "epoch" not in state:                     # checkpoint from before this field: no skip
            return
        self.done, self.epoch, self.restored = state["done"], state["epoch"], True
        self._begin(self.epoch, skip=self.done)
        print(f"[run] resume: epoch {self.epoch}, skipping the {self.done} batches it already trained on", flush=True)

    def _begin(self, epoch, skip=0):
        dl = self.m._train_dl
        s = getattr(dl.dataset, "sampler", None) or dl.sampler
        s.set_epoch(epoch)
        # resume mid-epoch: the sampler restarts the epoch (run5 replayed ~1,950 batches); skip what this epoch
        # already trained on: same epoch seed -> same order -> exact position (test_resume.py)
        if isinstance(s, SkipFirst):
            s.skip = skip
        if self.aug_schedule:                          # aug_online reads p in the main process (GPU)
            import aug_online
            aug_online.STATE.update(p=aug_online.p_for(self.aug_schedule, epoch), epoch=epoch)

    def on_train_epoch_start(self, trainer, pl_module):
        if not (self.restored and trainer.current_epoch == self.epoch):   # restored epoch: already set up at load
            self.done, self.epoch = 0, trainer.current_epoch
            self._begin(self.epoch)
        self.restored = False
        if self.aug_schedule:
            import aug_online
            pl_module.log("aug_p", aug_online.STATE["p"])
            print(f"[run] epoch {trainer.current_epoch}: augmentation p = {aug_online.STATE['p']}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.done += 1
        if self.blog:
            import hashlib
            h = lambda t: hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()[:12]  # noqa: E731
            # lens (augmentation changes them) + tokens (data order only)
            self.blog.write(f"{trainer.global_step} {trainer.current_epoch} {h(batch[1])} {h(batch[2])}\n")
            self.blog.flush()


class GradDiag(Callback):
    """Divergence forensics: every `every` steps log the grad norm per module group (before clipping) and the
    look-ahead (right context) that batch trained with, under diag/*. Groups: encoder attn / ff / conv / norm /
    pre_encode, decoder (prediction net), joint, ctc_decoder; plus the worst single encoder layer."""

    def __init__(self, every=50):
        self.every = every

    @staticmethod
    def group(n):
        if n.startswith("encoder.layers."):
            sub = n.split(".")[3]
            for g, keys in (("attn", ("self_attn",)), ("ff", ("feed_forward",)), ("conv", ("conv",)), ("norm", ("norm",))):
                if any(sub.startswith(k) for k in keys):
                    return "enc_" + g
            return "enc_other"
        for g in ("encoder.pre_encode", "decoder", "joint", "ctc_decoder"):
            if n.startswith(g):
                return g.split(".")[-1]
        return "other"

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.every:
            return
        acc, layer = {}, {}
        for n, p in pl_module.named_parameters():
            if p.grad is None:
                continue
            sq = p.grad.detach().float().pow(2).sum()
            g = self.group(n)
            acc[g] = acc.get(g, 0) + sq
            if n.startswith("encoder.layers."):
                li = int(n.split(".")[2])
                layer[li] = layer.get(li, 0) + sq
        out = {f"diag/gn_{g}": v.sqrt().item() for g, v in acc.items()}
        if layer:
            li, v = max(layer.items(), key=lambda kv: kv[1].item())
            out["diag/gn_worst_layer"] = v.sqrt().item()
            out["diag/worst_layer_idx"] = float(li)
        ctx = getattr(pl_module.encoder, "_fused_ctx", None)
        if ctx:
            out["diag/lookahead_r"] = float(ctx[0][1])
        pl_module.log_dict(out)


class StopAt(Callback):
    """--stop_step: end a (resumed) run at this global step WITHOUT changing max_steps, so the LR schedule is
    identical to the full run's (replay experiments)."""

    def __init__(self, step):
        self.step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.step:
            trainer.should_stop = True


class LrScale(Callback):
    """--lr_scale: multiply the scheduler's base lrs once training starts (after a resume restored them)."""

    def __init__(self, scale):
        self.scale = scale

    def on_train_start(self, trainer, pl_module):
        for cfg in trainer.lr_scheduler_configs:
            sch = cfg.scheduler
            if hasattr(sch, "base_lrs"):
                sch.base_lrs = [b * self.scale for b in sch.base_lrs]
            for g in sch.optimizer.param_groups:
                g["lr"] *= self.scale
        print(f"[run] lr scaled x{self.scale}", flush=True)


class SaveSteps(Callback):
    def __init__(self, steps, ckpt_dir):
        self.steps, self.ckpt_dir = set(steps), ckpt_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps:
            self.steps.discard(trainer.global_step)
            trainer.save_checkpoint(os.path.join(self.ckpt_dir, f"step{trainer.global_step}.ckpt"))


def wsd_lambda(warmup, max_steps, decay_frac, min_ratio):
    """lr multiplier: linear warm-up, flat at 1, linear decay over the last decay_frac of max_steps to min_ratio.
    The decay start depends only on max_steps, so a run's stable phase is identical to any longer run's: a branch
    resumed from a stable-phase checkpoint with a shorter --total_hours decays from exactly that point."""
    start = int(max_steps * (1 - decay_frac))

    def f(s):
        if s < warmup:
            return (s + 1) / warmup
        if s < start:
            return 1.0
        return max(min_ratio, 1.0 - (s - start) / max(max_steps - start, 1) * (1.0 - min_ratio))
    return f


class HFSync(Callback):
    """After every evaluation, push last.ckpt to the private HF repo in a background thread, so a dead box loses at
    most one eval interval. Lightning runs ModelCheckpoint AFTER every other callback, so at on_validation_end
    last.ckpt is still the PREVIOUS eval's (run5 resumed from step 9000, not 10800): push at the next train batch
    (or at train end), when the new file is on disk."""

    def __init__(self, run, ckpt_dir):
        self.run, self.ckpt_dir, self.pending = run, ckpt_dir, False

    def on_validation_end(self, trainer, pl_module):
        self.pending = trainer.global_step > 0 and not trainer.sanity_checking

    def _push(self):
        if self.pending:
            self.pending = False
            hf_sync.push(self.run, ckpt=os.path.join(self.ckpt_dir, "last.ckpt"))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._push()

    def on_train_end(self, trainer, pl_module):
        self._push()


class ValAggregate(Callback):
    """val_wer_all / _en / _hi = per-source WERs weighted by reference words (= corpus WER over the sources, given
    each source's WER). multispk and FLEURS are reported on their own, not in the aggregate."""

    def __init__(self, manifests):
        self.sets = {}
        for p in manifests:
            stem = os.path.basename(p)[:-6]
            if not stem.startswith("val_") or stem in ("val_en", "val_hi", "val_multispk"):
                continue
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            self.sets[stem] = (rows[0]["lang"], sum(len(r["text"].split()) for r in rows))

    def on_validation_end(self, trainer, pl_module):
        # NOT on_validation_epoch_end: Lightning runs callback hooks before the module's, and NeMo publishes the
        # per-dataloader WERs in the module's hook. Runs before ModelCheckpoint (callback order), which reads it.
        cm = trainer.callback_metrics
        acc = {"all": [0.0, 0], "en": [0.0, 0], "hi": [0.0, 0]}
        for stem, (lang, words) in self.sets.items():
            w = cm.get(f"{stem}_val_wer")
            if w is None:
                continue
            for k in ("all", lang):
                acc[k][0] += float(w) * words
                acc[k][1] += words
        if acc["all"][1] < sum(w for _, w in self.sets.values()):
            # a resume re-enters the interrupted validation loop and runs only its tail (run5 @ 9000 logged an
            # English-only 20.6 %): never publish an aggregate over a subset of the sources
            print(f"[run] partial validation at step {trainer.global_step}: aggregate not logged", flush=True)
            trainer.callback_metrics["val_wer_all"] = torch.tensor(float("inf"))  # ModelCheckpoint raises if missing
            return
        out = {f"val_wer_{k}": e / n for k, (e, n) in acc.items() if n}
        for k, v in out.items():
            trainer.callback_metrics[k] = torch.tensor(v)
        trainer.logger.log_metrics(out, step=trainer.global_step)


def wandb_fork(logger, project, old_id, upto_step):
    """Manual W&B fork for a resume (rewind / fork_from are private preview on this account): copy the old run's
    history up to the checkpoint's global step into the NEW run, then tag + rename the old one 'superseded'. The
    steps the old run logged after the checkpoint (lost with the box) no longer overlap the resumed ones."""
    import wandb
    api = wandb.Api()
    old = api.run(f"{logger.experiment.entity}/{project}/{old_id}")
    n = 0
    for row in old.scan_history(page_size=5000):
        st = row.get("trainer/global_step")
        if st is None or st > upto_step:
            continue
        logger.experiment.log({k: v for k, v in row.items() if not k.startswith("_") and v is not None
                               and not isinstance(v, dict)})
        n += 1
    old.name, old.tags = f"{old.name}-superseded", [*old.tags, "superseded"]
    old.update()
    print(f"[run] W&B: resumed as {logger.experiment.id}, copied {n} history rows <= step {upto_step} from {old_id} "
          f"(now '{old.name}')", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="/home/marimo/work/asr/exp")
    ap.add_argument("--init", default="stt_en_fastconformer_hybrid_large_streaming_multi")
    ap.add_argument("--batch_sec", type=float, default=1200, help="real audio seconds per batch (Lhotse bucketing)")
    ap.add_argument("--total_hours", type=float, default=3200, help="audio hours to train on (~7 x 460 h)")
    ap.add_argument("--eval_hours", type=float, default=150,
                    help="evaluate every this many audio hours (must be < one data epoch); 0 = every data epoch")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sched", choices=["cosine", "wsd"], default="cosine",
                    help="wsd = linear warm-up, flat at --lr, linear decay over the last --decay_frac to 1e-5")
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--lookahead_probs", nargs="*", default=None,
                    help="training mix of the multi-lookahead contexts as RIGHT:PROB, right context in 80 ms frames, "
                         "e.g. 0:0.2 1:0.4 6:0.2 13:0.2 (NeMo default: uniform)")
    ap.add_argument("--lookaheads", type=int, nargs="*", default=None,
                    help="replace the trained look-ahead set: RIGHT contexts in 80 ms frames, e.g. 13 6 3 1 0 "
                         "(the FIRST is the default = validation context; left context 70 kept)")
    ap.add_argument("--fastemit", type=float, default=None, help="FastEmit lambda (model default 0.005)")
    ap.add_argument("--save_steps", type=int, nargs="*", default=[],
                    help="also save a full checkpoint at these global steps (WSD: branch decays from the stable phase)")
    ap.add_argument("--aug_schedule", type=float, nargs="*", default=None,
                    help="per-epoch fraction of utterances augmented on the fly (aug_online.py), last value repeats, "
                         "e.g. 0 0.5 0.5 0.5 0.2")
    ap.add_argument("--grad_diag", type=int, default=0, help="log per-module grad norms + look-ahead every N steps")
    ap.add_argument("--stop_step", type=int, default=None, help="stop at this global step (schedule unchanged)")
    ap.add_argument("--lr_scale", type=float, default=None, help="multiply the scheduler lrs at train start (resumes)")
    ap.add_argument("--ckpt", default=None, help="resume from this checkpoint (e.g. a --save_steps one, for a decay branch)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--compile_layers", action="store_true")
    ap.add_argument("--fused_joint", action="store_true", help="tkf fused joint + RNN-T loss (voice/asr/fused_joint.py)")
    ap.add_argument("--no_hf", action="store_true", help="A/B and smoke runs: no HF checkpoint pull / push")
    ap.add_argument("--bf16_master", action="store_true", help="bf16 model + fp32 master weights, no autocast")
    ap.add_argument("--fused_layer", action="store_true", help="tkf residual+dropout+LayerNorm (voice/asr/fused_layer.py)")
    ap.add_argument("--fp32_residual", action="store_true", help="with --bf16_master --fused_layer: fp32 residual stream")
    ap.add_argument("--fused_attn", action="store_true", help="tkf banded rel-pos attention (voice/asr/fused_attn.py)")
    ap.add_argument("--fused_conv", action="store_true", help="tkf conv module + FFN silu/dropout (voice/asr/fused_conv.py)")
    ap.add_argument("--fused_ctc", action="store_true", help="tkf deterministic CTC loss (voice/asr/fused_ctc.py)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise same-seed runs (with --fused_ctc): torch deterministic algorithms, cuDNN deterministic")
    ap.add_argument("--project", default="bibo-asr")
    a = ap.parse_args()
    assert not a.fp32_residual or (a.bf16_master and a.fused_layer), "--fp32_residual needs --bf16_master --fused_layer"

    import nemo.collections.asr as nemo_asr
    torch.set_float32_matmul_precision("high")
    if a.deterministic:    # det_probe.py: without these the gradient bits still differ run to run (cuDNN / cuBLAS)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    # eval every eval_steps batches ACROSS data epochs (val_check_interval, check_val_every_n_epoch=None): the train
    # iterator keeps running through evals. (limit_train_batches=eval_steps ended a Lightning epoch at every eval and
    # rebuilt the iterator -> with a fixed shard_seed, the same batches every segment.) eval_hours 0 = every data epoch.
    eval_steps = int(a.eval_hours * 3600 / a.batch_sec) or None
    max_steps = int(a.total_hours * 3600 / a.batch_sec)

    if a.seed is not None:
        pl.seed_everything(a.seed)
    if a.init.endswith(".nemo"):                       # a trained run (same vocabulary): A/B arms start identical
        m = nemo_asr.models.ASRModel.restore_from(a.init)
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(a.init)
        m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")

    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=30,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=a.workers, shuffle_buffer_size=10000, seed=23 if a.seed is None else a.seed, pin_memory=True,
                  # same seed -> same batches (det_probe.py): NeMo's defaults draw per-worker seeds from the OS RNG
                  # (shard_seed "trng") and fill buckets from a background thread (timing-dependent batch contents)
                  shard_seed="randomized", concurrent_bucketing=False)
    m.setup_training_data(tr)
    wrap_train_sampler(m)
    if a.aug_schedule:
        import aug_online                              # on the GPU, after the host->device copy
        aug_online.install(m, 23 if a.seed is None else a.seed)
    m.compute_eval_loss = True          # the model config had compute_eval_loss: false -> no validation loss at all
    va = OmegaConf.create(OmegaConf.to_container(m.cfg.validation_ds))
    with open_dict(va):
        va.update(manifest_filepath=a.val, batch_size=64, num_workers=4, shuffle=False)
    m.setup_multiple_validation_data(va)
    with open_dict(m.cfg):
        m.cfg.train_ds, m.cfg.validation_ds = tr, va

    if a.compile_layers:   # see profile_train.py: whole-encoder compile graph-breaks and recompiles per length
        torch._dynamo.config.cache_size_limit = 64
        for i, layer in enumerate(m.encoder.layers):
            m.encoder.layers[i] = torch.compile(layer, dynamic=True)

    if a.lookaheads:
        left = m.encoder.att_context_size_all[0][0]
        m.encoder.att_context_size_all = [[left, r] for r in a.lookaheads]
        m.encoder.att_context_probs = [1.0 / len(a.lookaheads)] * len(a.lookaheads)
        m.encoder.set_default_att_context_size([left, a.lookaheads[0]])
        with open_dict(m.cfg):                                       # saved .nemo / GGUF export see the new set
            m.cfg.encoder.att_context_size = [[left, r] for r in a.lookaheads]
            m.cfg.encoder.att_context_probs = m.encoder.att_context_probs
        print(f"[run] look-ahead set {m.encoder.att_context_size_all}", flush=True)
    if a.lookahead_probs:
        want = {int(r): float(p) for r, p in (x.split(":") for x in a.lookahead_probs)}
        ctxs = [list(c) for c in m.encoder.att_context_size_all]
        assert sorted(want) == sorted(c[1] for c in ctxs) and abs(sum(want.values()) - 1) < 1e-6, (want, ctxs)
        m.encoder.att_context_probs = [want[c[1]] for c in ctxs]       # sampled per batch (random.choices)
        with open_dict(m.cfg):
            m.cfg.encoder.att_context_probs = m.encoder.att_context_probs
        print(f"[run] lookahead mix {dict((c[1] * 80, p) for c, p in zip(ctxs, m.encoder.att_context_probs))} (ms: prob)",
              flush=True)
    if a.fastemit is not None:                                       # fused_joint reads it from the loss config
        with open_dict(m.cfg):
            m.cfg.loss.warprnnt_numba_kwargs.fastemit_lambda = a.fastemit
    if a.fused_joint:
        import fused_joint
        fused_joint.enable(m)
    if a.bf16_master:
        import bf16_master
        bf16_master.to_bf16(m)
    if a.fused_layer:
        import fused_layer
        fused_layer.enable(m, fp32_residual=a.fp32_residual)
    if a.fused_attn:
        import fused_attn
        fused_attn.enable(m)
    if a.fused_conv:
        import fused_conv
        fused_conv.enable(m)
    if a.fused_ctc:
        import fused_ctc
        fused_ctc.enable(m)

    ckpt_dir = os.path.join(a.out, a.run)
    os.makedirs(ckpt_dir, exist_ok=True)
    # resume: local last.ckpt, else the one in the HF repo (fresh box); the W&B run continues under the same id
    _, hf_ckpt, wid = (None, None, None) if a.no_hf else hf_sync.pull(a.run, os.path.dirname(a.out))
    resume = os.path.join(ckpt_dir, "last.ckpt") if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")) else hf_ckpt
    resume = a.ckpt or resume
    # W&B: a resume gets a NEW run carrying the old one's history up to the checkpoint (wandb_fork); continuing the
    # old run drew two overlapping lines (run5: the lost 9000-11600 tail and the resumed steps)
    fork = bool(wid and resume and not a.ckpt)
    logger = RemapLogger(project=a.project, name=a.run, save_dir=ckpt_dir, id=None if fork else wid, resume="allow",
                         config={**vars(a), "eval_steps": eval_steps, "max_steps": max_steps})
    define_metrics(logger.experiment)
    if fork:
        wandb_fork(logger, a.project, wid, torch.load(resume, map_location="cpu", mmap=True, weights_only=False)["global_step"])
    if not a.no_hf:
        hf_sync.push(a.run, tok_dir=a.tok, wandb_id=logger.experiment.id, block=True)
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="32-true" if a.bf16_master else "bf16-mixed", max_steps=max_steps, max_epochs=-1,
        val_check_interval=eval_steps, check_val_every_n_epoch=None if eval_steps else 1,
        num_sanity_val_steps=0, gradient_clip_val=None if a.bf16_master else 1.0, log_every_n_steps=50,
        use_distributed_sampler=False, logger=logger, enable_checkpointing=True, benchmark=False,
        callbacks=[AudioMeter(), EpochShuffle(a.aug_schedule, m), TQDMProgressBar(refresh_rate=50), ValAggregate(a.val), LearningRateMonitor("step"),
                   ModelCheckpoint(dirpath=ckpt_dir, monitor="val_wer_all", mode="min", save_top_k=2,
                                   save_last=True, filename="{step}-{val_wer_all:.4f}", save_on_train_epoch_end=False),
                   *([SaveSteps(a.save_steps, ckpt_dir)] if a.save_steps else []),
                   *([GradDiag(a.grad_diag)] if a.grad_diag else []),
                   *([StopAt(a.stop_step)] if a.stop_step else []),
                   *([LrScale(a.lr_scale)] if a.lr_scale else []),
                   *([] if a.no_hf else [HFSync(a.run, ckpt_dir)])])
    m.set_trainer(trainer)
    m.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_steps": a.warmup, "min_lr": 1e-5, "max_steps": max_steps}}))
    if a.bf16_master:
        bf16_master.wrap(m._optimizer, clip=1.0)                    # clips on the fp32 masters
    if a.sched == "wsd":
        for g in m._optimizer.param_groups:                        # NeMo's scheduler may have set a warm-up lr
            g["lr"] = a.lr
            g.pop("initial_lr", None)
        sched = torch.optim.lr_scheduler.LambdaLR(m._optimizer, wsd_lambda(a.warmup, max_steps, a.decay_frac, 1e-5 / a.lr))
        m._scheduler = {"scheduler": sched, "interval": "step", "frequency": 1}
        # NeMo's configure_optimizers() re-runs setup_optimization() at fit time, rebuilding the cosine scheduler
        # over ours (smoke test: lr followed the cosine). Hand Lightning this optimizer + scheduler directly.
        m.configure_optimizers = lambda: ([m._optimizer], [m._scheduler])
        print(f"[run] WSD: warm-up {a.warmup}, flat to step {int(max_steps * (1 - a.decay_frac))}, "
              f"linear decay to {max_steps}", flush=True)
    print(f"[run] {a.run}: max_steps {max_steps}, eval every {eval_steps} steps (~{a.eval_hours} h), "
          f"batch {a.batch_sec} s, val {[os.path.basename(v) for v in a.val]}", flush=True)
    print(f"[run] resume from {resume}", flush=True)
    trainer.fit(m, ckpt_path=resume)
    trainer.validate(m)                                         # the final model, wherever the last eval fell
    m.save_to(os.path.join(ckpt_dir, f"{a.run}.nemo"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
