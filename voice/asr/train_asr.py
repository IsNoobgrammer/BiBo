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


def wrap_train_sampler(m, a=None):
    """Rebuild NeMo's train DataLoader around SkipFirst(sampler); every other DataLoader setting is kept.
    --sampler packed (default): our PackedBuckets (exact batches, packed_sampler.py) replaces Lhotse's buffered
    sampler, and the PREFLIGHT runs here: the epoch's fill / padding / clips per batch, abort if batches are short."""
    from torch.utils.data import DataLoader
    old = m._train_dl
    assert not hasattr(old.dataset, "sampler"), "iterable (tarred / Shar) Lhotse data: sampler lives in the workers"
    if a is not None and a.loader == "fast":           # fast_loader: numpy index, exact batches, int16 pinned
        import fast_loader
        tok_model = os.path.join(a.tok, "tokenizer.model")
        if a.keep_vocab:                 # hub model with its own vocabulary: tokenise the labels with ITS tokenizer
            tok_model = os.path.join(a.out, "hub_tok", a.init.replace("/", "_") + ".model")   # (labels past its
            if not os.path.exists(tok_model):                                             # embedding crashed RNN-T)
                os.makedirs(os.path.dirname(tok_model), exist_ok=True)
                with open(tok_model, "wb") as f:
                    f.write(m.tokenizer.tokenizer.serialized_model_proto())
            print(f"[run] fast loader tokenises with the model's own tokenizer {tok_model}", flush=True)
        ix = fast_loader.ManifestIndex.load(a.train, tok_model)
        ix = ix.filter(lambda v: (v.col("dur") >= 0.1) & (v.col("dur") <= 30.0))   # NeMo's train_ds limits
        if a.source_weight:                            # per-source mix weights (row repetition, no new audio)
            w = {k: float(v) for k, v in (x.rsplit(":", 1) for x in a.source_weight)}
            h0 = ix.hours_by_source()
            ix = ix.reweight(w, 23 if a.seed is None else a.seed)
            h1, T = ix.hours_by_source(), sum(ix.hours_by_source().values())
            print("[mix] " + " | ".join(f"{n} {h0[n]:.0f}->{h1.get(n, 0):.0f} h ({100 * h1.get(n, 0) / T:.1f}%)"
                                        for n in sorted(h0, key=lambda n: -h1.get(n, 0))) + f" | total {T:.0f} h",
                  flush=True)
        dl, plan = fast_loader.loader(ix, a.batch_sec, a.buckets, 23 if a.seed is None else a.seed, a.workers)
        st = plan.stats()
        print(f"[preflight] fast loader: {len(ix)} clips, {st['batches']} batches/epoch, fill {100 * st['fill']:.1f}% "
              f"(worst {100 * st['min_fill']:.1f}%) of {a.batch_sec:.0f} s, {st['tails']} bucket tails, padding "
              f"{100 * st['padding']:.1f}%, {st['utts']:.0f} clips/batch, {dl.num_workers} workers", flush=True)
        assert st["fill"] >= 0.95, f"preflight: batches only {100 * st['fill']:.1f}% full -- fix before training"
        if getattr(a, "kd_targets", None) or getattr(a, "kd_cache", False):   # the batch carries its row ids
            dl.dataset.with_ids = True
        m._train_dl = torch.utils.data.DataLoader(dl.dataset, sampler=SkipFirst(plan), batch_size=None,
                                                  num_workers=dl.num_workers, pin_memory=True,
                                                  persistent_workers=dl.num_workers > 0, prefetch_factor=4)
        fast_loader.install(m)                         # int16 -> float32 on the GPU, before aug_online's hook
        return
    sampler = old.sampler
    if a is not None and a.sampler == "packed":
        from packed_sampler import PackedBuckets
        sampler = PackedBuckets(old.sampler.cuts[0], a.batch_sec, a.buckets, 23 if a.seed is None else a.seed)
        st = sampler.stats()
        print(f"[preflight] sampler packed: {st['batches']} batches/epoch, fill {100 * st['fill']:.1f}% "
              f"(worst {100 * st['min_fill']:.1f}%) of {a.batch_sec:.0f} s, {st['tails']} bucket tails, padding "
              f"{100 * st['padding']:.1f}%, {st['utts']:.0f} clips/batch", flush=True)
        assert st["fill"] >= 0.95, f"preflight: batches only {100 * st['fill']:.1f}% full -- fix before training"
    m._train_dl = DataLoader(old.dataset, sampler=SkipFirst(sampler), batch_size=None, num_workers=old.num_workers,
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


class ProfileSteps(Callback):
    """--profile_steps A:B -- where does a REAL train_asr step go (loader, Lightning, optimizer, logging included)?
    Steps A..B under torch.profiler (CPU + CUDA): wall ms/step (one sync at each end), GPU kernel ms/step (-> GPU idle
    share), the top CPU ops by self time (host syncs show up as cudaStreamSynchronize / _local_scalar_dense / item),
    top CUDA kernels; then ONE more step under torch.cuda sync-debug mode listing every host-sync site (file:line)."""

    def __init__(self, spec):
        self.a, self.b = map(int, spec.split(":"))
        self.prof, self.sites, self.t0, self.marks, self.phases = None, None, None, None, {}

    def _ev(self, key):                                 # GPU-side phase mark (CUDA event on the current stream)
        if self.marks is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append((key, e))

    def _wrap_steps(self, trainer):
        """Time every (sub-)optimizer's step on the GPU: Combined -> its FusedMuon / AdamW; plain AdamW itself."""
        for o in trainer.optimizers:
            for sub in getattr(o, "opts", [o]):
                if getattr(sub, "_phase_wrapped", False):
                    continue
                f, name = sub.step, type(sub).__name__

                def step(closure=None, _f=f, _n=name):
                    if closure is None:                     # a Combined sub-optimizer: the update only
                        self._ev(f"opt:{_n}")
                    else:                                   # Lightning's step(closure) runs fwd+bwd first
                        def c(_c=closure):
                            r = _c()
                            self._ev(f"opt:{_n}")
                            return r
                        closure = c
                    r = _f(closure) if closure is not None else _f()
                    self._ev(f"end:{_n}")
                    return r
                sub.step, sub._phase_wrapped = step, True

    MODS = ("encoder", "decoder", "joint", "ctc_decoder")        # forward ranges per top-level module
    BWD = {"_RNNTJointBackward": "joint (fused joint + RNN-T loss)", "CudnnRnnBackward0": "decoder (pred-net LSTM)",
           "EmbeddingBackward0": "decoder (pred-net LSTM)", "_PassBackward": "ctc_decoder", "_CTCBackward": "ctc loss"}

    def _hook_modules(self, pl_module):
        self._hooks, self._open = [], {}
        for name in self.MODS:
            mod = getattr(pl_module, name, None)
            if mod is None:
                continue

            def pre(m, args, _n=name):
                r = torch.autograd.profiler.record_function(f"mod:{_n}")
                r.__enter__()
                self._open.setdefault(_n, []).append(r)

            def post(m, args, out, _n=name):
                self._open[_n].pop().__exit__(None, None, None)
            self._hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]

    def _by_module(self, n):
        """GPU ms/step per component: forward = kernels under each module's range; backward = kernels under each
        autograd node, classified by node name (everything unlisted is the encoder / glue)."""
        fwd, bwd = {}, {}
        for e in self.prof.events():
            if e.name.startswith("mod:"):
                fwd[e.name[4:]] = fwd.get(e.name[4:], 0.0) + e.device_time_total
            elif e.name.startswith("autograd::engine::evaluate_function: "):
                k = self.BWD.get(e.name.split(": ", 1)[1], "encoder + rest")
                bwd[k] = bwd.get(k, 0.0) + e.device_time_total
        print("[profile] GPU ms/step by component (forward ranges / backward autograd nodes):", flush=True)
        for tag, d in (("fwd", fwd), ("bwd", bwd)):
            for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
                print(f"    {tag} {v / n / 1000:7.2f}  {k}", flush=True)

    def on_before_backward(self, trainer, pl_module, loss):
        self._ev("backward")

    def on_after_backward(self, trainer, pl_module):
        self._ev("after_backward")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        g = trainer.global_step
        if self.a <= g <= self.b:
            if g == self.a:
                self._wrap_steps(trainer)
                self.phases = {}
            self.marks = []
            self._ev("forward")
        if g == self.a:
            opts = trainer.optimizers
            self._hook_modules(pl_module)
            print(f"[profile] optimizer: {[type(o).__name__ for o in opts]}, groups fused / foreach: "
                  f"{[(gr.get('fused'), gr.get('foreach')) for o in opts for gr in o.param_groups]}", flush=True)
            torch.cuda.synchronize()
            self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA])
            self.prof.__enter__()
            self.t0 = time.perf_counter()
        if g == self.b + 1:
            import collections, traceback, warnings
            self.sites = collections.Counter()

            def show(msg, cat, fn, ln, file=None, line=None):
                for fr in reversed(traceback.extract_stack()[:-1]):
                    if "/torch/" not in fr.filename and "warnings" not in fr.filename:
                        self.sites[f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"] += 1
                        return
            self._old = warnings.showwarning
            warnings.showwarning = show
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        g = trainer.global_step                         # already incremented for this step
        if self.marks is not None:                      # GPU ms per phase = gap between consecutive marks
            self._ev("rest")
            self._ev("stop")
            torch.cuda.synchronize()
            for (k, e), (_k2, e2) in zip(self.marks, self.marks[1:]):
                self.phases[k] = self.phases.get(k, 0.0) + e.elapsed_time(e2)
            self.marks = None
        if self.prof is not None and g == self.b + 1:
            n = self.b - self.a + 1
            tot = sum(self.phases.values())
            print(f"[profile] GPU timeline by phase (ms/step, event-to-event; total {tot / n:.1f}):", flush=True)
            for k, v in self.phases.items():
                print(f"    {v / n:7.2f}  {100 * v / tot:5.1f}%  {k}", flush=True)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - self.t0) / (self.b - self.a + 1) * 1000
            self.prof.__exit__(None, None, None)
            for h in self._hooks:
                h.remove()
            ev = self.prof.key_averages()
            n = self.b - self.a + 1
            self._by_module(n)
            from torch.autograd import DeviceType
            cpu_names = {e.key for e in ev if e.self_cpu_time_total > 0}  # a CUDA-side copy of a CPU range (e.g.
            kern = [e for e in self.prof.events() if e.device_type == DeviceType.CUDA   # Optimizer.step) is no kernel
                    and e.name not in cpu_names]
            gpu = sum(e.device_time for e in kern) / n / 1000
            print(f"[profile] steps {self.a}-{self.b}: wall {wall:.1f} ms/step, GPU kernels {gpu:.1f} ms/step "
                  f"({100 * gpu / wall:.0f}% busy, {wall - gpu:.1f} ms/step GPU idle)", flush=True)
            print("[profile] top CPU ops by self time (ms/step):", flush=True)
            for e in sorted(ev, key=lambda e: -e.self_cpu_time_total)[:25]:
                print(f"    {e.self_cpu_time_total / n / 1000:7.2f}  x{e.count / n:<6.1f} {e.key[:90]}", flush=True)
            print("[profile] top CUDA kernels (ms/step):", flush=True)
            agg = {}
            for e in kern:
                agg[e.name] = agg.get(e.name, 0.0) + e.device_time
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:25]:
                print(f"    {v / n / 1000:7.2f}  {k[:90]}", flush=True)
            self.prof = None
        if self.sites is not None and g == self.b + 2:
            import warnings
            torch.cuda.set_sync_debug_mode(0)
            warnings.showwarning = self._old
            print(f"[profile] host syncs in one step: {sum(self.sites.values())}", flush=True)
            for s_, c in self.sites.most_common(30):
                print(f"    {c:4d}x  {s_}", flush=True)
            self.sites = None


class StepTrace(Callback):
    """--trace FILE: one JSON line per training step (step, loss, grad norm, lr, and the step's look-ahead) -- the
    every-step record a reproducibility check compares (W&B only keeps every 50th). Appends, so a resumed run
    writes after the steps it continues from. Costs a host sync per step: tests only."""

    def __init__(self, path):
        self.path, self.gn = path, None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # fingerprints BEFORE the forward: which state differs first tells a non-exact resume's cause
        import hashlib
        import random
        import numpy as np

        def h(x):
            return hashlib.md5(repr(x).encode() if not isinstance(x, bytes) else x).hexdigest()[:10]
        self.fp = {
            "rng_py": h(random.getstate()), "rng_np": h(np.random.get_state()[1].tobytes()),
            "rng_torch": h(torch.get_rng_state().numpy().tobytes()),
            "rng_cuda": h(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all())),
            "w": h(b"".join(p.detach().float().cpu().numpy().tobytes() for p in pl_module.parameters())),
            "buf": {n: h(b.detach().float().cpu().numpy().tobytes()) for n, b in pl_module.named_buffers()},
            "batch": h(b"".join(t.cpu().numpy().tobytes() for t in batch if torch.is_tensor(t))[:4096 * 16]),
        }
        # module-level fingerprints for the steps named in TRACE_MODULES (e.g. "301,302"): each module's output hash
        # and the CUDA RNG right after it -> the first module whose output differs between two runs is the culprit
        want = {int(x) for x in os.environ.get("TRACE_MODULES", "").split(",") if x.strip()}
        for hk in getattr(self, "hooks", []):
            hk.remove()
        self.hooks, self.mods = [], []
        if trainer.global_step + 1 in want:
            def tens(o):
                if torch.is_tensor(o):
                    return [o]
                if isinstance(o, (list, tuple)):
                    return [t for x in o for t in tens(x)]
                if isinstance(o, dict):
                    return [t for x in o.values() for t in tens(x)]
                return []
            named = [("pre_encode", pl_module.encoder.pre_encode)]
            named += [(f"layer{i}", l) for i, l in enumerate(pl_module.encoder.layers)]
            named += [(n, getattr(pl_module, n)) for n in ("ctc_decoder", "decoder", "joint") if hasattr(pl_module, n)]
            for n, mod in named:
                def hook(m, inp, out, _n=n):
                    ts = tens(out)
                    self.mods.append((_n, h(b"".join(t.detach().float().cpu().numpy().tobytes() for t in ts)),
                                      h(b"".join(s.numpy().tobytes() for s in torch.cuda.get_rng_state_all()))))
                self.hooks.append(mod.register_forward_hook(hook))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        g = [p.grad.float().norm() for p in pl_module.parameters() if p.grad is not None]
        self.gn = torch.stack(g).norm().item() if g else 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        ctx = getattr(pl_module.encoder, "_fused_ctx", None)
        rec = {"step": trainer.global_step, "loss": float(loss), "grad_norm": self.gn,
               "lr": [g["lr"] for g in trainer.optimizers[0].param_groups][:2], "ctx": ctx[0] if ctx else None,
               "fp": getattr(self, "fp", None), "mods": getattr(self, "mods", None) or None}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class RngState(Callback):
    """Resume continues the RANDOM stream too: Python random (NeMo's per-batch look-ahead choice), NumPy, torch CPU
    (the fused dropout kernels draw their seeds from it) and CUDA generators ride in the checkpoint. Without this a
    resume reseeded from --seed, so every resume of a run replayed the same dropout masks / look-ahead draws (run6 vs
    run6r, Oct 11 2026: the two resumes tracked each other, not the uninterrupted run)."""

    def state_dict(self):
        import random
        import numpy as np
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()}

    # Restored at the FIRST training batch, not at checkpoint load: between the load and that batch Lightning builds
    # the dataloader iterators (each draws a base seed from torch's CPU generator) and runs the sanity validation,
    # which moved the stream a few draws ahead of the uninterrupted run (e2e_resume.sh: step 301 already differed).
    def load_state_dict(self, state):
        self.pending = state

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        state = getattr(self, "pending", None)
        if state is None:
            return
        import random
        import numpy as np
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])
        self.pending = None
        print(f"[run] resume: RNG states restored at step {trainer.global_step} (python / numpy / torch / cuda)", flush=True)


class StopAt(Callback):
    """--stop_step: end a (resumed) run at this global step WITHOUT changing max_steps, so the LR schedule is
    identical to the full run's (replay experiments)."""

    def __init__(self, step):
        self.step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step >= self.step:
            trainer.should_stop = True


class ParityDump(Callback):
    """--parity_dump PATH: after the FIRST backward (before clipping / the optimizer), save the loss and every grad, then
    exit. Run once without kernels, once with them, once --parity_fp32: voice/asr/parity_e2e.py compares the three."""

    def __init__(self, path):
        self.path, self.loss = path, None

    def wrap(self, m):
        orig = m.training_step

        def step(*args, **kw):
            out = orig(*args, **kw)
            self.loss = float((out["loss"] if isinstance(out, dict) else out).detach())
            return out
        m.training_step = step

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        g = {n: p.grad.detach().float().cpu() for n, p in pl_module.named_parameters() if p.grad is not None}
        torch.save({"loss": self.loss, "grads": g}, self.path)
        print(f"[parity] {self.path}: loss {self.loss:.6f}, {len(g)} grads", flush=True)
        os._exit(0)


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


def to_tdt(m, durations, sigma, omega):
    """RNN-T -> TDT partner head (Token-and-Duration Transducer): the joint gets len(durations) extra outputs (the
    duration logits, after blank), the loss becomes NeMo's TDT loss, greedy decoding TDT. change_vocabulary (--new_vocab)
    already rebuilt the joint and prediction net from scratch, so RNN-T and TDT arms start equally (only the joint's
    output layer is wider). With --fused_joint the loss runs in tkf kernels/sm120/tdt_joint.py (parity_tdt_joint.py:
    fp64 + NeMo TDT gates), else NeMo's numba TDT loss."""
    from omegaconf import open_dict
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    jc = m.joint.to_config_dict()
    with open_dict(jc):                                          # struct config: a new key needs open_dict
        jc["num_extra_outputs"] = len(durations)
    dev = next(m.encoder.parameters()).device
    m.joint = type(m).from_config_dict(jc).to(dev)
    kw = {"durations": list(durations), "sigma": sigma, "omega": omega, "fastemit_lambda": 0.0}
    with open_dict(m.cfg):
        m.cfg.joint = jc
        if "model_defaults" in m.cfg:
            m.cfg.model_defaults.tdt_durations = list(durations)
        m.cfg.loss.loss_name = "tdt"
        m.cfg.loss.tdt_kwargs = kw
        m.cfg.decoding.model_type = "tdt"
        m.cfg.decoding.durations = list(durations)
    red = getattr(m.loss, "reduction", "mean_batch")
    m.loss = RNNTLoss(num_classes=m.joint.num_classes_with_blank - 1 - len(durations), loss_name="tdt",
                      loss_kwargs=kw, reduction=red)
    m.change_decoding_strategy(m.cfg.decoding, decoder_type="rnnt")
    if m.joint.fuse_loss_wer:
        m.joint.set_loss(m.loss)
        m.joint.set_wer(m.wer)
    print(f"[run] partner head TDT: durations {list(durations)}, sigma {sigma}, omega {omega}; joint outputs "
          f"{m.joint.num_classes_with_blank} (vocab + blank + {len(durations)} durations), reduction {red}", flush=True)


OOD_LIKE = ("val_notsofar", "val_ami_sdm", "val_medical_room", "val_spotify_room", "val_svarah_room")


def bp_metrics(m, bp):
    """val_wer_ctcbp / val_wer_rnntbp: each head's greedy WER with `bp` subtracted from blank -- how the app decodes
    (blank penalty 1.0 won at every streaming look-ahead on the meeting clips, Oct 12 2026). The plain val_wer /
    val_wer_ctc and the validation losses are unchanged: each WER metric's update runs twice in validation, first
    with the penalty on (its num / den kept, then the metric reset), then plainly as NeMo expects. RNN-T: the penalty
    is a flag read by this model's joint (greedy decoding runs without CUDA graphs here, so the flag is honoured)."""
    # RNN-T greedy decoding captures a CUDA graph on its first call and replays it: the first call's penalty leaked
    # into the plain WER (test: plain == penalised == 21.48, 21.56 with the extra pass off). Graphs off for decoding;
    # change_decoding_strategy rebuilds m.wer, so this runs before the metrics are wrapped.
    from omegaconf import open_dict
    dec = m.cfg.decoding
    with open_dict(dec):
        dec.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(dec, decoder_type="rnnt")
    last, state = {}, {"bp": 0.0}
    cls = type(m.joint)                 # CLASS-level, as blank_penalty.py: greedy RNN-T decoding does not reach an
    jap = cls.joint_after_projection    # instance attribute (a test showed the penalised WER == the plain one)

    def joint_after_projection(self, f, g):
        out = jap(self, f, g)
        if state["bp"] and not self.training:
            out = out.clone()
            out[..., self.num_classes_with_blank - 1 - self.num_extra_outputs] -= state["bp"]   # blank (TDT: durations follow)
        return out
    cls.joint_after_projection = joint_after_projection

    def twice(metric, tag, penalise):
        upd = metric.update

        def update(*args, **kw):
            if not m.training:
                with torch.no_grad():
                    penalise(args, kw, True)
                    upd(*args, **kw)
                    _, last[f"num_{tag}"], last[f"den_{tag}"] = metric.compute()
                    metric.reset()
                    penalise(args, kw, False)
            return upd(*args, **kw)
        metric.update = update

    def rnnt_pen(args, kw, on):
        state["bp"] = bp if on else 0.0

    # CTC: penalise a copy of the log-probs for the extra pass
    cu = m.ctc_wer.update
    def ctc_update(*args, **kw):
        if m.training:
            return cu(*args, **kw)
        orig = kw["predictions"]
        p = orig.clone()
        p[..., -1] -= bp
        with torch.no_grad():
            cu(*args, **{**kw, "predictions": p})
            _, last["num_ctcbp"], last["den_ctcbp"] = m.ctc_wer.compute()
            m.ctc_wer.reset()
        return cu(*args, **kw)
    m.ctc_wer.update = ctc_update
    twice(m.wer, "rnntbp", rnnt_pen)
    vpass, vend = m.validation_pass, m.multi_validation_epoch_end

    def validation_pass(batch, batch_idx, dataloader_idx=0):
        last.clear()
        out = vpass(batch, batch_idx, dataloader_idx)
        if isinstance(out, dict):
            for tag in ("ctcbp", "rnntbp"):
                if f"num_{tag}" in last:
                    out[f"val_wer_num_{tag}"], out[f"val_wer_denom_{tag}"] = last[f"num_{tag}"], last[f"den_{tag}"]
        return out

    def multi_validation_epoch_end(outputs, dataloader_idx=0):
        res = vend(outputs, dataloader_idx)
        for tag in ("ctcbp", "rnntbp"):
            if outputs and all(isinstance(x, dict) and f"val_wer_num_{tag}" in x for x in outputs):
                num = torch.stack([x[f"val_wer_num_{tag}"] for x in outputs]).sum()
                den = torch.stack([x[f"val_wer_denom_{tag}"] for x in outputs]).sum()
                res.setdefault("log", {})[f"val_wer_{tag}"] = num.float() / den
        return res
    m.validation_pass, m.multi_validation_epoch_end = validation_pass, multi_validation_epoch_end
    print(f"[run] validation also reports both heads at blank penalty {bp} (val_wer_ctcbp / val_wer_rnntbp); "
          f"checkpoints select on the CTC one's aggregate", flush=True)


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
            trainer.callback_metrics["val_wer_ctcbp_all"] = torch.tensor(float("inf"))
            return
        out = {f"val_wer_{k}": e / n for k, (e, n) in acc.items() if n}
        for metric in ("val_wer", "val_wer_ctc", "val_wer_ctcbp", "val_wer_rnntbp"):   # each head, plain / penalised
            for name, stems in (("all", list(self.sets)), ("ood", [x for x in OOD_LIKE if x in self.sets])):
                got = [(float(cm[f"{x}_{metric}"]), self.sets[x][1]) for x in stems if f"{x}_{metric}" in cm]
                if got and len(got) == len(stems):
                    out[f"{metric}_{name}"] = sum(w * n for w, n in got) / sum(n for _, n in got)
        for k in ("val_wer_ctcbp_all",):                          # ModelCheckpoint raises if its monitor is missing
            out.setdefault(k, float("inf"))
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
    ap.add_argument("--init_weights", default=None, help="fine-tune: load every weight of this .nemo / Lightning .ckpt "
                    "(a run with the same --new_vocab / --ctc_head) over the freshly built model; fresh optimizer + schedule")
    ap.add_argument("--batch_sec", type=float, default=1200, help="real audio seconds per batch (Lhotse bucketing)")
    ap.add_argument("--buckets", type=int, default=30, help="Lhotse length buckets (more = less padding)")
    ap.add_argument("--sampler", choices=["packed", "lhotse"], default="packed",
                    help="packed = our exact bucketing sampler (packed_sampler.py); lhotse = NeMo's buffered one")
    ap.add_argument("--source_weight", nargs="*", default=[],
                    help="per-source mix weights SOURCE_GLOB:W, e.g. notsofar:3 'svarah*':4 librispeech:0.5 (fast loader;"
                         " rows repeated / subsampled, no new audio)")
    ap.add_argument("--loader", choices=["fast", "nemo"], default="fast",
                    help="fast = fast_loader.py (numpy index, exact batches, int16 pinned); nemo = NeMo/Lhotse loader")
    ap.add_argument("--total_hours", type=float, default=3200, help="audio hours to train on (~7 x 460 h)")
    ap.add_argument("--eval_hours", type=float, default=150,
                    help="evaluate every this many audio hours (must be < one data epoch); 0 = every data epoch")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=None, help="warm-up steps; default --warmup_frac of max_steps")
    ap.add_argument("--min_lr", type=float, default=0.0, help="final lr of the cosine / WSD decay, absolute (user, Oct 12 2026: decay to 0; runs before 83c1c58 ended at 1e-5 -- pass --min_lr 1e-5 to resume / reproduce one)")
    ap.add_argument("--warmup_frac", type=float, default=0.05, help="warm-up budget as a fraction of max_steps (5%%: user default from Oct 11 2026 -- a fixed 200 steps was 0.36%% of a 10-epoch run)")
    ap.add_argument("--sched", choices=["cosine", "wsd"], default="cosine",
                    help="wsd = linear warm-up, flat at --lr, linear decay over the last --decay_frac to --min_lr")
    ap.add_argument("--optim", choices=["adamw", "muown"], default="muown",
                    help="muown (default; beats AdamW -0.55 val WER over 2 epochs, 2 seeds, W&B asr-muown-vs-adamw-2ep) = tkf FusedMuon on the weight matrices + AdamW on the rest (muon_groups.py)")
    ap.add_argument("--muon_lr", type=float, default=None, help="Muon lr (scale adam: same band as AdamW); default --lr")
    ap.add_argument("--muon_variant", default="muown", help="FusedMuon variant: muown | aurora | normuon")
    ap.add_argument("--muon_heads", action="store_true", help="Muon on attention q/k/v/pos PER HEAD: (512,512) -> 8 x (64,512)")
    ap.add_argument("--muon_ns", default="ns8", help="Newton-Schulz preset: ns8 (6 quintic + 2 finishing, the BiBo board default) | ns6 | dsv4 (10)")
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--lookahead_probs", nargs="*", default=None,
                    help="training mix of the multi-lookahead contexts as RIGHT:PROB, right context in 80 ms frames, "
                         "e.g. 0:0.2 1:0.4 6:0.2 13:0.2 full:0.2 (full = the -1 look-ahead; NeMo default: uniform)")
    ap.add_argument("--lookaheads", type=int, nargs="*", default=None,
                    help="replace the trained look-ahead set: RIGHT contexts in 80 ms frames, e.g. 13 6 3 1 0 -1 "
                         "(-1 = full context; left context 70 kept)")
    ap.add_argument("--joint_dropout", type=float, default=0.1,
                    help="RNN-T / TDT joint dropout: 0.1 since Oct 12 2026 (user; pretrained config 0.2 -- drop01-fe0, "
                         "all dropout 0.1, tied the 0.2 baseline); --dropout_all, if given, sets it too")
    ap.add_argument("--muon_wd", type=float, default=None,
                    help="weight decay of the Muown (matrix) group; default = the AdamW group's 1e-3")
    ap.add_argument("--val_blank_penalty", type=float, default=1.0,
                    help="blank penalty of the extra validation metrics val_wer_ctcbp / val_wer_rnntbp (checkpoints select on the "
                         "CTC one); 0 = off (no extra decoding passes; checkpoints then select on val_wer_all)")
    ap.add_argument("--validate_only", action="store_true",
                    help="run validation on --ckpt and exit (with a new --run name and --no_hf)")
    ap.add_argument("--val_lookahead", type=int, default=3,
                    help="look-ahead (80 ms frames) validation runs at: 3 = 240 ms (default since Oct 11 2026; runs "
                         "before then validated at 13 = 1040 ms), -1 = full context")
    ap.add_argument("--fastemit", type=float, default=0.0,
                    help="FastEmit lambda: 0 since Oct 12 2026 (vs 0.01: validation tied, meeting CTC 1040 ms 19.66 vs "
                         "22.12, 480 ms 20.72 vs 21.70); the pretrained model's own default is 0.005")
    ap.add_argument("--aux", choices=["rnnt", "tdt"], default="rnnt", help="the CTC head's training partner")
    ap.add_argument("--tdt_durations", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tdt_sigma", type=float, default=0.05)
    ap.add_argument("--tdt_omega", type=float, default=0.1)
    ap.add_argument("--dropout_all", type=float, default=None,
                    help="set EVERY nn.Dropout (encoder residual / attention / pre-encoder / pos-emb, joint, prednet) to "
                         "this p; the fused kernels read p from the modules at each call (pretrained: 0.1, joint 0.2)")
    ap.add_argument("--save_steps", type=int, nargs="*", default=[],
                    help="also save a full checkpoint at these global steps (WSD: branch decays from the stable phase)")
    ap.add_argument("--aug_schedule", type=float, nargs="*", default=None,
                    help="per-epoch fraction of utterances augmented on the fly (aug_online.py), last value repeats, "
                         "e.g. 0 0.5 0.5 0.5 0.2")
    ap.add_argument("--grad_diag", type=int, default=0, help="log per-module grad norms + look-ahead every N steps")
    ap.add_argument("--log_every", type=int, default=50, help="Lightning log_every_n_steps (debug probes: 5)")
    ap.add_argument("--clip", type=float, default=1.0, help="global grad-norm clip (debug: late-training spikes)")
    ap.add_argument("--hold_lr_step", type=int, default=None, help="freeze the schedule's lr at this step (debug probes)")
    ap.add_argument("--stop_step", type=int, default=None, help="stop at this global step (schedule unchanged)")
    ap.add_argument("--parity_dump", default=None, help="save loss + grads of the first step and exit (parity_e2e.py); "
                    "use with --dropout_all 0 and a one-hot --lookahead_probs")
    ap.add_argument("--parity_fp32", action="store_true", help="with --parity_dump: fp32 reference (precision 32-true, "
                    "fp32 matmuls at 'highest', no autocast)")
    ap.add_argument("--profile_steps", default=None, help="A:B -- profile real steps A..B (+1 sync-debug step)")
    ap.add_argument("--lr_scale", type=float, default=None, help="multiply the scheduler lrs at train start (resumes)")
    ap.add_argument("--trace", default=None, help="per-step JSONL (loss, grad norm, lr, look-ahead) for reproducibility checks")
    ap.add_argument("--ckpt", default=None, help="resume from this checkpoint (e.g. a --save_steps one, for a decay branch)")
    ap.add_argument("--workers", type=int, default=None, help="loader workers (default: CPU count - 1)")
    ap.add_argument("--compile_layers", action="store_true")
    ap.add_argument("--fused_joint", action="store_true", help="tkf fused joint + RNN-T loss (voice/asr/fused_joint.py)")
    ap.add_argument("--no_hf", action="store_true", help="A/B and smoke runs: no HF checkpoint pull / push")
    ap.add_argument("--bf16_master", action="store_true", help="bf16 model + fp32 master weights, no autocast")
    ap.add_argument("--fused_layer", action="store_true", help="tkf residual+dropout+LayerNorm (voice/asr/fused_layer.py)")
    ap.add_argument("--fp32_residual", action="store_true", help="with --bf16_master --fused_layer: fp32 residual stream")
    ap.add_argument("--fused_attn", action="store_true", help="tkf banded rel-pos attention (voice/asr/fused_attn.py)")
    ap.add_argument("--fused_conv", action="store_true", help="tkf conv module + FFN silu/dropout (voice/asr/fused_conv.py)")
    ap.add_argument("--ctc_only", action="store_true", help="CTC-only training: no RNN-T decoder / joint / loss")
    ap.add_argument("--ctc_head", default=None, help="self-conditioned CTC head, readouts per pass: lin_glu, glu_glu, "
                    "lin_glu_glu ... (selfcond_head.py, fused training loss)")
    ap.add_argument("--fused_ctc", action="store_true", help="tkf deterministic CTC loss (voice/asr/fused_ctc.py)")
    ap.add_argument("--teacher", default=None, help="teacher_kd.py: a frozen transducer teacher (HF id or .nemo) on "
                    "the SAME tokenizer as the student (--tok = the teacher's pieces); every batch both student heads "
                    "learn --hard_weight x text loss + --kd_weight x KD")
    ap.add_argument("--teacher_ckpt", default=None, help="teacher weights over --teacher: a Lightning .ckpt path or hf:<repo>:<path>")
    ap.add_argument("--teacher_lookaheads", type=int, nargs="*", default=[13, 6, 3, 1, 0, -1],
                    help="the look-ahead set the teacher was fine-tuned with (its context follows the student's)")
    ap.add_argument("--kd_weight", type=float, default=0.7)
    ap.add_argument("--hard_weight", type=float, default=0.3)
    ap.add_argument("--kd_rnnt", type=float, default=None, help="RNN-T head KD weight (default --kd_weight; 0 = the "
                    "RNN-T head learns from the transcript only)")
    ap.add_argument("--hard_rnnt", type=float, default=None, help="RNN-T head text-loss weight (default --hard_weight; "
                    "0 = KD only, the alpha/beta lattice is skipped)")
    ap.add_argument("--kd_rnnt_mode", choices=["node", "fullsum", "pos"], default="node", help="RNN-T head KD: node = collapsed KL at every lattice node (matches the teacher's timing); fullsum = |student - teacher| -log P(Y|X) summed over all alignments (sequence level, timing-free; arXiv 2303.05958); pos = the teacher's full-vocabulary distribution of every token position (frames marginalised), matched at the student's own timing")
    ap.add_argument("--kd_ctc", type=float, default=None, help="CTC head KD weight (default --kd_weight)")
    ap.add_argument("--kd_ctc_mode", choices=["align", "pos"], default="align", help="CTC head KD: align = CE to the "
                    "teacher's one-token-per-frame alignment; pos = the teacher's per-token distributions at THIS "
                    "head's own CTC alignment (timing-free; needs --kd_rnnt_mode pos for the targets)")
    ap.add_argument("--hard_ctc", type=float, default=None, help="CTC head text-loss weight (default --hard_weight; "
                    "0 = KD only, the CTC lattice is skipped)")
    ap.add_argument("--teacher_bp", type=float, default=0.0, help="--teacher: blank penalty on the teacher's "
                    "distribution before the KD (its best decode setting, e.g. 1.0): the student does not inherit the "
                    "teacher's blank / deletion bias")
    ap.add_argument("--teacher_la", type=int, default=None, help="--teacher: FIXED teacher look-ahead in 80 ms frames "
                    "(-1 = full context); default: the student's sampled look-ahead of each batch")
    ap.add_argument("--kd_targets", default=None, help="per-token KD from DUMPED teacher targets (teacher_dump.py "
                    "shards: a local dir or hf:<repo>:<dir>) instead of a teacher forward every step; RNN-T head, "
                    "--kd_rnnt / --hard_rnnt weights; needs --loader fast")
    ap.add_argument("--kd_cache", action="store_true", help="--teacher, per-token mode, fixed --teacher_la: keep "
                    "each row's teacher targets in RAM after its first visit; later epochs skip the teacher forward")
    ap.add_argument("--no_mask_pieces", action="store_true", help="--teacher: do NOT mask the teacher's cased / "
                    "punctuated pieces our transcripts never produce")
    ap.add_argument("--ctc_weight", type=float, default=None, help="hybrid loss = (1-w) RNN-T + w CTC (model default 0.3)")
    ap.add_argument("--keep_vocab", action="store_true", help="hub --init (Nemotron, Parakeet): keep its own "
                    "tokenizer + trained decoder instead of switching to --tok")
    ap.add_argument("--new_vocab", action="store_true", help="with a .nemo --init: switch it to --tok (new CTC/RNN-T "
                    "output layers, encoder kept)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise same-seed runs (with --fused_ctc): torch deterministic algorithms, cuDNN deterministic")
    ap.add_argument("--project", default="bibo-asr")
    a = ap.parse_args()
    for h in ("rnnt", "ctc"):            # --teacher: per-head weights default to --kd_weight / --hard_weight
        if getattr(a, f"kd_{h}") is None:
            setattr(a, f"kd_{h}", a.kd_weight)
        if getattr(a, f"hard_{h}") is None:
            setattr(a, f"hard_{h}", a.hard_weight)
    if a.teacher:
        print(f"[run] teacher KD: RNN-T {a.hard_rnnt} x text + {a.kd_rnnt} x KD | CTC {a.hard_ctc} x text + {a.kd_ctc} x KD"
              f" (weight 0 = that loss is not computed)", flush=True)
    assert not a.fp32_residual or (a.bf16_master and a.fused_layer), "--fp32_residual needs --bf16_master --fused_layer"

    import nemo.collections.asr as nemo_asr
    torch.set_float32_matmul_precision(os.environ.get("PARITY_MATMUL") or ("highest" if a.parity_fp32 else "high"))   # PARITY_MATMUL=high: TF32 noise floor
    if a.deterministic:    # det_probe.py: without these the gradient bits still differ run to run (cuDNN / cuBLAS)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    # eval every eval_steps batches ACROSS data epochs (val_check_interval, check_val_every_n_epoch=None): the train
    # iterator keeps running through evals. (limit_train_batches=eval_steps ended a Lightning epoch at every eval and
    # rebuilt the iterator -> with a fixed shard_seed, the same batches every segment.) eval_hours 0 = every data epoch.
    eval_steps = int(a.eval_hours * 3600 / a.batch_sec) or None
    max_steps = int(a.total_hours * 3600 / a.batch_sec)
    if a.warmup is None:
        a.warmup = max(1, round(a.warmup_frac * max_steps))
    print(f"[run] warm-up {a.warmup} steps = {100 * a.warmup / max_steps:.1f}% of {max_steps}", flush=True)

    if a.seed is not None:
        pl.seed_everything(a.seed)
    if a.init.endswith(".nemo"):                       # a trained run (same vocabulary): A/B arms start identical
        m = nemo_asr.models.ASRModel.restore_from(a.init)
        if a.new_vocab:
            m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(a.init)
        if a.keep_vocab:                 # e.g. Nemotron / Parakeet: keep the pretrained tokenizer and decoder
            print(f"[run] {a.init}: keeping its own vocabulary ({m.tokenizer.vocab_size} pieces) and decoder", flush=True)
        else:
            m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")

    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=a.buckets,
                  batch_duration=a.batch_sec, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=a.workers or max(1, (os.cpu_count() or 2) - 1), shuffle_buffer_size=10000, seed=23 if a.seed is None else a.seed, pin_memory=True,
                  # same seed -> same batches (det_probe.py): NeMo's defaults draw per-worker seeds from the OS RNG
                  # (shard_seed "trng") and fill buckets from a background thread (timing-dependent batch contents)
                  shard_seed="randomized", concurrent_bucketing=False)
        for k in [k for k, v in tr.items() if v is None and k != "batch_size"]:   # hub configs: None where newer NeMo
            tr.pop(k)                                                           # wants an int (bucket_buffer_size)
    m.setup_training_data(tr)
    wrap_train_sampler(m, a)
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
        ctx = lambda r: [-1, -1] if r < 0 else [left, r]                   # -1 = full context (offline)
        m.encoder.att_context_size_all = [ctx(r) for r in a.lookaheads]
        m.encoder.att_context_probs = [1.0 / len(a.lookaheads)] * len(a.lookaheads)
        # validation (and the training default) context: --val_lookahead, 240 ms unless asked (user, Oct 11 2026)
        assert a.val_lookahead in a.lookaheads, f"--val_lookahead {a.val_lookahead} not in --lookaheads {a.lookaheads}"
        m.encoder.set_default_att_context_size(ctx(a.val_lookahead))
        with open_dict(m.cfg):                                       # saved .nemo / GGUF export see the new set
            # NeMo's constructor demands left % (r+1) == 0; the mask uses left // (r+1) chunks anyway, so
            # [70 - 70 % (r+1), r] is the SAME mask ([70,3] made run5.nemo unloadable for streaming eval)
            m.cfg.encoder.att_context_size = [[-1, -1] if r < 0 else [left - left % (r + 1), r] for r in a.lookaheads]
            m.cfg.encoder.att_context_probs = m.encoder.att_context_probs
        print(f"[run] look-ahead set {m.encoder.att_context_size_all}, validation at "
              f"{'full context' if a.val_lookahead < 0 else f'{a.val_lookahead * 80} ms'}", flush=True)
    if a.lookahead_probs:
        want = {(-1 if r == "full" else int(r)): float(p) for r, p in (x.split(":") for x in a.lookahead_probs)}
        ctxs = [list(c) for c in m.encoder.att_context_size_all]
        assert sorted(want) == sorted(c[1] for c in ctxs) and abs(sum(want.values()) - 1) < 1e-6, (want, ctxs)
        m.encoder.att_context_probs = [want[c[1]] for c in ctxs]       # sampled per batch (random.choices)
        with open_dict(m.cfg):
            m.cfg.encoder.att_context_probs = m.encoder.att_context_probs
        print(f"[run] lookahead mix {dict(('full' if c[1] < 0 else c[1] * 80, p) for c, p in zip(ctxs, m.encoder.att_context_probs))} (ms: prob)",
              flush=True)
    if a.fastemit is not None:                                       # fused_joint reads it from the loss config
        with open_dict(m.cfg):                                       # hub TDT models (Parakeet) carry tdt_kwargs
            key = "tdt_kwargs" if m.cfg.loss.get("loss_name") == "tdt" else "warprnnt_numba_kwargs"
            if m.cfg.loss.get(key) is None:
                m.cfg.loss[key] = {}
            m.cfg.loss[key].fastemit_lambda = a.fastemit
    if a.ctc_weight is not None and "aux_ctc" not in m.cfg:
        print(f"[run] --ctc_weight {a.ctc_weight} ignored: {type(m).__name__} has no CTC head", flush=True)
    elif a.ctc_weight is not None:
        m.ctc_loss_weight = a.ctc_weight
        with open_dict(m.cfg):
            m.cfg.aux_ctc.ctc_loss_weight = a.ctc_weight
    if a.aux == "tdt":                       # before the dropout settings and fused_joint: both act on the NEW joint
        to_tdt(m, a.tdt_durations, a.tdt_sigma, a.tdt_omega)
    if a.dropout_all is not None:
        n = 0
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p, n = a.dropout_all, n + 1
        print(f"[run] dropout_all: {n} Dropout modules -> p {a.dropout_all}", flush=True)
    if a.joint_dropout is not None and a.dropout_all is None:   # the joint's Dropout; before fused_joint (reads it)
        for mod in m.joint.joint_net:
            if isinstance(mod, torch.nn.Dropout):
                print(f"[run] RNN-T joint dropout {mod.p} -> {a.joint_dropout}", flush=True)
                mod.p = a.joint_dropout
    if (a.fused_joint or a.parity_dump) and hasattr(m, "joint"):   # parity: the eager run gets the same loss setup
        if not m.joint._fuse_loss_wer or m.loss.reduction != "mean_batch":   # hub models (Nemotron / Parakeet) ship
            print(f"[run] fused_joint: joint fuse_loss_wer {m.joint._fuse_loss_wer} -> True, loss reduction "   # with
                  f"{m.loss.reduction} -> mean_batch (as the 114M streaming model)", flush=True)   # other settings
            m.joint._fuse_loss_wer = True
            m.joint._fused_batch_size = m.joint._fused_batch_size or 2
            m.joint.set_loss(m.loss)
            m.joint.set_wer(m.wer)
            m.loss.reduction = "mean_batch"
            with open_dict(m.cfg):
                m.cfg.joint.fuse_loss_wer, m.cfg.joint.fused_batch_size = True, m.joint._fused_batch_size
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
    if a.ctc_only:                       # before ctc_head: selfcond_head wraps this training_step
        import ctc_only
        ctc_only.enable(m)
    if a.ctc_head:                       # after fused_ctc: the eval-time CTC loss still goes through it
        import selfcond_head
        if a.teacher:                    # the CTC head learns from the TEACHER's per-frame lattice targets
            assert a.fused_joint and a.fused_attn, "--teacher needs --fused_joint and --fused_attn"
            selfcond_head.enable(m, a.ctc_head, distill=a.kd_ctc, hard=a.hard_ctc, mode=a.kd_ctc_mode)
        else:
            selfcond_head.enable(m, a.ctc_head)
    if hasattr(m, "ctc_wer") and a.val_blank_penalty > 0:   # hybrid model: + both heads at the app's blank penalty
        bp_metrics(m, a.val_blank_penalty)
    if a.init_weights:                   # fine-tune from a trained run: the model is built as that run was (stock
        import tarfile                   # --init + --new_vocab + --ctc_head), then EVERY weight is loaded over it
        if a.init_weights.endswith(".nemo"):
            with tarfile.open(a.init_weights) as t:
                sd = torch.load(t.extractfile(next(n for n in t.getnames() if n.endswith("model_weights.ckpt"))),
                                map_location="cpu", weights_only=False)
        else:
            sd = torch.load(a.init_weights, map_location="cpu", weights_only=False)["state_dict"]
        miss, unexp = m.load_state_dict(sd, strict=False)
        params = {n for n, _ in m.named_parameters()}
        assert not [k for k in miss if k in params], f"--init_weights: parameters not in the checkpoint: {miss[:8]}"
        print(f"[run] init_weights {a.init_weights}: {len(sd)} tensors loaded, all {len(params)} parameters "
              f"(missing buffers {len(miss)}, unused {len(unexp)})", flush=True)
    if a.kd_targets:                     # per-token KD from dumped targets: no teacher model at all
        assert not a.teacher and a.fused_joint and a.loader == "fast", "--kd_targets: no --teacher, --fused_joint, fast loader"
        assert a.kd_ctc == 0, "--kd_targets: the CTC head trains on text (--kd_ctc 0)"
        import teacher_kd
        teacher_kd.enable_store(m, a.kd_targets, a.kd_rnnt, a.hard_rnnt, mask=not a.no_mask_pieces, manifest=a.train)
    if a.teacher:                        # after every student kernel / head / weight load: the teacher hooks the joint
        assert a.fused_joint and a.fused_attn, "--teacher needs --fused_joint and --fused_attn"
        import teacher_kd
        teacher_kd.enable(m, a.teacher, a.teacher_ckpt, a.teacher_lookaheads, a.kd_rnnt, a.hard_rnnt,
                          frame_k=1 if (a.ctc_head and a.kd_ctc > 0 and a.kd_ctc_mode == "align") else 0,
                          mask=not a.no_mask_pieces,
                          manifest=a.train, teacher_bp=a.teacher_bp, teacher_la=a.teacher_la,
                          rnnt_mode=a.kd_rnnt_mode, cache=a.kd_cache)
    # Exact resume: a 1-layer nn.LSTM ignores its `dropout` (applied only between layers), but with dropout > 0 the
    # first call per process makes cuDNN's dropout state, drawing a seed from the CUDA generator -- at step 1 in an
    # uninterrupted run, at the first step after a resume in a resumed one, so every later dropout mask differed
    # (e2e_resume.sh: the prediction net was the first module to diverge). Zero it: same outputs, no hidden draw.
    for mod in m.modules():
        if isinstance(mod, torch.nn.LSTM) and mod.num_layers == 1 and mod.dropout > 0:
            print(f"[run] 1-layer LSTM dropout {mod.dropout} -> 0 (it never applied; it only drew a hidden RNG seed)",
                  flush=True)
            mod.dropout = 0.0

    parity = None
    if a.parity_dump:
        m.spec_augmentation = None
        parity = ParityDump(a.parity_dump)
        parity.wrap(m)
        print(f"[parity] SpecAugment off; dump -> {a.parity_dump}", flush=True)
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
    # RNN-T-only models (Nemotron / Parakeet without our CTC head) select on the RNN-T WER
    MON = "val_wer_ctcbp_all" if a.val_blank_penalty > 0 and hasattr(m, "ctc_wer") else "val_wer_all"
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="32-true" if a.bf16_master or a.parity_fp32 else "bf16-mixed", max_steps=max_steps, max_epochs=-1,
        val_check_interval=eval_steps, check_val_every_n_epoch=None if eval_steps else 1,
        num_sanity_val_steps=0, gradient_clip_val=None if a.bf16_master else a.clip, log_every_n_steps=a.log_every,
        use_distributed_sampler=False, logger=logger, enable_checkpointing=True, benchmark=False,
        callbacks=[AudioMeter(), RngState(), EpochShuffle(a.aug_schedule, m), TQDMProgressBar(refresh_rate=50), ValAggregate(a.val), LearningRateMonitor("step"),
                   ModelCheckpoint(dirpath=ckpt_dir, monitor=MON, mode="min", save_top_k=2,
                                   save_last=True, filename="{step}-{" + MON + ":.4f}", save_on_train_epoch_end=False),
                   *([SaveSteps(a.save_steps, ckpt_dir)] if a.save_steps else []),
                   *([StepTrace(a.trace)] if a.trace else []),
                   *([GradDiag(a.grad_diag)] if a.grad_diag else []),
                   *([StopAt(a.stop_step)] if a.stop_step else []),
                   *([parity] if parity else []),
                   *([ProfileSteps(a.profile_steps)] if a.profile_steps else []),
                   *([LrScale(a.lr_scale)] if a.lr_scale else []),
                   *([] if a.no_hf else [HFSync(a.run, ckpt_dir)])])
    m.set_trainer(trainer)
    lightning_log = m.log

    def log(name, value, *args, **kw):   # Lightning turns a Python number into a GPU tensor: a blocking H2D copy per
        if isinstance(value, (int, float)) and not isinstance(value, bool):   # logged value per step (--profile_steps)
            value = torch.tensor(float(value))
        return lightning_log(name, value, *args, **kw)
    m.log = log
    m.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_steps": a.warmup, "min_lr": a.min_lr, "max_steps": max_steps}}))
    def fuse(opt):                       # NeMo builds AdamW with foreach/fused None: the per-parameter path cost
        if not a.bf16_master:            # 107.7 ms of CPU per step launching ~700 tensors' kernels (--profile_steps)
            for g in opt.param_groups:
                g["foreach"], g["fused"] = False, True
        return opt
    fuse(m._optimizer)
    nemo_configure = m.configure_optimizers

    def configure_optimizers():          # NeMo re-runs setup_optimization at fit time: fuse what it rebuilds
        out = nemo_configure()
        fuse(m._optimizer)
        return out
    m.configure_optimizers = configure_optimizers
    if a.optim == "muown":               # FusedMuon(muown) on the matrices + fused AdamW on the rest (muon_groups.py)
        assert not a.bf16_master, "--optim muown: autocast only"
        import muon_groups
        opt = muon_groups.build(m, a.lr, muon_lr=a.muon_lr, wd=1e-3, muon_wd=a.muon_wd, variant=a.muon_variant, ns=a.muon_ns,
                                heads=m.encoder.layers[0].self_attn.h if a.muon_heads else 0)
        sch = (torch.optim.lr_scheduler.LambdaLR(opt, wsd_lambda(a.warmup, max_steps, a.decay_frac, a.min_lr / a.lr))
               if a.sched == "wsd" else muon_groups.warmup_cosine(opt, a.warmup, max_steps, a.min_lr, hold=a.hold_lr_step))
        m._optimizer, m._scheduler = opt, sch
        m.configure_optimizers = lambda: ([opt], [{"scheduler": sch, "interval": "step", "frequency": 1}])
    if a.bf16_master:
        bf16_master.wrap(m._optimizer, clip=1.0)                    # clips on the fp32 masters
    if a.sched == "wsd" and a.optim != "muown":                     # (muown builds its own WSD above)
        for g in m._optimizer.param_groups:                        # NeMo's scheduler may have set a warm-up lr
            g["lr"] = a.lr
            g.pop("initial_lr", None)
        sched = torch.optim.lr_scheduler.LambdaLR(m._optimizer, wsd_lambda(a.warmup, max_steps, a.decay_frac, a.min_lr / a.lr))
        m._scheduler = {"scheduler": sched, "interval": "step", "frequency": 1}
        # NeMo's configure_optimizers() re-runs setup_optimization() at fit time, rebuilding the cosine scheduler
        # over ours (smoke test: lr followed the cosine). Hand Lightning this optimizer + scheduler directly.
        m.configure_optimizers = lambda: ([fuse(m._optimizer)], [m._scheduler])
    if a.sched == "wsd":
        print(f"[run] WSD: warm-up {a.warmup}, flat to step {int(max_steps * (1 - a.decay_frac))}, "
              f"linear decay to {max_steps}", flush=True)
    print(f"[run] {a.run}: max_steps {max_steps}, eval every {eval_steps} steps (~{a.eval_hours} h), "
          f"batch {a.batch_sec} s, val {[os.path.basename(v) for v in a.val]}", flush=True)
    print(f"[run] resume from {resume}", flush=True)
    if a.validate_only:     # re-score a finished checkpoint (e.g. at a new --val_lookahead) without training; use a
        assert resume, "--validate_only needs --ckpt"           # NEW --run name + --no_hf so nothing is overwritten
        trainer.validate(m, ckpt_path=resume)
        print("VALIDATE_DONE", flush=True)
        return
    trainer.fit(m, ckpt_path=resume)
    trainer.validate(m)                                         # the final model, wherever the last eval fell
    m.save_to(os.path.join(ckpt_dir, f"{a.run}.nemo"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
