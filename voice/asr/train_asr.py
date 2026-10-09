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

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.every == 0:
            g = [p.grad.norm() for p in pl_module.parameters() if p.grad is not None]
            pl_module.log("grad_norm", torch.stack(g).norm().item() if g else 0.0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # samples summed ON the device: a .item() here every step drained the GPU queue (prof_step --sync_debug)
        s = batch[1].sum()
        self.win = s if self.win is None else self.win + s
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
            self.t0, self.win, self.win_steps = time.perf_counter(), None, 0


class HFSync(Callback):
    """After every evaluation (ModelCheckpoint has just written last.ckpt), push it to the private HF repo in a
    background thread, so a dead box loses at most one eval interval."""

    def __init__(self, run, ckpt_dir):
        self.run, self.ckpt_dir = run, ckpt_dir

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step > 0:
            hf_sync.push(self.run, ckpt=os.path.join(self.ckpt_dir, "last.ckpt"))


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
        out = {f"val_wer_{k}": e / n for k, (e, n) in acc.items() if n}
        for k, v in out.items():
            trainer.callback_metrics[k] = torch.tensor(v)
        trainer.logger.log_metrics(out, step=trainer.global_step)


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
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--compile_layers", action="store_true")
    ap.add_argument("--fused_joint", action="store_true", help="tkf fused joint + RNN-T loss (voice/asr/fused_joint.py)")
    ap.add_argument("--no_hf", action="store_true", help="A/B and smoke runs: no HF checkpoint pull / push")
    ap.add_argument("--bf16_master", action="store_true", help="bf16 model + fp32 master weights, no autocast")
    ap.add_argument("--fused_layer", action="store_true", help="tkf residual+dropout+LayerNorm (voice/asr/fused_layer.py)")
    ap.add_argument("--fp32_residual", action="store_true", help="with --bf16_master --fused_layer: fp32 residual stream")
    ap.add_argument("--fused_attn", action="store_true", help="tkf banded rel-pos attention (voice/asr/fused_attn.py)")
    ap.add_argument("--fused_conv", action="store_true", help="tkf conv module + FFN silu/dropout (voice/asr/fused_conv.py)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--project", default="bibo-asr")
    a = ap.parse_args()
    assert not a.fp32_residual or (a.bf16_master and a.fused_layer), "--fp32_residual needs --bf16_master --fused_layer"

    import nemo.collections.asr as nemo_asr
    torch.set_float32_matmul_precision("high")
    # eval_hours 0 = every data epoch. A Lightning "epoch" ends when the Lhotse iterator runs dry, and validation only
    # fires when batch_idx reaches limit_train_batches: an eval interval longer than the data epoch NEVER validates.
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

    ckpt_dir = os.path.join(a.out, a.run)
    os.makedirs(ckpt_dir, exist_ok=True)
    # resume: local last.ckpt, else the one in the HF repo (fresh box); the W&B run continues under the same id
    _, hf_ckpt, wid = (None, None, None) if a.no_hf else hf_sync.pull(a.run, os.path.dirname(a.out))
    resume = os.path.join(ckpt_dir, "last.ckpt") if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")) else hf_ckpt
    logger = RemapLogger(project=a.project, name=a.run, save_dir=ckpt_dir, id=wid, resume="allow",
                         config={**vars(a), "eval_steps": eval_steps, "max_steps": max_steps})
    if not a.no_hf:
        hf_sync.push(a.run, tok_dir=a.tok, wandb_id=logger.experiment.id, block=True)
    define_metrics(logger.experiment)
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="32-true" if a.bf16_master else "bf16-mixed", max_steps=max_steps, max_epochs=-1,
        limit_train_batches=eval_steps, num_sanity_val_steps=0, gradient_clip_val=None if a.bf16_master else 1.0, log_every_n_steps=50,
        use_distributed_sampler=False, logger=logger, enable_checkpointing=True, benchmark=False,
        callbacks=[AudioMeter(), TQDMProgressBar(refresh_rate=50), ValAggregate(a.val), LearningRateMonitor("step"),
                   ModelCheckpoint(dirpath=ckpt_dir, monitor="val_wer_all", mode="min", save_top_k=2,
                                   save_last=True, filename="{step}-{val_wer_all:.4f}", save_on_train_epoch_end=False),
                   *([] if a.no_hf else [HFSync(a.run, ckpt_dir)])])
    m.set_trainer(trainer)
    m.setup_optimization(OmegaConf.create({
        "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
        "sched": {"name": "CosineAnnealing", "warmup_steps": a.warmup, "min_lr": 1e-5, "max_steps": max_steps}}))
    if a.bf16_master:
        bf16_master.wrap(m._optimizer, clip=1.0)                    # clips on the fp32 masters
    print(f"[run] {a.run}: max_steps {max_steps}, eval every {eval_steps} steps (~{a.eval_hours} h), "
          f"batch {a.batch_sec} s, val {[os.path.basename(v) for v in a.val]}", flush=True)
    print(f"[run] resume from {resume}", flush=True)
    trainer.fit(m, ckpt_path=resume)
    trainer.validate(m)                                         # the final model, wherever the last eval fell
    m.save_to(os.path.join(ckpt_dir, f"{a.run}.nemo"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
