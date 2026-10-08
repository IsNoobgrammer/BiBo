"""W&B layout for BiBo voice (ASR): ONE place that decides where every logged number lives, plus the saved workspace.
Mirrors ablate/common/wb_layout.py (BiBo text): key names are the contract between training and the dashboard.

Sections (first path segment):

    core/      train loss (rnnt / ctc / total), lr, audio hours seen, val WER all / en / hi (word-weighted)
    val/       per-source speaker-held-out val: val/<source>/{wer, wer_ctc, loss}; val/multispk = <spk> windows
    eval/      public anchors: eval/fleurs_en, eval/fleurs_hi  ({wer, wer_ctc, loss})
    speed/     audio-seconds per wall-second, ms/step, peak memory
    optim/     lr, grad norm
    misc/      anything new that no rule claims yet -- visible, never dropped

NeMo logs a validation set as "<manifest stem>_<metric>" (val_kathbath_hi_val_wer); wb_key turns that into
val/kathbath_hi/wer. Build the workspace (local terminal, `pip install wandb-workspaces`):

    python voice/asr/wb_layout.py --project bibo-asr
"""
import re

_EXACT = {
    "train_loss": "core/loss",
    "train_rnnt_loss": "core/loss_rnnt",
    "train_ctc_loss": "core/loss_ctc",
    "learning_rate": "optim/lr",
    "lr-AdamW": "optim/lr",
    "training_batch_wer": "core/train_batch_wer",
    "training_batch_wer_ctc": "core/train_batch_wer_ctc",
    "audio_hours_seen": "core/audio_hours",
    "audio_s_per_s": "speed/audio_s_per_s",
    "ms_per_step": "speed/ms_per_step",
    "mem_gb": "speed/mem_gb",
    "grad_norm": "optim/grad_norm",
    "val_wer_all": "core/val_wer",
    "val_wer_en": "core/val_wer_en",
    "val_wer_hi": "core/val_wer_hi",
    "global_step": "misc/global_step",
    "epoch": "misc/eval_interval",
}
# NeMo per-dataloader metric suffixes -> our metric names
_SUFFIX = {"val_wer": "wer", "val_wer_ctc": "wer_ctc", "val_loss": "loss", "val_rnnt_loss": "loss_rnnt",
           "val_ctc_loss": "loss_ctc"}
_KEEP = ("core/", "val/", "eval/", "speed/", "optim/", "misc/")
_SET = re.compile(r"^(val_|fleurs_)?(.+?)_(val_wer_ctc|val_wer|val_loss|val_rnnt_loss|val_ctc_loss)$")


def wb_key(k):
    if k in _EXACT:
        return _EXACT[k]
    if k.startswith(_KEEP):
        return k
    m = _SET.match(k)
    if m:
        prefix, name, metric = m.groups()
        if prefix == "fleurs_":
            return f"eval/fleurs_{name}/{_SUFFIX[metric]}"
        return f"val/{name}/{_SUFFIX[metric]}"
    return "misc/" + k


def wb_keys(d):
    return {wb_key(k): v for k, v in d.items()}


def define_metrics(run):
    for k in ("core/val_wer", "core/val_wer_en", "core/val_wer_hi", "eval/fleurs_en/wer", "eval/fleurs_hi/wer"):
        run.define_metric(k, summary="min,last")
    run.define_metric("speed/audio_s_per_s", summary="mean,last")
    run.define_metric("speed/mem_gb", summary="max")


# ---------------------------------------------------------------- workspace

def build_workspace(entity, project, name="BiBo voice board"):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    def line(title, y=None, regex=None, x="core/audio_hours", **kw):
        if regex is not None:
            return wr.LinePlot(title=title, metric_regex=regex, x=x, **kw)
        return wr.LinePlot(title=title, y=[y] if isinstance(y, str) else y, x=x, **kw)

    def sec(name, panels, open_=False, cols=3):
        return ws.Section(name=name, panels=panels, is_open=open_,
                          layout_settings=ws.SectionLayoutSettings(columns=cols, rows=2))

    overview = sec("Overview (x = audio hours seen)", [
        line("Val WER all / en / hi (RNNT)", ["core/val_wer", "core/val_wer_en", "core/val_wer_hi"]),
        line("FLEURS WER en / hi (RNNT)", ["eval/fleurs_en/wer", "eval/fleurs_hi/wer"]),
        line("Train loss (RNNT / CTC)", ["core/loss_rnnt", "core/loss_ctc"], smoothing_factor=0.8,
             smoothing_type="exponential", smoothing_show_original=True),
        line("Learning rate", "optim/lr"),
        line("Throughput (audio s / s)", "speed/audio_s_per_s"),
        line("Train-batch WER (RNNT / CTC)", ["core/train_batch_wer", "core/train_batch_wer_ctc"]),
    ], open_=True)
    val = sec("Val WER per source (speaker-held-out)", [
        line("RNNT WER, every source", regex=r"val/[^/]+/wer$"),
        line("CTC WER, every source", regex=r"val/[^/]+/wer_ctc$"),
        line("Hindi sources", regex=r"val/(indicvoices_hi|vaani_hi|kathbath_hi|hinglish|lahaja)/wer$"),
        line("English sources", regex=r"val/(svarah|peoples_speech|voxpopuli|ami_ihm)/wer$"),
        line("Accent sets (Svarah en, Lahaja hi)", regex=r"val/(svarah|lahaja)/wer$"),
        line("Multi-speaker windows (WER incl. <spk> tokens)", regex=r"val/multispk/wer.*"),
        line("Val loss per source", regex=r"val/[^/]+/loss$"),
    ], open_=True)
    ev = sec("Public eval (FLEURS)", [
        line("FLEURS en (RNNT / CTC)", ["eval/fleurs_en/wer", "eval/fleurs_en/wer_ctc"]),
        line("FLEURS hi (RNNT / CTC)", ["eval/fleurs_hi/wer", "eval/fleurs_hi/wer_ctc"]),
    ])
    speed = sec("Speed and memory", [
        line("Audio seconds / wall second", "speed/audio_s_per_s"),
        line("ms / step", "speed/ms_per_step"),
        line("Peak memory (GB)", "speed/mem_gb"),
        line("Audio hours vs step", "core/audio_hours", x="Step"),
    ])
    optim = sec("Optimizer", [line("Learning rate", "optim/lr"), line("Grad norm", "optim/grad_norm", log_y=True)])
    misc = sec("Misc (unmapped keys)", [line("misc/*", regex=r"misc/.*")])
    w = ws.Workspace(entity=entity, project=project, name=name,
                     sections=[overview, val, ev, speed, optim, misc])
    w.save()
    return w.url


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="bibo-asr")
    ap.add_argument("--entity", default=None)
    a = ap.parse_args()
    import wandb
    print(build_workspace(a.entity or wandb.Api().default_entity, a.project))
else:
    assert wb_key("val_kathbath_hi_val_wer") == "val/kathbath_hi/wer"
    assert wb_key("fleurs_hi_val_wer_ctc") == "eval/fleurs_hi/wer_ctc"
    assert wb_key("val_multispk_val_wer") == "val/multispk/wer"
    assert wb_key("train_loss") == "core/loss" and wb_key("weird") == "misc/weird"
