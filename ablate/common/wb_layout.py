"""W&B layout: ONE place that decides where every logged number lives, plus the saved workspace
that reads it.

The two halves are in one file on purpose. The key names are the contract between the training
loop and the dashboard; kept apart, a renamed key silently empties a chart.

Sections (the first path segment is the W&B section):

    core/      the basic panel: loss, smoothed loss, val loss, grad norm, lr, tokens
    val/       per-source validation (ppl, ext/*)
    speed/     tps, mfu, ms/step, peak memory, wall time
    optim/     per-optimizer lr, weight decay, Muon variant phase, probe
    health/    per-tensor grad / param norms, smallest grad norm (inert-tensor alarm)
    router/    routing mechanics, model-wide and per layer
    moe/       cross-expert redundancy
    xsa/ attnres/ act/ typed/   per-feature interpretability
    final/     end-of-run report (context ablation, degeneration, samples)
    misc/      anything new that no rule claims yet -- visible, never dropped

Build the workspace (local terminal, needs `pip install wandb-workspaces`):

    python -m ablate.common.wb_layout --project bibo-aurora-vs-muown
"""

# exact renames first (the headline numbers), then prefix rules in order, first match wins
_EXACT = {
    "train/loss": "core/loss",
    "train/loss_smooth": "core/loss_smooth",
    "val/loss": "core/val_loss",
    "train/grad_norm": "core/grad_norm",
    "train/lr": "core/lr",
    "tokens": "core/tokens",
    "train/tps": "speed/tps",
    "train/mfu": "speed/mfu",
    "train/ms_per_step": "speed/ms_per_step",
    "train/mem_gb": "speed/mem_gb",
    "train/elapsed_s": "speed/elapsed_s",
    "train/expert_corr": "moe/expert_corr",
    "train/router_corr": "router/router_corr",
    "grad/norm_min_over_tensors": "health/grad_norm_min",
    "samples": "final/samples",
}
_PREFIX = [
    ("train/loss_clean", "optim/loss_clean"),
    ("train/probe_", "optim/probe_"),
    ("grad/norm/", "health/grad_norm/"),
    ("params/norm/", "health/param_norm/"),
    ("interp/router/", "router/"),
    ("interp/router_", "router/"),
    ("interp/balance_entropy", "router/balance_entropy"),
    ("interp/max_expert_load", "router/max_expert_load"),
    ("interp/special_load", "router/special_load"),
    ("interp/neg_identity_load", "router/neg_identity_load"),
    ("interp/xsa/", "xsa/"),
    ("interp/xsa_a_", "xsa/alpha_"),
    ("interp/attn_res_", "attnres/"),
    ("interp/radial_p", "act/radial_p"),
    ("interp/act_alpha_", "act/theta_"),
    ("interp/typed/", "typed/"),
    ("ctxabl/", "final/ctx/"),
    ("degen/", "final/degen/"),
    ("interp/", "misc/"),
    ("train/", "misc/"),
]
_KEEP = ("core/", "val/", "speed/", "optim/", "health/", "router/", "moe/", "xsa/", "attnres/",
         "act/", "typed/", "final/", "misc/")


def wb_key(k):
    if k in _EXACT:
        return _EXACT[k]
    if k.startswith(_KEEP):
        return k
    for old, new in _PREFIX:
        if k.startswith(old):
            return new + k[len(old):]
    return "misc/" + k


def wb_keys(d):
    return {wb_key(k): v for k, v in d.items()}


def define_metrics(wb):
    """Run-table summaries: best AND last for the numbers two runs are ranked on."""
    for k in ("core/loss", "core/loss_smooth", "core/val_loss"):
        wb.define_metric(k, summary="min,last")
    for k in ("speed/tps", "speed/mfu"):
        wb.define_metric(k, summary="mean,last")
    wb.define_metric("speed/mem_gb", summary="max")


# ---------------------------------------------------------------- workspace

# per-layer families: one chart per metric, every layer a line (regex), instead of 10 panels each
_ROUTER_PER_LAYER = ["balance_entropy", "routing_entropy", "weight_entropy", "max_load",
                     "dead_experts", "boundary_gap", "max_weight", "score_max", "score_mean",
                     "load_balancing_loss", "router_z_loss", "weight_over_load_max"]


def build_workspace(entity, project, name="BiBo board"):
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    def line(title, y=None, regex=None, **kw):
        if regex is not None:
            return wr.LinePlot(title=title, metric_regex=regex, **kw)
        return wr.LinePlot(title=title, y=[y] if isinstance(y, str) else y, **kw)

    def sec(name, panels, open_=False, cols=3):
        return ws.Section(name=name, panels=panels, is_open=open_,
                          layout_settings=ws.SectionLayoutSettings(columns=cols, rows=2))

    overview = sec("Overview", [
        line("Train loss (20-step mean)", "core/loss_smooth", title_y="loss"),
        line("Val loss (frozen batch)", "core/val_loss", title_y="loss"),
        line("Train loss (raw, smoothed)", "core/loss", smoothing_factor=0.8,
             smoothing_type="exponential", smoothing_show_original=True),
        line("Grad norm (global, pre-clip)", "core/grad_norm", log_y=True),
        line("Learning rate (Muon / AdamW)", ["optim/lr_muon", "optim/lr_adamw"]),
        line("Throughput (tokens/s)", "speed/tps"),
    ], open_=True)
    seeds = sec("Seed mean +- range (grouped by run group)", [
        line("Train loss (20-step mean)", "core/loss_smooth", groupby="group",
             groupby_aggfunc="mean", groupby_rangefunc="minmax"),
        line("Val loss", "core/val_loss", groupby="group",
             groupby_aggfunc="mean", groupby_rangefunc="minmax"),
        line("Grad norm", "core/grad_norm", groupby="group", log_y=True,
             groupby_aggfunc="mean", groupby_rangefunc="minmax"),
    ], open_=True)
    loss = sec("Loss detail", [
        line("Train loss, log y", "core/loss_smooth", log_y=True),
        line("Train loss vs tokens", "core/loss_smooth", x="core/tokens"),
        line("Val perplexity", "val/ppl"),
        line("Val loss per source", regex=r"val/ext/.*"),
    ])
    speed = sec("Speed and memory", [
        line("Tokens / s", "speed/tps"),
        line("MFU %", "speed/mfu"),
        line("ms / step", "speed/ms_per_step"),
        line("Peak memory (GB)", "speed/mem_gb"),
        line("Wall time (s)", "speed/elapsed_s"),
    ])
    optim = sec("Optimizer", [
        line("Muon lr", "optim/lr_muon"),
        line("AdamW lr", "optim/lr_adamw"),
        line("Muon weight decay", "optim/wd_muon"),
        line("Muon variant phase (0 = start, 1 = after switch)", "optim/phase"),
        line("Probe gap / gamma", regex=r"optim/probe_.*"),
    ])
    health = sec("Gradient and weight health", [
        line("Smallest per-tensor grad norm (0 = inert tensor)", "health/grad_norm_min", log_y=True),
        line("Grad norm per tensor group", regex=r"health/grad_norm/.*", log_y=True),
        line("Param norm per tensor group", regex=r"health/param_norm/.*", log_y=True),
    ])
    router = sec("Router and MoE", [
        line("Balance entropy (1 = flat load)", "router/balance_entropy"),
        line("Routing entropy", "router/router_entropy"),
        line("Top-1 weight", "router/router_top1_weight"),
        line("Boundary gap (k vs k+1 score)", "router/router_boundary_gap"),
        line("Router direction collapse", "router/router_corr"),
        line("Cross-expert redundancy", "moe/expert_corr"),
        line("Max expert load", "router/max_expert_load"),
        line("Special-expert load", regex=r"router/(special_load|neg_identity_load)"),
    ])
    per_layer = sec("Router per layer (one line per layer)", [
        line(m.replace("_", " "), regex=rf"router/layer_\d+/{m}") for m in _ROUTER_PER_LAYER])
    attn = sec("XSA and AttnRes", [
        line("XSA alpha mean/min/max", regex=r"xsa/alpha_(mean|min|max)"),
        line("XSA alpha per layer", regex=r"xsa/layer_\d+/mean"),
        line("AttnRes carry c per layer", regex=r"attnres/s/L\d+"),
        line("AttnRes carry c spread per layer", regex=r"attnres/s_std/L\d+"),
        line("AttnRes carry c mean/min/max", regex=r"attnres/s_(mean|min|max)"),
        line("AttnRes emb d per layer", regex=r"attnres/d/L\d+"),
    ])
    act = sec("Activation (radial p)", [
        line("Radial p mean/min/max", regex=r"act/radial_p_(mean|min|max)"),
        line("Radial p per layer", regex=r"act/radial_p/layer_\d+/mean"),
        line("Radial theta mean/min/max", regex=r"act/theta_.*"),
    ])
    final = sec("Final report", [
        line("Context ablation", regex=r"final/ctx/.*"),
        line("Degeneration", regex=r"final/degen/.*"),
    ])
    misc = sec("Unsorted (new keys land here)", [line("misc", regex=r"misc/.*")])

    w = ws.Workspace(
        entity=entity, project=project, name=name,
        sections=[overview, seeds, loss, speed, optim, health, router, per_layer, attn, act,
                  final, misc],
        settings=ws.WorkspaceSettings(x_axis="Step", max_runs=20, group_by_prefix="first",
                                      tooltip_number_of_runs="all_runs"),
    )
    return w.save_as_new_view()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--entity", default="ablations-tinycompany-ai")
    ap.add_argument("--project")
    ap.add_argument("--name", default="BiBo board")
    a = ap.parse_args()
    if a.selftest:
        assert wb_key("train/loss") == "core/loss"
        assert wb_key("interp/router/layer_3/max_load") == "router/layer_3/max_load"
        assert wb_key("interp/attn_res_s/L0") == "attnres/s/L0"
        assert wb_key("interp/radial_p/layer_2/mean") == "act/radial_p/layer_2/mean"
        assert wb_key("grad/norm/layers.self_attn.q_proj") == "health/grad_norm/layers.self_attn.q_proj"
        assert wb_key("val/ext/en") == "val/ext/en" and wb_key("core/lr") == "core/lr"
        assert wb_key("something_new") == "misc/something_new"
        assert wb_key(wb_key("interp/xsa_a_mean")) == "xsa/alpha_mean"    # idempotent
        print("wb_layout selftest ok")
    else:
        print(build_workspace(a.entity, a.project, a.name).url.replace("\\", "/"))
