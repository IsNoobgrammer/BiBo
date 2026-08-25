"""Re-upload a finished run with the diagnostics moved from `train/` to `interp/`.

    python -m ablate.tools.wandb_reorg entity/project/runid [--name NEW] [--dry]

W&B history is append-only: a logged key cannot be renamed in place. The only way to give an old
run the new sectioning is to read its history back and log it again under the new names, which is
what this does -- values, histograms and images all survive, so the copy is the same run with a
different table of contents.

The ORIGINAL IS NOT DELETED. Deleting it is a click in the UI once the copy looks right, and doing
it from a script would mean destroying the only source if the copy came out wrong.

Only the live-panel keys stay in `train/`; everything else train.py puts there is a diagnostic.
That list is duplicated from train.py's wb.log call on purpose -- this tool has to reproduce what
_interp() does at the sink for runs that predate it.
"""
import argparse
import os

import wandb

# Exactly the keys train.py logs OUTSIDE the `rt` diagnostics bundle.
CORE = {"loss", "grad_norm", "lr", "ms_per_step", "tps", "mfu", "mem_gb", "elapsed_s",
        "expert_corr", "router_corr", "loss_clean", "probe_gap", "probe_gamma"}


def remap(key):
    if not key.startswith("train/"):
        return key
    return key if key[6:] in CORE else "interp/" + key[6:]


def rebuild(v, media_root):
    """Turn a history value back into the wandb object it was logged as."""
    if not isinstance(v, dict):
        return v
    t = v.get("_type")
    if t == "histogram":
        # A history histogram carries values+bins; the same _type also appears as a bare summary
        # stub with neither, and wandb requires len(bins) == len(values) + 1.
        vals, bins = v.get("values"), v.get("bins")
        if not vals or not bins or len(bins) != len(vals) + 1:
            return None
        return wandb.Histogram(np_histogram=(vals, bins))
    if t in ("image-file", "images/separated"):
        p = os.path.join(media_root, v["path"])
        return wandb.Image(p) if os.path.exists(p) else None
    return None                      # anything else (tables, audio) is not used by this repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="entity/project/runid")
    ap.add_argument("--name", default=None, help="name for the copy (default: the original's)")
    ap.add_argument("--media", default="wandb_reorg_media", help="where to stage downloaded images")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    api = wandb.Api()
    src = api.run(args.run)
    rows = list(src.scan_history(page_size=1000))
    moved = sorted({k for r in rows for k in r if remap(k) != k})
    print(f"{src.name}\n  {len(rows)} steps, {len(rows[0])} keys, {len(moved)} moving to interp/")
    for k in moved[:5]:
        print("   ", k, "->", remap(k))
    if args.dry:
        return

    imgs = [f for f in src.files() if f.name.startswith("media/") and f.name.endswith(".png")]
    if imgs:
        print(f"  downloading {len(imgs)} images -> {args.media}")
        for f in imgs:
            if not os.path.exists(os.path.join(args.media, f.name)):
                f.download(root=args.media)

    entity, project, _ = args.run.split("/")
    dst = wandb.init(entity=entity, project=project, name=args.name or src.name,
                     config=dict(src.config), tags=list(src.tags) + ["reorg"],
                     notes=f"key-reorg copy of {src.id}: diagnostics moved train/ -> interp/")
    # finish() in a finally: without it an exception anywhere in the copy loop leaves the run
    # unsynced and the whole upload is lost, which is exactly what happened the first time.
    try:
        _copy(src, dst, rows, args.media)
    finally:
        print("  copy:", dst.url)
        dst.finish()


def _copy(src, dst, rows, media):
    for r in rows:
        step = r.get("_step")
        payload = {}
        for k, v in r.items():
            if k.startswith("_") or v is None:
                continue
            nv = rebuild(v, media)
            if nv is not None:
                payload[remap(k)] = nv
        dst.log(payload, step=step)
    # src.summary.items() raises on histogram entries (its subdict has no "bins" until fetched),
    # so go through the raw json and keep the scalars.
    dst.summary.update({remap(k): v for k, v in src.summary._json_dict.items()
                        if not k.startswith("_") and not isinstance(v, dict)})


if __name__ == "__main__":
    main()
