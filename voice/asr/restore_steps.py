"""Restore a run's per-epoch step<N>.ckpt files from the HF checkpoint repo's history (a fresh box after a box death).

hf_sync pushes <run>/last.ckpt after every validation, one commit each ("<run> ckpt"); for a run validating once per
epoch every commit is an epoch checkpoint. Downloads each commit's last.ckpt, reads its global_step, and keeps it as
<dir>/step<N>.ckpt when N is wanted and missing.

    python voice/asr/restore_steps.py --run ep5-w10 --dir /home/marimo/work/asr/exp_long5/ep5-w10 --steps 5625 11250
"""
import argparse
import os
import shutil

import torch
from huggingface_hub import HfApi, hf_hub_download

REPO = "fhai50032/bibo-asr-ckpt"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--steps", type=int, nargs="+", required=True)
a = ap.parse_args()
os.makedirs(a.dir, exist_ok=True)
want = {s for s in a.steps if not os.path.exists(os.path.join(a.dir, f"step{s}.ckpt"))}
if want:
    tmp = os.path.join(a.dir, "_restore")
    for c in HfApi().list_repo_commits(REPO):
        if not want:
            break
        if c.title.strip() != f"{a.run} ckpt":
            continue
        try:
            f = hf_hub_download(REPO, f"{a.run}/last.ckpt", revision=c.commit_id, local_dir=tmp)
        except Exception as e:                                   # noqa: BLE001 -- a commit without the file
            print(f"[restore] {c.commit_id[:10]} skipped: {e}", flush=True)
            continue
        step = int(torch.load(f, map_location="cpu", weights_only=False, mmap=True)["global_step"])
        if step in want:
            shutil.move(f, os.path.join(a.dir, f"step{step}.ckpt"))
            want.discard(step)
            print(f"[restore] step{step}.ckpt from {c.commit_id[:10]}", flush=True)
        else:
            os.remove(f)
    shutil.rmtree(tmp, ignore_errors=True)
print(f"[restore] missing after restore: {sorted(want) or 'none'}", flush=True)
