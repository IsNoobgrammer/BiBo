"""run_eval rebuilds every checkpoint's architecture exactly, proven by a STRICT load.

run_eval.py used to call build_arm() with no architecture kwargs, then load with strict=False. A
64-expert SWA+XSA checkpoint therefore loaded into a 6-expert global model with every mismatched
key dropped silently -- no error, no warning, and bpb numbers that looked entirely plausible. The
fix reads the architecture back out of the run's own _result.json; this asserts the fix works
against the real checkpoints on disk rather than a synthetic one.

A strict load is the whole test: it fails on a missing key, an extra key, OR a shape mismatch, so
it catches every way the rebuild could be subtly wrong.

CPU-only, no GPU needed. Scans ../runs for <name>_final.pt + <name>_result.json pairs:
    python -m ablate.common.test_eval_rebuild
"""
from . import _paths  # noqa: F401

import os
import sys
import glob
import json
import torch

from .models import build_arm
from .run_eval import _arch_kwargs, _sidecar
from . import patches as patchmod


def main():
    runs = os.path.join(os.path.dirname(__file__), "..", "runs")
    ckpts = sorted(glob.glob(os.path.join(runs, "*_final.pt")))
    if not ckpts:
        print(f"no checkpoints under {os.path.abspath(runs)} -- nothing to check")
        return
    print(f"{len(ckpts)} checkpoint(s)")

    bad = 0
    for ck in ckpts:
        name = os.path.basename(ck)
        res = _sidecar(ck)
        if res is None:
            print(f"  SKIP {name[:60]}: no _result.json sidecar")
            continue
        saved = json.load(open(res))["config"]
        kw = _arch_kwargs(saved)
        patchmod.RADIAL_P = saved.get("radial_p", "sigmoid")
        patchmod.EXPERT_ACT = saved.get("act", "radial")
        model, _ = build_arm(saved.get("arm", "bibo_min"), device="cpu",
                             dtype=torch.float32, **kw)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(sd, strict=True)
            print(f"  OK   {name[:60]}  act={patchmod.EXPERT_ACT} experts={kw['num_experts']} "
                  f"window={kw['sliding_window']} qkn={kw['swa_qk_norm']}")
        except RuntimeError as e:
            bad += 1
            print(f"  FAIL {name[:60]}\n       {str(e)[:400]}")
        del model, sd

    if bad:
        print(f"\n{bad} checkpoint(s) do NOT round-trip -- run_eval would score them wrong")
        sys.exit(1)
    print("\nPASS: every checkpoint rebuilds exactly")


if __name__ == "__main__":
    main()
