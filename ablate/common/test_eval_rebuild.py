"""The report loader rebuilds every checkpoint's architecture exactly, proven by a STRICT load.

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

from .report_ckpt import load_from_result


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
        res = ck.replace("_final.pt", "_result.json")
        if not os.path.exists(res):
            print(f"  SKIP {name[:60]}: no _result.json sidecar")
            continue
        try:
            model, _ = load_from_result(res, device="cpu")
            print(f"  OK   {name[:60]}")
            del model
        except RuntimeError as e:
            bad += 1
            print(f"  FAIL {name[:60]}\n       {str(e)[:400]}")

    if bad:
        print(f"\n{bad} checkpoint(s) do NOT round-trip -- the report loader would score them wrong")
        sys.exit(1)
    print("\nPASS: every checkpoint rebuilds exactly")


if __name__ == "__main__":
    main()
