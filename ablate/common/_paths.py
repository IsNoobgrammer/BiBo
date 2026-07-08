"""Path bootstrap: put the BiBo repo root and the sibling triton-kernel-fused root on sys.path.
Import this first from any ablate entrypoint so `src`, `baseline`, and `kernels` all resolve."""
import sys
from pathlib import Path

BIBO_ROOT = Path(__file__).resolve().parents[2]          # ablate/common/_paths.py -> BiBo/
TKF_ROOT = BIBO_ROOT.parent / "triton-kernel-fused"      # sibling repo with kernels/

for p in (str(BIBO_ROOT), str(TKF_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
