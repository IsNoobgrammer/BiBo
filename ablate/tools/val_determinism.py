"""Is the VAL forward bit-reproducible across processes? (training is; val drifted in the 4th decimal)

    python -m ablate.tools.val_determinism <result.json> [--drop PATCH[,PATCH]] [--repeat 3]

Loads the checkpoint the way train.py builds the model (same patches, same attention flags), runs the
exact val call train.py makes (validation.losses on the frozen holdout, bf16 autocast) `repeat` times,
and prints full-precision losses plus a fingerprint of the last hidden state. Run it in two separate
processes and diff the lines: in-process repeats test run-to-run determinism, across processes tests
per-process decisions (autotune timing). --drop removes patches to bisect which kernel causes it.
"""
from ablate.common import _paths  # noqa: F401
import argparse
import importlib
import json
import tempfile

import torch

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from kernels.sm120.cross_entropy import fused_linear_cross_entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--drop", default="")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    res = json.load(open(a.result))
    c = res["config"]
    drop = {x for x in a.drop.split(",") if x}
    if drop:
        c["patches"] = ",".join(p for p in c["patches"].split(",") if p.strip() not in drop)
        f = tempfile.NamedTemporaryFile("w", suffix="_result.json", delete=False)
        json.dump(res, f)
        f.close()
        a.result = f.name
    # the two module flags train.py sets from its args (report_ckpt does not)
    importlib.import_module("src.modeling.attn.full_attention").GLOBAL_FLEX = c.get("global_attn") == "flex"
    importlib.import_module("src.modeling.attn.base").FUSED_ATTN = c.get("attn_kernel") == "fused"
    model, cfg = load_from_result(a.result)
    hold = _val.build_holdout(cfg.dataset, cfg.seq_len, cfg.val_seqs, "cuda")
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    pad = 0 if c.get("pad_id") is None else int(c["pad_id"])
    vals = [_val.losses(model, hold, None, fused_linear_cross_entropy, amp, pad_id=pad)[0]
            for _ in range(a.repeat)]
    with torch.no_grad(), amp:
        h = model.model(input_ids=hold[:, :-1], use_cache=False).last_hidden_state.float()
    print(f"[valdet] patches={c['patches']}")
    print(f"[valdet] VALS {[v.hex() for v in vals]}  ({vals[0]!r})")
    print(f"[valdet] HIDDEN sum={h.sum().item().hex()} absmax={h.abs().max().item().hex()}")


if __name__ == "__main__":
    main()
