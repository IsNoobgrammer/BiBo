"""Validation WER of saved A/B arms (exp_ab/<arm>/<arm>.nemo) on the per-source val sets + FLEURS, same settings as
train_asr's validation (batch 64, bf16 autocast, greedy RNN-T), weights cast to fp32 first so a bf16-master arm is
scored the same way as the others.

    python voice/asr/eval_ab.py --arms ab-fused-amp ab-fused-amp2 ab-fused-bf16 --val R/val_*.jsonl E/fleurs_*.jsonl
"""
import argparse
import json
import os
import sys

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--root", default="/home/marimo/work/asr/exp_ab")
    ap.add_argument("--fused", action="store_true", help="score with the tkf joint + layer kernels (faster)")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    words = {}
    for p in a.val:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        words[os.path.basename(p)[:-6]] = (rows[0].get("lang"), sum(len(r["text"].split()) for r in rows))
    out = {}
    for arm in a.arms:
        m = nemo_asr.models.ASRModel.restore_from(os.path.join(a.root, arm, f"{arm}.nemo")).float().cuda().eval()
        if a.fused:
            import fused_joint
            import fused_layer
            import fused_attn
            fused_joint.enable(m)
            fused_layer.enable(m)
            fused_attn.enable(m)
        va = OmegaConf.create(OmegaConf.to_container(m.cfg.validation_ds))
        with open_dict(va):
            va.update(manifest_filepath=a.val, batch_size=64, num_workers=4, shuffle=False)
        m.setup_multiple_validation_data(va)
        tr = pl.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", logger=False, enable_progress_bar=False,
                        enable_checkpointing=False)
        tr.validate(m, verbose=False)
        cm = {k: float(v) for k, v in tr.callback_metrics.items() if k.endswith("val_wer")}
        res = {}
        for stem in words:
            w = cm.get(f"{stem}_val_wer")
            if w is not None:
                res[stem] = w
        agg = {}
        for k in ("en", "hi"):
            num = sum(res[s] * words[s][1] for s in res if words[s][0] == k and s.startswith("val_") and s != "val_multispk")
            den = sum(words[s][1] for s in res if words[s][0] == k and s.startswith("val_") and s != "val_multispk")
            if den:
                agg[k] = num / den
        num = sum(res[s] * words[s][1] for s in res if s.startswith("val_") and s != "val_multispk")
        den = sum(words[s][1] for s in res if s.startswith("val_") and s != "val_multispk")
        agg["all"] = num / den
        out[arm] = (agg, res)
        print(f"{arm}: val_wer all {agg['all']:.4f}  en {agg.get('en', 0):.4f}  hi {agg.get('hi', 0):.4f}  |  " +
              "  ".join(f"{s.replace('val_', '')} {w:.4f}" for s, w in sorted(res.items())), flush=True)
        del m
        torch.cuda.empty_cache()
    print("EVAL_AB_DONE", flush=True)


if __name__ == "__main__":
    main()
