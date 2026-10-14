"""Save a depth-pruned copy of a NeMo conformer as a .nemo (the layers to keep, by original index), e.g. the greedy
order from prune_layers.py's stages.log. The saved config's n_layers matches, so restore_from / train_asr --init work.

    python voice/asr/prune_save.py --keep 0 1 2 4 5 7 8 15 17 18 19 20 21 22 23 --out nt-l15.nemo
    python voice/asr/prune_save.py --stages /home/marimo/work/asr/prune_nt/stages.log --layers 16 --out nt-l16.nemo
"""
import argparse
import ast
import re

import torch


def keep_from_stages(path, layers):
    for line in open(path):
        if re.match(rf"STAGE kept {layers} \(", line):
            return ast.literal_eval(line.split("| keep ")[1].strip())
    raise SystemExit(f"no 'STAGE kept {layers}' line in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    ap.add_argument("--keep", type=int, nargs="*", default=None)
    ap.add_argument("--stages", default=None, help="prune_layers.py stages.log: take the keep list of --layers")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep = a.keep if a.keep else keep_from_stages(a.stages, a.layers)
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    m = nemo_asr.models.ASRModel.from_pretrained(a.model, map_location="cpu")
    n0 = len(m.encoder.layers)
    m.encoder.layers = torch.nn.ModuleList([m.encoder.layers[j] for j in keep])
    m.encoder.n_layers = len(keep)
    if hasattr(m.encoder, "layer_drop_probs"):
        m.encoder.layer_drop_probs = m.encoder.layer_drop_probs[: len(keep)]
    with open_dict(m.cfg):
        m.cfg.encoder.n_layers = len(keep)
    m.save_to(a.out)
    r = nemo_asr.models.ASRModel.restore_from(a.out, map_location="cpu")   # round trip: same layers, same weights
    assert len(r.encoder.layers) == len(keep)
    for i, j in enumerate(keep):
        sa, sb = m.encoder.layers[i].state_dict(), r.encoder.layers[i].state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa), f"layer {j} changed in the round trip"
    print(f"PRUNE_SAVE {a.out}: {n0} -> {len(keep)} layers keep {keep}, "
          f"{sum(p.numel() for p in r.parameters()) / 1e6:.1f}M params", flush=True)


if __name__ == "__main__":
    main()
