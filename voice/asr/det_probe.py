"""Where does same-seed run-to-run nondeterminism come from? Run twice with the same seed and diff the prints.

    python voice/asr/det_probe.py --train R/train.jsonl --tok TOK [--seed 23] [--fused]

Prints hashes of: the initial weights (after change_vocabulary), the first 3 Lhotse batches (lengths + audio), then the
loss and grad norm of 3 training steps on ONE fixed batch in train mode (dropout etc.) -- each layer of the stack.
"""
import argparse
import hashlib
import os
import sys

import torch
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def h(t):
    return hashlib.sha1(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--fused", action="store_true")
    a = ap.parse_args()
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(a.seed)
    m = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_hybrid_large_streaming_multi")
    m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
    print("init weights", h(torch.cat([p.flatten() for p in m.parameters()])), flush=True)
    tr = OmegaConf.create(OmegaConf.to_container(m.cfg.train_ds))
    with open_dict(tr):
        tr.pop("tarred_audio_filepaths", None)
        tr.update(manifest_filepath=a.train, is_tarred=False, use_lhotse=True, use_bucketing=True, num_buckets=30,
                  batch_duration=1200, batch_size=None, max_duration=30, min_duration=0.1, shuffle=True,
                  num_workers=a.workers, shuffle_buffer_size=10000, seed=23, pin_memory=True)
    m.setup_training_data(tr)
    it = iter(m._train_dl)
    batches = [next(it) for _ in range(3)]
    for i, b in enumerate(batches):
        print(f"batch {i}: {tuple(b[0].shape)} lens {h(b[1])} audio {h(b[0])} tokens {h(b[2])}", flush=True)
    m = m.cuda().train()
    if a.fused:
        import fused_attn
        import fused_conv
        import fused_joint
        import fused_layer
        fused_joint.enable(m)
        fused_layer.enable(m)
        fused_attn.enable(m)
        fused_conv.enable(m)
    b = [x.cuda() if torch.is_tensor(x) else x for x in batches[0]]
    torch.manual_seed(a.seed)
    for s in range(3):
        m.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.training_step(b, s)
        loss = out["loss"] if isinstance(out, dict) else out
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        print(f"step {s}: loss {loss.item():.6f}  grad {h(g)}  |g| {g.norm().item():.6f}", flush=True)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc, _ = m.forward(input_signal=b[0], input_signal_length=b[1])
    print("eval encoder out", h(enc), flush=True)


if __name__ == "__main__":
    main()
