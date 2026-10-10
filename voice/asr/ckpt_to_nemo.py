"""Rebuild a run's .nemo from its Lightning checkpoint (HF bibo-asr-ckpt <run>/last.ckpt) on a fresh box.

    python voice/asr/ckpt_to_nemo.py --ckpt asr/hf_ckpt/run5/last.ckpt --tok asr/run1/tok/tokenizer_spe_bpe_v4096 \
        --lookaheads 13 6 3 1 0 --out asr/exp/run5/run5.eval.nemo

Same base model / tokenizer / look-ahead config as train_asr.py; the saved look-ahead set is the loadable
[left - left % (r+1), r] form (= nemo_ctx_fix.py), so the result is directly usable for streaming eval and --init.
"""
import argparse
import os

import torch
from omegaconf import open_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="stt_en_fastconformer_hybrid_large_streaming_multi")
    ap.add_argument("--lookaheads", type=int, nargs="*", default=None)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(a.init, map_location="cpu")
    m.change_vocabulary(new_tokenizer_dir=a.tok, new_tokenizer_type="bpe")
    if a.lookaheads:
        left = m.encoder.att_context_size_all[0][0]
        m.encoder.att_context_size_all = [[left, r] for r in a.lookaheads]
        m.encoder.set_default_att_context_size([left, a.lookaheads[0]])
        with open_dict(m.cfg):
            m.cfg.encoder.att_context_size = [[left - left % (r + 1), r] for r in a.lookaheads]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    m.save_to(a.out)
    print(f"[ckpt_to_nemo] step {ck['global_step']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
