"""Where does an ASR training step spend its time? Measures only -- the training pipeline (run1.sh) is untouched.

    python voice/asr/profile_train.py --manifest train.jsonl [--bs 64] [--steps 30]

Same model + settings as run1 (114M cache-aware FastConformer hybrid RNNT+CTC, vocab swapped to a 4k SPE-BPE, bf16
autocast, AdamW). Reports:
  1. dataloader alone: batches/s and audio-seconds/s with NeMo's own loader (num_workers as in run1)
  2. step breakdown (CUDA-synchronised wall time): data wait / forward / backward / optimizer, + audio-s per wall-s
  3. torch.profiler top CUDA kernels by self time over a few steps (what a Triton kernel would have to beat)
  4. padding waste: real audio seconds / padded seconds per batch
The tokenizer is a throwaway 4k BPE trained on the manifest text (same size as run1's).
"""
import argparse
import json
import os
import tempfile
import time

import torch


def tokenizer_dir(manifest, vocab=4096):
    import sentencepiece as spm
    d = tempfile.mkdtemp()
    txt = os.path.join(d, "text.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for l in open(manifest, encoding="utf-8"):
            f.write(json.loads(l)["text"].lower() + "\n")
    spm.SentencePieceTrainer.train(input=txt, model_prefix=os.path.join(d, "tokenizer"), vocab_size=vocab,
                                   model_type="bpe", character_coverage=1.0, byte_fallback=True,
                                   hard_vocab_limit=False, user_defined_symbols=["<spk1>", "<spk2>", "<spk3>", "<spk4>"])
    with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8") as f:
        sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "tokenizer.model"))
        f.writelines(sp.id_to_piece(i) + "\n" for i in range(sp.get_piece_size()))
    return d


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lhotse_sec", type=float, default=0, help="> 0: Lhotse duration-bucketed batches of this many audio seconds")
    ap.add_argument("--compile", choices=["none", "default", "ro", "layers"], default="none",
                    help="torch.compile the whole encoder (default / ro = CUDA graphs) or each conformer layer (layers)")
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    m = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_hybrid_large_streaming_multi")
    m.change_vocabulary(new_tokenizer_dir=tokenizer_dir(a.manifest), new_tokenizer_type="bpe")
    cfg = m.cfg.train_ds
    with open_dict(cfg):
        cfg.manifest_filepath, cfg.batch_size, cfg.num_workers = a.manifest, a.bs, a.workers
        cfg.max_duration, cfg.min_duration, cfg.shuffle, cfg.is_tarred = 30, 0.1, True, False
        cfg.pop("tarred_audio_filepaths", None)
        if a.lhotse_sec:
            cfg.use_lhotse, cfg.use_bucketing, cfg.num_buckets = True, True, 30
            cfg.batch_duration, cfg.batch_size, cfg.shuffle_buffer_size = a.lhotse_sec, None, 10000
    m.setup_training_data(cfg)
    m = m.cuda().train()
    if a.compile == "layers":
        # whole-encoder compile graph-breaks (random lookahead pick, NeMo's Triton subsampling) and then recompiles
        # for every new length; the 17 conformer layers have no breaks and share one dynamic-shape graph
        torch._dynamo.config.cache_size_limit = 64
        for i, layer in enumerate(m.encoder.layers):
            m.encoder.layers[i] = torch.compile(layer, dynamic=True)
    elif a.compile != "none":
        m.encoder = torch.compile(m.encoder, mode="reduce-overhead" if a.compile == "ro" else None, dynamic=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    m._optimizer = opt                                            # training_step logs its lr
    dl = m._train_dl
    print(f"params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M  vocab {m.tokenizer.vocab_size}", flush=True)

    # 1. dataloader alone
    it, t0, secs, pad_secs = iter(dl), time.perf_counter(), 0.0, 0.0
    for i in range(a.steps):
        b = next(it)
        secs += b[1].sum().item() / 16000
        pad_secs += b[0].shape[0] * b[0].shape[1] / 16000
    dt = time.perf_counter() - t0
    print(f"[loader] {a.steps / dt:.2f} batches/s, {secs / dt:.0f} audio-s/s, padding waste {100 * (1 - secs / pad_secs):.0f}%",
          flush=True)

    # 2. step breakdown
    def step(b):
        sig, sig_len, tgt, tgt_len = [x.cuda(non_blocking=True) for x in b[:4]]
        t1 = sync()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.training_step((sig, sig_len, tgt, tgt_len), 0)
        loss = out["loss"] if isinstance(out, dict) else out
        t2 = sync()
        loss.backward()
        t3 = sync()
        opt.step()
        opt.zero_grad(set_to_none=True)
        t4 = sync()
        return t1, t2, t3, t4, sig_len.sum().item() / 16000

    import types
    m.log = lambda *x, **k: None                                  # no Lightning trainer attached
    m.log_dict = lambda *x, **k: None
    m._trainer = types.SimpleNamespace(global_step=1, log_every_n_steps=10**9, current_epoch=0)  # never logs WER
    it = iter(dl)
    for _ in range(3 if a.compile == "none" else 15):             # warm-up (numba JIT, cudnn, compile shapes)
        step(next(it))
    tot = {"data": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
    secs, t_start = 0.0, sync()
    for _ in range(a.steps):
        t0 = sync()
        b = next(it)
        t1, t2, t3, t4, s = step(b)
        tot["data"] += t1 - t0
        tot["fwd"] += t2 - t1
        tot["bwd"] += t3 - t2
        tot["opt"] += t4 - t3
        secs += s
    wall = sync() - t_start
    print("[step] " + "  ".join(f"{k} {1000 * v / a.steps:.0f} ms" for k, v in tot.items())
          + f"  | {wall / a.steps * 1000:.0f} ms/step, {secs / wall:.0f} audio-s/s, peak mem "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

    # 3. top kernels
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            step(next(it))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=70), flush=True)


if __name__ == "__main__":
    main()
