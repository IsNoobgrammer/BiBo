"""Offline per-token teacher targets for distillation (train_asr --kd_rnnt_mode pos without the teacher in the step).

For every manifest row: its transcript tokens (sp.encode(text) with the student's SentencePiece = the teacher's, exactly
as fast_loader tokenises the labels) and, for every token u, the teacher's distribution over the NON-blank vocabulary
with the frames marginalised by its own emission posteriors (tkf rnnt_joint_node_logp(pos_targets=True)): frame-free,
so the teacher may run at any look-ahead (default full context, its best) whatever the student's.
Stored as the top-K entries per token (idx int16, prob float16); the rest mass is 1 - sum.

Shards of --shard consecutive manifest rows: <out>/shard_<first row>.npz with
    row (n,) int64 manifest line | U (n,) int32 | off (n,) int64 into the token arrays | idx (sum U, K) int16 |
    p (sum U, K) float16 | tok (sum U,) int16 the transcript tokens (a join check for the trainer)
Resumable (finished shards are skipped, locally or on HF); --hf_repo uploads each shard as <hf_dir>/shard_*.npz.

    python voice/asr/teacher_dump.py --manifest en3/train_cleanA.jsonl --tok tok_nemotron/tokenizer.model \
        --ckpt hf:fhai50032/bibo-asr-ckpt:hub-nt-rnnt-cleanA-3ep/last.ckpt --out kd/nt3ep_full --la -1 \
        --hf_repo fhai50032/bibo-asr-ckpt --hf_dir kd_targets/nt3ep_full
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TKF = os.environ.get("TKF", "/home/marimo/work/triton-kernel-fused")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tok", required=True, help="the student's tokenizer.model (must equal the teacher's)")
    ap.add_argument("--init", default="nvidia/nemotron-speech-streaming-en-0.6b")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lookaheads", type=int, nargs="*", default=[13, 6, 3, 1, 0, -1])
    ap.add_argument("--la", type=int, default=-1, help="teacher look-ahead in 80 ms frames (-1 = full context)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--shard", type=int, default=4096)
    ap.add_argument("--batch_sec", type=float, default=900.0)
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--hf_dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (a probe)")
    a = ap.parse_args()
    sys.path.insert(0, TKF)
    import sentencepiece as spm
    import teacher_kd
    from kernels.sm120.rnnt_joint import rnnt_joint_node_logp

    rows = [json.loads(line) for line in open(a.manifest, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    sp = spm.SentencePieceProcessor(model_file=a.tok)
    t = teacher_kd.load_teacher(a.init, a.ckpt, a.lookaheads, fused=True)
    probe = "the quick brown fox jumps over the lazy dog's medical appointment on tuesday"
    assert sp.encode(probe) == t.tokenizer.text_to_ids(probe), "student and teacher tokenizers differ"
    t.encoder.set_default_att_context_size([-1, -1] if a.la < 0 else [t.encoder.att_context_size_all[0][0], a.la])
    blank = t.tokenizer.vocab_size
    lin = t.joint.joint_net[-1]
    os.makedirs(a.out, exist_ok=True)
    done_hf = set()
    api = None
    if a.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        done_hf = {os.path.basename(f) for f in api.list_repo_files(a.hf_repo) if f.startswith(a.hf_dir + "/")}
    pool = ThreadPoolExecutor(16)
    t0, n_done, h_done = time.time(), 0, 0.0
    for s0 in range(0, len(rows), a.shard):
        name = f"shard_{s0:08d}.npz"
        path = os.path.join(a.out, name)
        if name in done_hf or os.path.exists(path):
            continue
        part = list(range(s0, min(s0 + a.shard, len(rows))))
        part.sort(key=lambda i: rows[i]["duration"])
        toks = {i: sp.encode(rows[i]["text"]) for i in part}
        res = {}
        batch, sec = [], 0.0
        batches = []
        for i in part:
            if batch and sec + rows[i]["duration"] > a.batch_sec:
                batches.append(batch)
                batch, sec = [], 0.0
            batch.append(i)
            sec += rows[i]["duration"]
        if batch:
            batches.append(batch)
        for bt in batches:
            wavs = list(pool.map(lambda i: sf.read(rows[i]["audio_filepath"], dtype="float32")[0], bt))
            L = torch.tensor([len(w) for w in wavs])
            sig = torch.zeros(len(wavs), int(L.max()))
            for j, w in enumerate(wavs):
                sig[j, :len(w)] = torch.from_numpy(w)
            ys = [toks[i] for i in bt]
            ylen = torch.tensor([len(y) for y in ys])
            Y = torch.zeros(len(ys), max(1, int(ylen.max())), dtype=torch.long)
            for j, y in enumerate(ys):
                if y:
                    Y[j, :len(y)] = torch.tensor(y)
            sig, L, Y, ylen = sig.cuda(), L.cuda(), Y.cuda(), ylen.cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc, elen = t.forward(input_signal=sig, input_signal_length=L)
                dec, _, _ = t.decoder(targets=Y, target_length=ylen)
                f = t.joint.project_encoder(enc.transpose(1, 2))
                g = t.joint.project_prednet(dec.transpose(1, 2))
            *_, P = rnnt_joint_node_logp(f, g, lin.weight, lin.bias, Y, elen, ylen, pos_targets=True)
            P[..., blank] = 0.0
            pk, ik = P.topk(a.k, dim=-1)
            pk, ik = pk.cpu(), ik.cpu()
            for j, i in enumerate(bt):
                u = len(ys[j])
                res[i] = (ik[j, :u].numpy().astype(np.int16), pk[j, :u].numpy().astype(np.float16))
        order = sorted(res)
        U = np.array([len(toks[i]) for i in order], dtype=np.int32)
        off = np.concatenate([[0], np.cumsum(U)[:-1]]).astype(np.int64)
        idx = np.concatenate([res[i][0] for i in order]) if U.sum() else np.zeros((0, a.k), np.int16)
        p = np.concatenate([res[i][1] for i in order]) if U.sum() else np.zeros((0, a.k), np.float16)
        tok = np.concatenate([np.array(toks[i], dtype=np.int16) for i in order]) if U.sum() else np.zeros(0, np.int16)
        np.savez(path, row=np.array(order, dtype=np.int64), U=U, off=off, idx=idx, p=p, tok=tok)
        if api is not None:
            for k in range(5):
                try:
                    api.upload_file(path_or_fileobj=path, path_in_repo=f"{a.hf_dir}/{name}", repo_id=a.hf_repo,
                                    commit_message=f"kd targets {a.hf_dir} {name}")
                    break
                except Exception as e:
                    print(f"[dump] upload retry {k}: {e}", flush=True)
                    time.sleep(30)
        n_done += len(order)
        h_done += sum(rows[i]["duration"] for i in order) / 3600
        mass = float(p.astype(np.float32).sum(1).mean()) if len(p) else 0.0
        el = time.time() - t0
        print(f"DUMP_SHARD {name} rows {len(order)} tokens {int(U.sum())} | top-{a.k} mass {mass:.4f} | "
              f"{n_done} rows {h_done:.1f} h in {el / 60:.1f} min ({h_done * 3600 / max(el, 1):.0f}x real time)",
              flush=True)
    print("DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
