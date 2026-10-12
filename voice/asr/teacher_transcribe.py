"""Transcribe a whole training manifest with a stock hub model (default Parakeet TDT 0.6B v2, full context) for
(1) the data audit -- clips whose reference misses speech the teacher hears (deletions source, Oct 2026), and
(2) runv7's pseudo-labels / distillation.

Duration-sorted batches of up to --batch_sec audio, a DataLoader with workers decoding the FLACs, one bf16 encoder
forward + batched greedy decode per batch. Output: <out>/hyp_<shard>.tsv lines "row_index<TAB>hypothesis", appended per
batch (a dead box resumes: rows already written are skipped). Shard with --shard i --nshard n (rows i::n).

    python voice/asr/teacher_transcribe.py --manifest en3/train.jsonl --out en3/teacher_pk [--shard 0 --nshard 1]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

MODELS = {"parakeet": "nvidia/parakeet-tdt-0.6b-v2", "nemotron": "nvidia/nemotron-speech-streaming-en-0.6b"}


class Clips(torch.utils.data.Dataset):
    def __init__(self, batches, rows):
        self.batches, self.rows = batches, rows

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, j):
        ids = self.batches[j]
        wavs = []
        for i in ids:
            x, sr = sf.read(self.rows[i]["audio_filepath"], dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(1)
            assert sr == 16000, f"{self.rows[i]['audio_filepath']}: {sr} Hz"
            wavs.append(x)
        n = max(len(w) for w in wavs)
        a = np.zeros((len(wavs), n), np.float32)
        for k, w in enumerate(wavs):
            a[k, :len(w)] = w
        return ids, torch.from_numpy(a), torch.tensor([len(w) for w in wavs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="parakeet")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch_sec", type=float, default=1500.0)
    ap.add_argument("--max_dur", type=float, default=60.0, help="longer clips are skipped (reported)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    mine = [i for i in range(len(rows)) if i % a.nshard == a.shard]
    path = os.path.join(a.out, f"hyp_{a.shard}.tsv")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            done.add(int(l.split("\t", 1)[0]))
    todo = [i for i in mine if i not in done and rows[i]["duration"] <= a.max_dur]
    skipped = sum(rows[i]["duration"] > a.max_dur for i in mine)
    todo.sort(key=lambda i: rows[i]["duration"])
    batches, cur, tot = [], [], 0.0
    for i in todo:
        d = rows[i]["duration"]
        if cur and (tot + d > a.batch_sec or len(cur) >= 512):
            batches.append(cur)
            cur, tot = [], 0.0
        cur.append(i)
        tot += d
    if cur:
        batches.append(cur)
    hours = sum(rows[i]["duration"] for i in todo) / 3600
    print(f"[teacher] {a.model}: shard {a.shard}/{a.nshard}: {len(todo)} clips ({hours:.1f} h) in {len(batches)} batches; "
          f"{len(done)} already done; {skipped} over {a.max_dur} s skipped", flush=True)
    m = nemo_asr.models.ASRModel.from_pretrained(MODELS.get(a.model, a.model)).cuda().eval()
    dec = m.cfg.decoding
    with open_dict(dec):
        dec.strategy = "greedy_batch"
    m.change_decoding_strategy(dec)
    dl = torch.utils.data.DataLoader(Clips(batches, rows), batch_size=None, shuffle=False, num_workers=a.workers,
                                     prefetch_factor=4, persistent_workers=False)
    t0, n_s = time.time(), 0.0
    with open(path, "a", encoding="utf-8") as f:
        for j, (ids, audio, lens) in enumerate(dl):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc, enc_len = m.forward(input_signal=audio.cuda(non_blocking=True), input_signal_length=lens.cuda())
                hyps = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len)
            hyps = hyps[0] if isinstance(hyps, tuple) else hyps
            for i, h in zip(ids, hyps):
                f.write(f"{int(i)}\t{(h.text if hasattr(h, 'text') else str(h)).replace(chr(9), ' ').replace(chr(10), ' ')}\n")
            n_s += float(lens.sum()) / 16000
            if j % 50 == 0:
                f.flush()
                el = time.time() - t0
                print(f"[teacher] batch {j}/{len(batches)}: {n_s / 3600:.1f} h in {el / 60:.1f} min, RTFx {n_s / max(el, 1e-3):.0f}",
                      flush=True)
    print(f"[teacher] TEACHER_SHARD_DONE {a.shard} {n_s / 3600:.1f} h in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
