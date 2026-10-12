"""Validation sets: CTC greedy vs CTC beam search (pyctcdecode, no LM) for train_asr runs with a self-conditioned CTC
head, per set, at one look-ahead. CTC logits on the GPU (batched by duration), beam search on CPU workers.

    python voice/asr/val_beam.py --nemo X.nemo [Y.nemo] --val en3/val_*.jsonl [--la 3] [--bp 0 0.5 1.0] [--beam 32]

Prints one VALBEAM line per (model, set, decoder, bp): WER %, then per model the mean over sets and the OOD-like mean
(NOTSOFAR, AMI SDM, medical / Spotify / Svarah room: train_asr's group). WER is pooled word edits / reference words
after score.normalize (the data-prep normaliser), the same for greedy and beam, so the two compare exactly; it is not
NeMo's validation WER (different normaliser), so compare within this script only.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_meeting import load  # noqa: E402
from ctc_heads import encode, set_lookahead  # noqa: E402
from score import edits, normalize  # noqa: E402

OOD = ("notsofar", "sdm", "medical_room", "spotify_room", "svarah_room")
G = {}


def _init(labels):
    from pyctcdecode import build_ctcdecoder
    G["dec"] = build_ctcdecoder(labels)


def _beam(job):
    i, z, bp, beam = job
    z = z.astype(np.float32).copy()
    z[:, -1] -= bp
    lp = z - np.logaddexp.reduce(z, axis=1, keepdims=True)
    return i, bp, G["dec"].decode(lp, beam_width=beam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--head", default="glu_glu")
    ap.add_argument("--la", type=int, default=3, help="look-ahead in 80 ms steps (3 = 240 ms, the validation default)")
    ap.add_argument("--bp", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--batch_sec", type=float, default=400.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tok", default="/home/marimo/work/asr/en3/tok/tokenizer_spe_bpe_v2047/tokenizer.model")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import blank_penalty
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tok)
    labels = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    labels = ["⁇" if p == "<unk>" else p for p in labels] + [""]
    sets = []
    for f in sorted(a.val):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        sets.append((os.path.basename(f)[4:-6] if os.path.basename(f).startswith("val_") else os.path.basename(f), rows))
    print(f"[val_beam] {len(sets)} sets, {sum(len(r) for _, r in sets)} utterances, la {a.la * 80} ms, bp {a.bp}, "
          f"beam {a.beam}", flush=True)
    pool = Pool(a.workers, initializer=_init, initargs=(labels,))
    for path in a.nemo:
        m = load(path, a.head)
        name = os.path.basename(path)[:-5]
        set_lookahead(m.encoder, a.la)
        lock = blank_penalty.devanagari_ids(a.tok)
        blank = m.ctc_decoder.num_classes_with_blank - 1
        res = {}
        for sname, rows in sets:
            refs = [normalize(r["text"]) for r in rows]
            order = sorted(range(len(rows)), key=lambda i: rows[i].get("duration", 0))
            zs = [None] * len(rows)
            i = 0
            while i < len(order):
                j, tot = i, 0.0
                while j < len(order) and (j == i or tot + rows[order[j]].get("duration", 10) <= a.batch_sec):
                    tot += rows[order[j]].get("duration", 10); j += 1
                ids = order[i:j]
                xs = [sf.read(rows[k]["audio_filepath"], dtype="float32")[0] for k in ids]
                xs = [x.mean(1) if x.ndim > 1 else x for x in xs]
                L = max(len(x) for x in xs)
                audio = torch.stack([F.pad(torch.from_numpy(x), (0, L - len(x))) for x in xs]).cuda()
                e, el = encode(m, audio, torch.tensor([len(x) for x in xs]).cuda())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = m.ctc_decoder.head(e).float() + lock
                for b, k in enumerate(ids):
                    zs[k] = z[b, : int(el[b])].cpu().numpy()
                i = j
            tot_ref = sum(map(len, refs))
            for bp in a.bp:                                   # greedy
                errs = 0
                for z, ref in zip(zs, refs):
                    zz = z.copy(); zz[:, blank] -= bp
                    ids = zz.argmax(-1)
                    keep = (ids != blank) & (ids != np.concatenate([[-1], ids[:-1]]))
                    errs += sum(edits(ref, normalize(m.tokenizer.ids_to_text(ids[keep].tolist()))))
                res[(sname, "greedy", bp)] = 100 * errs / max(tot_ref, 1)
            jobs = [(k, zs[k], bp, a.beam) for bp in a.bp for k in range(len(zs))]
            errs = {bp: 0 for bp in a.bp}
            for k, bp, text in pool.imap_unordered(_beam, jobs, chunksize=4):
                errs[bp] += sum(edits(refs[k], normalize(text)))
            for bp in a.bp:
                res[(sname, "beam", bp)] = 100 * errs[bp] / max(tot_ref, 1)
            for dec in ("greedy", "beam"):
                print(f"VALBEAM {name:14s} {sname:22s} {dec:6s} " + " | ".join(
                    f"bp {bp:.2f} {res[(sname, dec, bp)]:6.2f}" for bp in a.bp) + f" | words {tot_ref}", flush=True)
        names = [s for s, _ in sets]
        ood = [s for s in names if any(o in s for o in OOD)]
        for dec in ("greedy", "beam"):
            for bp in a.bp:
                mean = sum(res[(s, dec, bp)] for s in names) / len(names)
                om = sum(res[(s, dec, bp)] for s in ood) / max(len(ood), 1)
                print(f"VALBEAM_MEAN {name:14s} {dec:6s} bp {bp:.2f} | mean {mean:6.2f} over {len(names)} sets | "
                      f"ood-like {om:6.2f} over {len(ood)} ({' '.join(ood)})", flush=True)
        del m
        torch.cuda.empty_cache()
    pool.close()
    print("VAL_BEAM_DONE", flush=True)


if __name__ == "__main__":
    main()
