"""Dump self-conditioned CTC-head LOGITS (fp16 .npy, [T, V], blank last) per utterance, for an external decoder
(RT Captions' Rust beam search, tools/ctc-decode in the app repo) -- validation sets at one look-ahead and / or the
meeting clips at several. No penalty, no script lock: the decoder applies its own (the app subtracts the blank
penalty after log-softmax).

    python voice/asr/dump_ctc.py --nemo X.nemo --out DIR [--val en3/val_*.jsonl --la 3] [--meeting --las 0 1 3 6 13 -1]

Writes DIR/<set>/<i>.npy, DIR/list.txt (one .npy path per line, for `ctc-decode @list.txt`), DIR/refs.jsonl
({"file", "set", "ref"} with the raw reference text) and DIR/pieces.txt (the tokenizer's pieces, one per line; the
blank is implicit, last). Score the decoder output with score_ctc_decode.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_meeting import load  # noqa: E402
from ctc_heads import MEETING, encode, set_lookahead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="glu_glu")
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--la", type=int, default=3, help="validation look-ahead in 80 ms steps (3 = 240 ms)")
    ap.add_argument("--meeting", action="store_true", help="also the 2 meeting recordings at every --las")
    ap.add_argument("--las", type=int, nargs="+", default=[0, 1, 3, 6, 13, -1])
    ap.add_argument("--meeting_dir", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--batch_sec", type=float, default=400.0)
    ap.add_argument("--tok", default="/home/marimo/work/asr/en3/tok/tokenizer_spe_bpe_v2047/tokenizer.model")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tok)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pieces.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sp.id_to_piece(i) for i in range(sp.get_piece_size())) + "\n")
    m = load(a.nemo, a.head)
    lst = open(os.path.join(a.out, "list.txt"), "w", encoding="utf-8")
    refs = open(os.path.join(a.out, "refs.jsonl"), "w", encoding="utf-8")

    def run(sname, items, la):                     # items: [(audio float32, ref text)]
        d = os.path.join(a.out, sname); os.makedirs(d, exist_ok=True)
        set_lookahead(m.encoder, la)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        i = 0
        while i < len(order):
            j, tot = i, 0.0
            while j < len(order) and (j == i or tot + len(items[order[j]][0]) / 16000 <= a.batch_sec):
                tot += len(items[order[j]][0]) / 16000; j += 1
            ids = order[i:j]
            xs = [items[k][0] for k in ids]
            L = max(len(x) for x in xs)
            audio = torch.stack([F.pad(torch.from_numpy(x), (0, L - len(x))) for x in xs]).cuda()
            e, el = encode(m, audio, torch.tensor([len(x) for x in xs]).cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m.ctc_decoder.head(e).float()
            for b, k in enumerate(ids):
                p = os.path.join(d, f"{k}.npy")
                np.save(p, z[b, : int(el[b])].half().cpu().numpy())
                lst.write(p + "\n")
                refs.write(json.dumps({"file": p, "set": sname, "ref": items[k][1]}) + "\n")
            i = j
        print(f"[dump_ctc] {sname}: {len(items)} utterances", flush=True)

    for f in sorted(a.val):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        items = []
        for r in rows:
            x = sf.read(r["audio_filepath"], dtype="float32")[0]
            items.append((x.mean(1) if x.ndim > 1 else x, r["text"]))
        b = os.path.basename(f)
        run(b[4:-6] if b.startswith("val_") else b[:-6], items, a.la)
    if a.meeting:
        clips = [w for w in ("clip16k.wav", "g5_16k.wav")]
        items = []
        for w, r in MEETING:
            if w in clips:
                x = sf.read(os.path.join(a.meeting_dir, w), dtype="float32")[0]
                items.append((x.mean(1) if x.ndim > 1 else x, open(os.path.join(a.meeting_dir, r), encoding="utf-8").read()))
        for la in a.las:
            run(f"meeting_la{'full' if la < 0 else la * 80}", items, la)
    lst.close(); refs.close()
    print("DUMP_CTC_DONE", flush=True)


if __name__ == "__main__":
    main()
