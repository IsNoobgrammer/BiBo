"""Greedy depth pruning of a NeMo conformer (Nemotron streaming 0.6B: 24 layers x 25.2M): at each stage drop, one at a
time, every remaining layer, score streaming WER (RNN-T greedy, 240 ms look-ahead) + per-token RNN-T NLL on a few val
sets and the meeting clips, and remove the layer whose loss hurts the mean WER least. No retraining between stages.

Several worker processes share one stage: worker w scores candidates w, w+N, ...; each writes <out>/s<k>_l<i>.json;
every worker waits for the whole stage and picks the same winner (deterministic), so there is no master. Worker 0
logs each stage to W&B (layers_kept on x: WER per set, delta vs the unpruned teacher and vs the previous stage).

    for w in 0 1 2 3; do python voice/asr/prune_layers.py --worker $w --workers 4 --out /home/marimo/work/asr/prune_nt & done
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctc_heads import encode, set_lookahead  # noqa: E402
from score import edits, normalize_v2  # noqa: E402

MEET = (("clip16k.wav", "reference.txt"), ("g5_16k.wav", "g5_ref_gemini.txt"))


def load_sets(a):
    sets = {}
    for s in a.sets:
        rows = [json.loads(l) for l in open(f"{a.val_dir}/val_{s}.jsonl")]
        rows = [r for r in rows if r["text"].strip()]
        random.Random(23).shuffle(rows)
        rows = sorted(rows[: a.n], key=lambda r: r["duration"])
        sets[s] = [(sf.read(r["audio_filepath"], dtype="float32")[0], r["text"]) for r in rows]
    sets["meeting"] = [(sf.read(os.path.join(a.meeting, w), dtype="float32")[0],
                        open(os.path.join(a.meeting, r), encoding="utf-8").read()) for w, r in MEET]
    return sets


def batches(items, sec):
    b, t = [], 0.0
    for x in items:
        if b and t + len(x[0]) / 16000 > sec:
            yield b
            b, t = [], 0.0
        b.append(x)
        t += len(x[0]) / 16000
    if b:
        yield b


@torch.no_grad()
def score(m, sets, a):
    import torchaudio
    out = {}
    for name, items in sets.items():
        s = d = i = n = 0
        nll = ntok = 0.0
        for b in batches(items, a.batch_sec):
            L = torch.tensor([len(x) for x, _ in b]).cuda()
            A = torch.zeros(len(b), int(L.max())).cuda()
            for j, (x, _) in enumerate(b):
                A[j, : len(x)] = torch.from_numpy(x)
            e, el = encode(m, A, L)                                   # (B, T, D) fp32, the training look-ahead mask
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hyp = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=e.transpose(1, 2), encoded_lengths=el)
            hyp = hyp[0] if isinstance(hyp, tuple) else hyp
            for (_, ref), h in zip(b, hyp):
                r = normalize_v2(ref)
                x = edits(r, normalize_v2(h.text))
                s, d, i, n = s + x[0], d + x[1], i + x[2], n + len(r)
            if name == "meeting":                                     # T x U joint of a 16-min clip: no NLL
                continue
            for c in range(0, len(b), a.nll_chunk):                  # (B, T, U, V) fp32 joint: small chunks
                ys = [m.tokenizer.text_to_ids(ref) or [0] for _, ref in b[c: c + a.nll_chunk]]
                yl = torch.tensor([len(y) for y in ys]).cuda()
                Y = torch.zeros(len(ys), int(yl.max()), dtype=torch.long).cuda()
                for j, y in enumerate(ys):
                    Y[j, : len(y)] = torch.tensor(y)
                ec, lc = e[c: c + a.nll_chunk], el[c: c + a.nll_chunk]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    g, _, _ = m.decoder(targets=Y, target_length=yl)
                    z = m.joint.joint(ec[:, : int(lc.max())].to(torch.bfloat16), g.transpose(1, 2))  # T == max len
                loss = torchaudio.functional.rnnt_loss(z.float(), Y.int(), lc.int(), yl.int(),
                                                       blank=z.shape[-1] - 1, reduction="sum")
                nll, ntok = nll + float(loss), ntok + float(yl.sum())
                del z
        out[name] = {"wer": 100 * (s + d + i) / n, "S": 100 * s / n, "D": 100 * d / n, "I": 100 * i / n,
                     "nll": nll / ntok if ntok else float("nan")}
    out["mean"] = float(np.mean([v["wer"] for v in out.values()]))
    return out


def wait_for(paths, poll=5):
    while not all(os.path.exists(p) for p in paths):
        time.sleep(poll)
    return [json.load(open(p)) for p in paths]


def dump(path, obj):
    json.dump(obj, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)


def beam_search(a, L, teacher, run_eval, sets, fixed, per, wb):
    """Beam search over the set of kept layers: each stage expands every beam member by dropping each of its layers,
    scores the unique sets (cached by set, k_<bitmask>.json, so a set reached two ways is scored once; --import_dir
    seeds the cache with a greedy run's results), keeps the --beam best by
    obj = w * meeting WER + (1 - w) * mean WER of the other sets (w = --w_meeting)."""
    others = [s for s in sets if s != "meeting"]

    def obj(r):
        return a.w_meeting * r["meeting"]["wer"] + (1 - a.w_meeting) * float(np.mean([r[s]["wer"] for s in others]))

    def key(keep):
        return os.path.join(a.out, "k_" + "".join("1" if j in keep else "0" for j in range(L)) + ".json")

    if a.worker == 0 and a.import_dir:                    # seed the cache (same model / look-ahead / bp / sets / n)
        import glob
        n = 0
        for f in glob.glob(os.path.join(a.import_dir, "*.json")):
            r = json.load(open(f))
            if "keep" in r and not os.path.exists(key(r["keep"])):
                dump(key(r["keep"]), r)
                n += 1
        print(f"[beam] imported {n} scored layer sets from {a.import_dir}", flush=True)
    t0 = obj(teacher)
    beam = [tuple(range(L))]
    log = os.path.join(a.out, "beam.log")
    if a.worker == 0:
        with open(log, "a") as f:
            f.write(f"BEAM kept {L} rank 0 | obj {t0:.2f} | dropped [] | " + " ".join(f"{s} {teacher[s]['wer']:.2f}" for s in sets) + "\n")
    for k in range(L - 1, a.stop - 1, -1):
        cands = sorted({tuple(x for x in b if x != j) for b in beam for j in b})
        for idx, c in enumerate(cands):
            if idx % a.workers == a.worker:
                run_eval(list(c), key(c))
        res = dict(zip(cands, wait_for([key(c) for c in cands])))
        ranked = sorted(cands, key=lambda c: (obj(res[c]), c))
        beam = ranked[: a.beam]
        if a.worker == 0:
            lines = []
            for i, c in enumerate(beam):
                r = res[c]
                lines.append(f"BEAM kept {k} ({(fixed + k * per) / 1e6:.0f}M) rank {i} | obj {obj(r):.2f} (vs full {obj(r) - t0:+.2f}) | "
                             f"dropped {sorted(set(range(L)) - set(c))} | " + " ".join(f"{s} {r[s]['wer']:.2f}" for s in sets)
                             + f" | {len(cands)} candidates")
            print("\n".join(lines), flush=True)
            with open(log, "a") as f:
                f.write("\n".join(lines) + "\n")
            if wb is not None:
                r = res[beam[0]]
                wb.log({"layers_kept": k, "params_M": (fixed + k * per) / 1e6, "beam/obj": obj(r), "beam/d_full_obj": obj(r) - t0,
                        **{f"beam/wer/{s}": r[s]["wer"] for s in sets}})
    if wb is not None:
        wb.finish()
    print(f"PRUNE_DONE worker {a.worker}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b", help="HF id or a local .nemo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stop", type=int, default=8, help="last stage = this many layers kept")
    ap.add_argument("--sets", nargs="+", default=["ami_ihm", "notsofar", "librispeech", "earnings22"])
    ap.add_argument("--n", type=int, default=300, help="clips per val set")
    ap.add_argument("--la", type=int, default=3, help="look-ahead in 80 ms frames (3 = the app's 240 ms)")
    ap.add_argument("--bp", type=float, default=0.0, help="RNN-T blank penalty for the greedy decode (WER)")
    ap.add_argument("--batch_sec", type=float, default=300)
    ap.add_argument("--nll_chunk", type=int, default=8, help="clips per RNN-T NLL joint")
    ap.add_argument("--val_dir", default="/home/marimo/work/asr/en3")
    ap.add_argument("--meeting", default="/home/marimo/work/asr/eval_meeting")
    ap.add_argument("--wandb", default="ablations-tinycompany-ai/nemotron-prune", help="entity/project; '' = no W&B")
    ap.add_argument("--run", default="greedy-depth-la240")
    ap.add_argument("--beam", type=int, default=0, help="beam width (0 = greedy, ranked by the mean of all sets)")
    ap.add_argument("--w_meeting", type=float, default=0.8, help="beam objective weight of the meeting WER")
    ap.add_argument("--import_dir", default=None, help="seed the beam cache with another run's scored sets (same settings)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import nemo.collections.asr as nemo_asr
    m = (nemo_asr.models.ASRModel.restore_from(a.model, map_location="cuda") if a.model.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(a.model, map_location="cuda")).eval()
    from omegaconf import open_dict
    dec = m.cfg.decoding
    with open_dict(dec):              # the CUDA-graph greedy decoder hit illegal memory accesses with 4 workers + training
        dec.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(dec)
    if a.bp:                          # RNN-T blank penalty for the WER (the decoding setting that ships)
        import blank_penalty
        blank_penalty.apply(a.bp, head="rnnt")
    set_lookahead(m.encoder, a.la)
    sets = load_sets(a)
    layers = list(m.encoder.layers)
    fixed = sum(p.numel() for p in m.parameters()) - sum(p.numel() for p in m.encoder.layers.parameters())
    per = sum(p.numel() for p in layers[0].parameters())

    def use(keep):
        m.encoder.layers = torch.nn.ModuleList([layers[j] for j in keep])
        m.encoder.n_layers = len(keep)

    def run_eval(keep, path):
        if not os.path.exists(path):
            use(keep)
            t = time.time()
            r = score(m, sets, a)
            r.update(keep=keep, sec=time.time() - t)
            dump(path, r)
            print(f"[w{a.worker}] {os.path.basename(path)} mean {r['mean']:.2f} ({r['sec']:.0f} s)", flush=True)

    keep = list(range(len(layers)))
    base = os.path.join(a.out, f"s{len(keep)}_base.json")
    if a.worker == 0:
        run_eval(keep, base)
    teacher = prev = wait_for([base])[0]
    wb = None
    if a.worker == 0 and a.wandb:
        import wandb
        ent, proj = a.wandb.split("/")
        wb = wandb.init(entity=ent, project=proj, name=a.run, config=vars(a))
        wb.define_metric("layers_kept")
        wb.define_metric("*", step_metric="layers_kept")

    def log(kept, dropped, r, cands):
        row = {"layers_kept": kept, "params_M": (fixed + kept * per) / 1e6, "dropped_layer": dropped,
               "wer/mean": r["mean"], "d_teacher/mean": r["mean"] - teacher["mean"], "d_prev/mean": r["mean"] - prev["mean"]}
        for s in sets:
            row[f"wer/{s}"] = r[s]["wer"]
            row[f"nll/{s}"] = r[s]["nll"]
            row[f"d_teacher/{s}"] = r[s]["wer"] - teacher[s]["wer"]
            row[f"d_prev/{s}"] = r[s]["wer"] - prev[s]["wer"]
        line = (f"STAGE kept {kept:2d} ({row['params_M']:.0f}M) dropped L{dropped} | mean {r['mean']:.2f} "
                f"(vs teacher {row['d_teacher/mean']:+.2f}, vs prev {row['d_prev/mean']:+.2f}) | "
                + " ".join(f"{s} {r[s]['wer']:.2f}" for s in sets) + f" | keep {r['keep']}")
        print(line, flush=True)
        with open(os.path.join(a.out, "stages.log"), "a") as f:
            f.write(line + "\n")
        if wb is not None:
            import wandb
            cols = ["dropped_layer", "mean"] + list(sets)
            row["candidates"] = wandb.Table(columns=cols, data=[[c["dropped"], c["mean"]] + [c[s]["wer"] for s in sets]
                                                                for c in cands])
            wb.log(row)

    if a.beam:
        beam_search(a, len(layers), teacher, run_eval, sets, fixed, per, wb)
        return
    if a.worker == 0:
        log(len(keep), -1, teacher, [])
    while len(keep) > a.stop:
        k = len(keep) - 1
        paths = {j: os.path.join(a.out, f"s{k}_drop{j}.json") for j in keep}
        for idx, j in enumerate(keep):
            if idx % a.workers == a.worker:
                run_eval([x for x in keep if x != j], paths[j])
        res = dict(zip(paths, wait_for(list(paths.values()))))
        for j, r in res.items():
            r["dropped"] = j
        best = min(keep, key=lambda j: (res[j]["mean"], j))
        if a.worker == 0:
            log(k, best, res[best], sorted(res.values(), key=lambda c: c["mean"]))
        keep = [x for x in keep if x != best]
        prev = res[best]
    if wb is not None:
        wb.finish()
    print(f"PRUNE_DONE worker {a.worker}", flush=True)


if __name__ == "__main__":
    main()
