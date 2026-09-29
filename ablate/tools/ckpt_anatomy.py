"""Anatomy of one trained checkpoint: outliers / massive activations, expert load, routing stability,
fp8 health and per-token loss -- and the fp8-vs-bf16 EVALUATION delta on the same weights.

    python -m ablate.tools.ckpt_anatomy <result.json> [--seqs 32] [--json out.json]

Pass A runs the checkpoint the way it trained (fp8 experts for a --moe_fp8 run, via report_ckpt);
pass B re-runs the SAME holdout with the other expert kernel (fp8 <-> bf16). Only the expert GEMMs
differ between the passes, so B - A is exactly what the fp8 kernel does to this model at inference.

Seams (identical on the fused and eager paths): the experts' forward_pre_hook gets (x, top_k_index,
top_k_weights) where x is the RESIDUAL STREAM entering the MoE norm (patches._mk_mlp replays the
hook with the pre-norm `flat`), so the residual statistics are per layer, before layer i's FFN.

  resid     per layer: rms, max|x|, max/median, kurtosis, count of MASSIVE elements (|x| > 100 x
            the layer's median |x|, the Sun et al. 2024 criterion at our scale), which dims and
            which tokens (id, position) carry the layer max -- sink tokens / sink dims show up here
  route     per routed layer: load entropy, hottest / coldest expert share (x E, 1 = fair), dead
            experts, mean top-1 weight, the selection margin s_k - s_{k+1} (incl. the balance
            bias) and the share of tokens with margin < 1e-3 (a coin-flip routing decision)
  flip      per routed layer: share of tokens whose top-k SET differs between pass A and B, and
            the mean number of swapped experts -- routing instability under fp8 noise
  fp8       tkf moe_fp8.HEALTH per layer (x_F1_in, GU, act_up_F3_in, eo), whenever a pass runs fp8
  tokens    per-token CE: worst sequences, CE of massive-activation tokens vs the rest
"""
from ablate.common import _paths  # noqa: F401
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
from ablate.common import fp8_health

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
PAD = 0


def _moe_layers(model):
    return [(i, l.mlp) for i, l in enumerate(model.model.layers) if hasattr(getattr(l, "mlp", None), "experts")]


def run_pass(model, hold, fp8_mode, keep_idx):
    fp8_health.enable(fp8_mode)
    layers = _moe_layers(model)
    E = {i: m.experts.gate_up_proj.shape[0] for i, m in layers}
    R = {i: dict(n=0, s2=0.0, s4=0.0, amax=0.0, med=[], massive=0, dims={}, toks={}, load=torch.zeros(E[i], device=DEV),
                 w1=0.0, marg=[], coin=0, idx=[]) for i, _ in layers}
    cur = {}

    def mk(i, mlp):
        k = mlp.gate.top_k
        def hook(mod, args):
            x, idx = args[0].reshape(-1, args[0].shape[-1]).float(), args[1].reshape(-1, args[1].shape[-1])
            w, r = args[2].reshape(idx.shape).float(), R[i]
            ax = x.abs()
            med = ax.median().item()
            r["n"] += x.shape[0]; r["s2"] += x.pow(2).sum().item(); r["s4"] += x.pow(4).sum().item()
            r["amax"] = max(r["amax"], ax.max().item()); r["med"].append(med)
            mass = ax > 100 * med
            r["massive"] += int(mass.sum())
            tmax, tdim = ax.max(-1)                                   # per-token max and its dim
            hot = (tmax > 100 * med).nonzero().flatten()
            for d in tdim[hot].tolist():
                r["dims"][d] = r["dims"].get(d, 0) + 1
            ids = cur["ids"]
            for p in hot.tolist():
                t = int(ids[p]); r["toks"].setdefault(t, [0, []]); r["toks"][t][0] += 1
                if len(r["toks"][t][1]) < 8: r["toks"][t][1].append(p)
            cur.setdefault("mass_tok", torch.zeros(x.shape[0], dtype=torch.bool, device=DEV))
            cur["mass_tok"] |= tmax > 100 * med
            r["load"] += torch.bincount(idx.reshape(-1), minlength=E[i]).float()
            r["w1"] += w.max(-1).values.sum().item()
            if k < E[i]:                                              # routed: recompute the selection
                ln = model.model.layers[i].post_attention_layernorm
                hn = F.rms_norm(x, (x.shape[-1],), ln.weight.float(), eps=ln.variance_epsilon)
                s = torch.sigmoid(hn @ mlp.gate.gate_proj.weight.float().t())
                if mlp.gate.bias is not None:
                    s = s + mlp.gate.bias.float()
                top = s.topk(k + 1, -1).values
                m = top[:, k - 1] - top[:, k]
                r["marg"].append(m.median().item()); r["coin"] += int((m < 1e-3).sum())
            if keep_idx:
                r["idx"].append(idx.to(torch.uint8).cpu())
        return hook

    hs = [m.experts.register_forward_pre_hook(mk(i, m)) for i, m in layers]
    m8 = fp8_health._mod()
    health = {}
    ce_tok, mass_all = [], []
    W = model.lm_head.weight
    for b in range(hold.shape[0]):
        ids = hold[b]
        cur.clear(); cur["ids"] = ids[:-1].tolist()
        if fp8_mode:
            m8.HEALTH = {}
        with torch.no_grad(), AMP:
            h = model.model(input_ids=ids[None, :-1], use_cache=False).last_hidden_state[0]
            ce = F.cross_entropy((h.float() @ W.float().t()), ids[1:], reduction="none")
        if fp8_mode:
            for tag, rows in m8.HEALTH.items():
                health.setdefault(tag, [[] for _ in rows])
                for li, row in enumerate(rows):
                    if li < len(health[tag]): health[tag][li].append(row)
            m8.HEALTH = None
        ce_tok.append(torch.where(ids[1:] == PAD, torch.nan, ce))
        mass_all.append(cur.get("mass_tok", torch.zeros(ids.shape[0] - 1, dtype=torch.bool, device=DEV)))
    for h_ in hs:
        h_.remove()
    return R, E, health, torch.stack(ce_tok), torch.stack(mass_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    model, cfg = load_from_result(a.result_json, device=DEV)
    model.eval()
    trained = int(getattr(cfg, "moe_fp8", 0) or 0)
    other = 0 if trained else 1
    hold = _val.build_holdout(cfg.dataset, a.seq_len, a.seqs, DEV)   # seq_len + 1 tokens per row
    tag = getattr(cfg, "run_tag", os.path.basename(a.result_json))
    name = {0: "bf16", 1: "fp8 ALL", 2: "fp8 LEAN"}
    RA, E, HA, ceA, massA = run_pass(model, hold, trained, True)
    RB, _, HB, ceB, _ = run_pass(model, hold, other, True)
    L = sorted(RA)
    out = {"tag": tag, "trained": name[trained], "other": name[other], "seqs": a.seqs}

    def val(ce):
        return torch.nanmean(ce).item()
    print(f"\n==== {tag}: {a.seqs} holdout seqs x {a.seq_len} tok, trained {name[trained]}")
    print(f"val CE  {name[trained]} eval {val(ceA):.4f} | {name[other]} eval {val(ceB):.4f} | "
          f"delta {val(ceB) - val(ceA):+.4f}   (same weights, only the expert kernel differs)")
    out["val"] = {name[trained]: val(ceA), name[other]: val(ceB)}

    print("\nresidual stream entering each MoE layer (pass A)")
    print(f"{'L':>3} {'rms':>8} {'max':>9} {'max/med':>9} {'kurt':>8} {'massive':>8}  top dims (tokens at max)   top token ids (count, first positions)")
    out["resid"] = {}
    for i in L:
        r = RA[i]
        ne = r["n"] * model.config.hidden_size
        rms = math.sqrt(r["s2"] / ne)
        kurt = (r["s4"] / ne) / (r["s2"] / ne) ** 2
        med = sum(r["med"]) / len(r["med"])
        dims = sorted(r["dims"].items(), key=lambda t: -t[1])[:4]
        toks = sorted(r["toks"].items(), key=lambda t: -t[1][0])[:3]
        out["resid"][i] = dict(rms=rms, amax=r["amax"], max_over_med=r["amax"] / med, kurt=kurt, massive=r["massive"],
                               dims=dims, toks=[(t, c, p) for t, (c, p) in toks])
        print(f"{i:>3} {rms:8.3f} {r['amax']:9.2f} {r['amax'] / med:9.0f} {kurt:8.1f} {r['massive']:>8}  "
              f"{str(dims):<26} {[(t, c, p[:3]) for t, (c, p) in toks]}")

    print("\nrouting (pass A) and stability vs pass B")
    print(f"{'L':>3} {'E':>3} {'bal_ent':>8} {'hot xE':>7} {'cold xE':>8} {'dead':>5} {'top1_w':>7} {'margin':>8} {'coin%':>7} {'flip%':>7} {'swaps':>6}")
    out["route"] = {}
    for i in L:
        r, rb = RA[i], RB[i]
        p = (r["load"] / r["load"].sum()).clamp_min(1e-12)
        ent = float(-(p * p.log()).sum() / math.log(E[i]))
        ia, ib = torch.cat(r["idx"]).long(), torch.cat(rb["idx"]).long()
        sa = torch.zeros(ia.shape[0], E[i], dtype=torch.bool).scatter_(1, ia, True)
        sb = torch.zeros(ib.shape[0], E[i], dtype=torch.bool).scatter_(1, ib, True)
        diff = (sa ^ sb).sum(-1) // 2
        row = dict(bal_ent=ent, hot=float(p.max() * E[i]), cold=float(p.min() * E[i]), dead=int((r["load"] == 0).sum()),
                   top1_w=r["w1"] / r["n"], margin=(sum(r["marg"]) / len(r["marg"]) if r["marg"] else float("nan")),
                   coin=100 * r["coin"] / r["n"], flip=100 * float((diff > 0).float().mean()), swaps=float(diff.float().mean()))
        out["route"][i] = row
        print(f"{i:>3} {E[i]:>3} {ent:8.4f} {row['hot']:7.2f} {row['cold']:8.3f} {row['dead']:>5} {row['top1_w']:7.3f} "
              f"{row['margin']:8.2e} {row['coin']:7.3f} {row['flip']:7.3f} {row['swaps']:6.3f}")

    H = HA or HB
    if H:
        which = name[trained] if HA else name[other]
        print(f"\nfp8 health per layer ({which} pass; x_F1_in / act_up_F3_in)")
        print(f"{'fp8 layer':>9} {'x kurt':>8} {'x bmax/rms':>11} {'act kurt':>9} {'act bmax/rms':>13} {'act zero%':>10} {'act subn%':>10} {'act exp':>8}")
        out["fp8"] = {}
        mean = lambda rows, k: sum(r_[k] for r_ in rows if k in r_) / max(1, sum(k in r_ for r_ in rows))
        for li in range(len(H.get("x_F1_in", []))):
            x_, ac = H["x_F1_in"][li], H["act_up_F3_in"][li]
            row = {k: mean(x_, k) for k in x_[0]} | {"act_" + k: mean(ac, k) for k in ac[0]}
            out["fp8"][li] = row
            print(f"{li:>9} {row['kurtosis']:8.1f} {row.get('block_max_over_rms_p99', float('nan')):11.2f} {row['act_kurtosis']:9.1f} "
                  f"{row.get('act_block_max_over_rms_p99', float('nan')):13.2f} {row['act_zero_pct']:10.4f} "
                  f"{row['act_subnormal_pct']:10.3f} {row['act_scale_exp_mean']:8.2f}")

    seq = torch.nanmean(ceA, 1)
    order = seq.argsort()
    mz = torch.nanmean(ceA[massA]).item() if massA.any() else float("nan")
    rest = torch.nanmean(ceA[~massA]).item()
    print(f"\nper-seq CE: mean {seq.mean().item():.3f} sd {seq.std().item():.3f} | worst seqs {[(int(j), round(seq[j].item(), 3)) for j in order[-3:].flip(0)]}"
          f" | best {[(int(j), round(seq[j].item(), 3)) for j in order[:3]]}")
    print(f"tokens with a massive activation in any layer: {int(massA.sum())} / {massA.numel()} "
          f"({100 * massA.float().mean().item():.3f}%), their next-token CE {mz:.3f} vs {rest:.3f} for the rest")
    print(f"pass B - pass A per-token CE: mean {torch.nanmean(ceB - ceA).item():+.5f}, "
          f"p99 |d| {torch.nanquantile((ceB - ceA).abs().flatten()[~torch.isnan((ceB - ceA).flatten())], 0.99).item():.4f}")
    out["tokens"] = dict(seq_mean=seq.tolist(), massive_share=massA.float().mean().item(), ce_massive=mz, ce_rest=rest)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
