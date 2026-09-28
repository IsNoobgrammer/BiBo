"""Branch magnitudes inside each layer's AttnRes MLP input: mlp_in = attn_read + c * attn_out.

    python -m ablate.tools.mtp_branch_norms <result.json> [...] [--n_seqs 8]

Per layer (trunk L0-L9, then the MTP layer as L10): rms(attn_read), rms(c*attn_out), their cosine,
mean c, and rms of the MLP output -- read by wrapping the fused residual add and the MLP forward.
Answers "which path does the MTP layer take": depth read vs its own attention vs the MLP update.
"""
from ablate.common import _paths  # noqa: F401
import argparse

import torch
import torch.nn.functional as F

from ablate.common.report_ckpt import load_from_result
from ablate.common import validation as _val
import ablate.tools.mtp_probe  # noqa: F401  (same loader path)

DEV = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
rms = lambda t: t.float().pow(2).mean().sqrt().item()


def run(res, hold):
    model, c = load_from_result(res)
    rec = {}
    layers = list(model.model.layers) + ([model.mtp.layer] if getattr(model, "mtp", None) is not None else [])
    mlp_out = {}
    for i, L in enumerate(layers):
        fc, f = L._fused_carry, L._attn_res_mlp_forward

        def carry(read, ao, ps, fc=fc, i=i, L=L):
            hid, new_ps = fc(read, ao, ps)
            cao = hid.float() - read.float()                   # = c * attn_out
            th = L.attn_res_carry_theta
            rec[i] = dict(read=rms(read), cao=rms(cao),
                          c=(2 * torch.sigmoid(th.float())).mean().item() if th is not None else 1.0,
                          cos=F.cosine_similarity(read.float().reshape(-1), cao.reshape(-1), dim=0).item())
            return hid, new_ps

        def g(x, f=f, i=i):
            y = f(x)
            mlp_out[i] = (rms(x), rms(y))
            return y
        L._fused_carry, L._attn_res_mlp_forward = carry, g
    with torch.no_grad(), AMP:
        model.model.training = True          # hand over RoPE + archive to the MTP head (top flag only)
        h = model.model(input_ids=hold[:, :-1], use_cache=False).last_hidden_state
        model.model.training = False
        if getattr(model, "mtp", None) is not None:
            pe, br = model.model._mtp_cache
            model.mtp(h, model.model.embed_tokens(hold[:, 1:]), pe, br)
    out = []
    for i, r in sorted(rec.items()):
        mi, mo = mlp_out.get(i, (float("nan"), float("nan")))
        out.append((i, r["c"], r["read"], r["cao"], r["cao"] / r["read"], r["cos"], mo, mo / r["read"]))
    del model
    torch.cuda.empty_cache()
    return getattr(c, "run_tag", res), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--n_seqs", type=int, default=8)
    a = ap.parse_args()
    import json
    c0 = json.load(open(a.results[0]))["config"]
    hold = _val.build_holdout(c0["dataset"], c0["seq_len"], a.n_seqs, DEV)
    for r in a.results:
        tag, rows = run(r, hold)
        print(f"\n== {tag}\n  L   c_mean  rms(read)  rms(c*attn)  attn/read  cos(read,c*attn)  rms(mlp_out)  mlp/read")
        for i, cm, rd, ca, ratio, cs, mo, mr in rows:
            print(f"  {i:<3d} {cm:7.3f} {rd:10.4f} {ca:12.4f} {ratio:10.3f} {cs:17.3f} {mo:13.4f} {mr:9.3f}")
    print("BRANCH_DONE")


if __name__ == "__main__":
    main()
