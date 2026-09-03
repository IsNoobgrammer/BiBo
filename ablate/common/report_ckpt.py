"""Regenerate the end-of-run report for a checkpoint that already exists.

    python -m ablate.common.report_ckpt <..._result.json> [--wandb_project P] [--run_tag T]

`train.py --final_report` only fires at the end of a run, so any checkpoint trained before that
option existed has no samples, no degeneration metrics and no extrapolation numbers. This rebuilds
the architecture from the run's OWN `_result.json` -- never from flags retyped by hand, which is how
a report ends up faithfully describing a model that was never trained -- loads `_final.pt`, and runs
the same `final_report`.

Also the entry point for later interpretability work: it gets a saved checkpoint back into memory
with the exact patches, expert layout, SWA pattern and XSA settings it was trained with.
"""
from . import _paths  # noqa: F401
import argparse
import json
import os
import torch

from .models import build_arm, count_params
from .configs import SHARED, resolve_swa
from . import patches as patchmod
from . import final_report as fr
from kernels.sm120.cross_entropy import fused_linear_cross_entropy

DEV = "cuda"


def _arch_kwargs(c):
    """build_arm(**kwargs) reproducing the architecture a training run built from `c` = its saved
    vars(args). Every key defaulted, so a result.json written before an axis existed still loads."""
    pattern, window = resolve_swa(c.get("swa_pattern", "none"),
                                  c.get("sliding_window", 128),
                                  SHARED["num_hidden_layers"])
    bias_factor = c.get("bias_update_factor", -1)
    moe_overrides = {}
    for spec in filter(None, (v.strip() for v in c.get("moe_override", "").split(","))):
        layer, experts, top_k, width = map(int, spec.split(":"))
        moe_overrides[layer] = dict(num_routed_experts=experts, num_experts_per_tok=top_k,
                                    moe_intermediate_size=width)
    return dict(
        attn_res_fp32_stream=c.get("attn_res_fp32_stream", False),
        attn_res_carry_scale=c.get("attn_res_carry_scale", "none"),
        attn_res_emb_term=c.get("attn_res_emb_term", False),
        attn_res_emb_scale=c.get("attn_res_emb_scale", "none"),
        attn_res_emb_site=c.get("attn_res_emb_site", "mlp"),
        attn_res_emb_gain=c.get("attn_res_emb_gain", False),
        attn_res_score=c.get("attn_res_score", "softmax"),
        attn_res_topk=c.get("attn_res_topk", 0),
        num_pos_identity_experts=c.get("pos_identity_n"),
        num_neg_identity_experts=c.get("neg_identity_n"),
        attn_res_carry_per_dim=c.get("attn_res_carry_per_dim", False),
        attn_res_carry_gate=c.get("attn_res_carry_gate", "none"),
        attn_res_emb_per_dim=c.get("attn_res_emb_per_dim", False),
        bf16_residual_stream=c.get("bf16_residual_stream", False),
        bf16_moe_out=c.get("bf16_moe_out", False),
        mlp_only_layers=(None if c.get("mlp_only_layers") is None else
                         [] if str(c["mlp_only_layers"]).lower() == "none" else
                         [int(v) for v in str(c["mlp_only_layers"]).split(",")]),
        intermediate_size=c.get("dense_inter"),
        moe_overrides=moe_overrides,
        num_experts=c.get("experts") or None,
        special_pairs=c.get("special_pairs", 0),
        use_xsa=c.get("use_xsa", False),
        xsa_alpha_init=c.get("xsa_alpha_init", 0.0),
        pos_identity_expert=c.get("pos_identity_expert", True),
        neg_identity_expert=c.get("neg_identity_expert", True),
        top_k=(c.get("top_k") or None),
        moe_intermediate_size=(c.get("moe_inter") or None),
        num_shared_experts=c.get("n_shared", 0),
        hybrid_layer_pattern=pattern,
        sliding_window=window,
        swa_qk_norm=c.get("swa_qk_norm", True),
        attn_res=c.get("attn_res", "off"),
        attn_res_sites=c.get("attn_res_sites", 2),
        attn_res_carry=c.get("attn_res_carry", False),
        use_typed_attn_res=c.get("typed_attn_res", False),
        typed_attn_res_long_memory=c.get("typed_attn_res_long_memory", True),
        typed_attn_res_extra_init=c.get("typed_attn_res_extra_init", 0.01),
        use_typed_attn_res_fast_slow_memory=c.get("typed_attn_res_fast_slow_memory", False),
        typed_attn_res_fast_decay_init=c.get("typed_attn_res_fast_decay_init", 0.5),
        typed_attn_res_slow_decay_init=c.get("typed_attn_res_slow_decay_init", 0.95),
        use_typed_attn_res_innovation_write=c.get("typed_attn_res_innovation_write", False),
        typed_attn_res_innovation_init=c.get("typed_attn_res_innovation_init", 0.01),
        bias_update_threshold=c.get("bias_update_threshold", 10240),
        bias_update_factor=(None if bias_factor is not None and bias_factor < 0 else bias_factor),
        aux_coef=c.get("aux_coef", 0.001),
    )


def load_from_result(result_json, device=DEV):
    """(model, cfg_namespace). Rebuilds the arm from the recorded config and loads the weights."""
    with open(result_json) as f:
        res = json.load(f)
    c = argparse.Namespace(**res["config"])
    ckpt = res.get("ckpt") or result_json.replace("_result.json", "_final.pt")
    assert os.path.exists(ckpt), f"checkpoint missing: {ckpt}"

    # Patches must be applied BEFORE the weights load: 'moe'/'megakernel' only swap forward methods,
    # but --act / --radial_p steer which kernel act code those forwards use, and a mismatch here
    # silently evaluates a different activation than the one that was trained.
    patchmod.RADIAL_P = c.radial_p
    patchmod.EXPERT_ACT = c.act
    patch_list = [p.strip() for p in c.patches.split(",") if p.strip() and p.strip() != "ce"]
    patchmod.apply(patch_list)

    model, _ = build_arm(
        c.arm, device=device, dtype=torch.float32, attn_impl=c.attn,
        **_arch_kwargs(vars(c)))

    sd = torch.load(ckpt, map_location=device)
    # strict=True on purpose: a silently-ignored missing key is a randomly-initialised tensor, and
    # the report would read as a genuine (bad) result rather than a loading bug.
    model.load_state_dict(sd, strict=True)
    model.eval()
    total, _tr, active = count_params(model, top_k=c.top_k or None, num_experts=c.experts or SHARED["num_experts"])
    print(f"[report_ckpt] loaded {os.path.basename(ckpt)} | params total={total/1e6:.2f}M "
          f"active={active/1e6:.2f}M | patches={patch_list} act={c.act}/{c.radial_p}", flush=True)
    return model, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--run_tag", default="")
    # attach to an EXISTING run; taken from result.json when present, this is the override for
    # checkpoints trained before the id was recorded
    ap.add_argument("--wandb_id", default="")
    ap.add_argument("--sample_tokens", type=int, default=96)
    ap.add_argument("--extrap_lens", default="1024,2048,4096")
    ap.add_argument("--n_seqs", type=int, default=4)
    a = ap.parse_args()

    model, c = load_from_result(a.result_json)
    from transformers import AutoTokenizer
    from .train import TOKENIZER
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    wb = None
    if a.wandb_project:
        import wandb
        with open(a.result_json) as f:
            _res = json.load(f)
        rid = a.wandb_id or _res.get("wandb_id")
        # console="wrap" so the generations print into the run's Logs tab, where long text is
        # actually readable -- W&B's table view is not
        _st = wandb.Settings(console="wrap")
        if rid:
            # RESUME the training run. A report in its own run sits next to the curves it describes
            # instead of on them, and cannot be compared -- which is the whole reason to log it.
            wb = wandb.init(project=a.wandb_project, id=rid, resume="must", settings=_st)
            print(f"[report_ckpt] resumed W&B run {rid}", flush=True)
        else:
            wb = wandb.init(project=a.wandb_project, settings=_st,
                            name=(a.run_tag or os.path.basename(a.result_json).replace("_result.json", "")) + "-report",
                            config=vars(c))
            print("[report_ckpt] WARNING: no wandb_id in the result json (run predates it) -- "
                  "logging to a NEW run. Pass --wandb_id <id> to attach to the training run.",
                  flush=True)
    amp = torch.autocast("cuda", torch.bfloat16)
    lens = tuple(int(x) for x in a.extrap_lens.split(",") if x.strip())
    flat = fr.run(model, tok, c.dataset, fused_linear_cross_entropy, amp, device=DEV, wb=wb,
                  max_new=a.sample_tokens, n_seqs=a.n_seqs, lens=lens)
    out = a.result_json.replace("_result.json", "_report.json")
    with open(out, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"\n[report_ckpt] wrote {out}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
