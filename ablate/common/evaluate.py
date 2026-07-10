"""Unified eval entrypoint used BOTH periodically during training and standalone. Returns a nested
result dict + a flat {metric: value} dict for W&B. Reports English AND Hindi separately for every task.
Model is toggled to eval() and restored. Datasets/tokenizer are cached, so periodic calls are cheap."""
from . import _paths  # noqa: F401
import torch
from .eval.bpb import run_bpb
from .eval.manifest import MANIFEST
from .eval.mcq import run_mcq, default_sources as mcq_sources
from .eval.length_extrap import run_length_extrap
from .eval.probes import run_probes
from .eval.icl import run_icl
from .eval import interp as interp_mod
from .eval.sample import generate_samples

TOKENIZER = "fhai50032/QTK-81K"


class Tok:
    """QTK-81K wrapper: .encode(text)->list[int], NO special tokens (bpb/LL correctness)."""
    def __init__(self):
        from transformers import AutoTokenizer
        self._t = AutoTokenizer.from_pretrained(TOKENIZER)

    def encode(self, text):
        return self._t.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self._t.decode(ids, skip_special_tokens=True)


def evaluate(model, tokenizer, *, seq_len=1024, mcq_n=200, bpb_n=None, extrap_lengths=None,
             do_probes=True, with_global_mmlu=False, do_samples=True, do_icl=False, icl_n=100,
             device="cuda", dtype=torch.bfloat16):
    """Periodic (cheap): small mcq_n/bpb_n, extrap_lengths=None, do_icl=False. Final (full): larger
    mcq_n, extrap set, do_icl=True. ICL is a SEPARATE metric (own eval/icl_* keys), off by default."""
    was_training = model.training
    model.eval()
    res = {}
    num_experts = getattr(model.config, "num_routed_experts", None) or getattr(model.config, "num_experts", 9)
    # collect MoE interp (expert utilization + router confidence) FOR FREE during the bpb forwards
    with interp_mod.collect(model, num_experts) as moe:
        res["bpb"] = run_bpb(model, tokenizer, MANIFEST, seq_len=seq_len, device=device, dtype=dtype,
                             n_override=bpb_n)
    res["interp"] = moe.result()
    res["mcq"] = run_mcq(model, tokenizer, mcq_sources(mcq_n, with_global_mmlu), device=device, dtype=dtype)
    if do_probes:
        res["probes"] = run_probes(model, tokenizer, device=device, dtype=dtype)
    if extrap_lengths:
        res["length_extrap"] = run_length_extrap(model, tokenizer, MANIFEST, lengths=extrap_lengths,
                                                 train_len=seq_len, device=device, dtype=dtype)
    if do_icl:                                              # SEPARATE metric — own eval/icl_* namespace
        res["icl"] = run_icl(model, tokenizer, n=icl_n, device=device, dtype=dtype)
    if do_samples:                                           # 2 en + 2 hi qualitative samples (KV-cache decode)
        res["samples"] = generate_samples(model, tokenizer, device=device, dtype=dtype)
    if was_training:
        model.train()

    flat = {f"eval/bpb_{k}": v for k, v in res["bpb"]["per_language"].items()}
    flat["eval/bpb_overall"] = res["bpb"]["overall"]
    # per-subset bpb (each benchmark source), grouped by language -> eval/bpb/<lang>/<source>
    flat.update({f"eval/bpb/{d['lang']}/{name}": d["bpb"] for name, d in res["bpb"]["per_source"].items()})
    flat.update({f"eval/acc_{k}": d["acc"] for k, d in res["mcq"]["per_language"].items()})
    it = res["interp"]
    flat.update({"eval/expert_balance_entropy": it["balance_entropy"], "eval/max_expert_load": it["max_expert_load"],
                 "eval/router_top1_weight": it["router_top1_weight"], "eval/router_entropy": it["router_entropy"]})
    if "probes" in res:
        flat.update({f"eval/probe_{k}": d["acc"] for k, d in res["probes"]["per_language"].items()})
    if "length_extrap" in res:
        flat.update({f"eval/extrap_degradation_{k}": v["degradation"]
                     for k, v in res["length_extrap"].items()})
    if "icl" in res:                                        # separate eval/icl_* keys (slopes + per-shot curve)
        flat.update(res["icl"]["flat"])
    return res, flat


def summarize(flat):
    """One-line en/hi summary for the training log (adds ICL slope/jump when present)."""
    g = lambda k: flat.get(k, float("nan"))
    s = (f"bpb hi={g('eval/bpb_hi'):.3f} en={g('eval/bpb_en'):.3f} | "
         f"acc hi={g('eval/acc_hi'):.3f} en={g('eval/acc_en'):.3f}")
    if "eval/icl_slope_acc_en" in flat:
        s += (f" | icl slope en={g('eval/icl_slope_acc_en'):.3f} hi={g('eval/icl_slope_acc_hi'):.3f}"
              f" jump en={g('eval/icl_jump_acc_en'):.3f} hi={g('eval/icl_jump_acc_hi'):.3f}")
    return s
