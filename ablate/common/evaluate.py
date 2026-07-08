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

TOKENIZER = "fhai50032/QTK-81K"


class Tok:
    """QTK-81K wrapper: .encode(text)->list[int], NO special tokens (bpb/LL correctness)."""
    def __init__(self):
        from transformers import AutoTokenizer
        self._t = AutoTokenizer.from_pretrained(TOKENIZER)

    def encode(self, text):
        return self._t.encode(text, add_special_tokens=False)


def evaluate(model, tokenizer, *, seq_len=1024, mcq_n=200, bpb_n=None, extrap_lengths=None,
             do_probes=True, with_global_mmlu=False, device="cuda", dtype=torch.bfloat16):
    """Periodic (cheap): small mcq_n/bpb_n, extrap_lengths=None. Final (full): larger mcq_n, extrap set."""
    was_training = model.training
    model.eval()
    res = {}
    res["bpb"] = run_bpb(model, tokenizer, MANIFEST, seq_len=seq_len, device=device, dtype=dtype,
                         n_override=bpb_n)
    res["mcq"] = run_mcq(model, tokenizer, mcq_sources(mcq_n, with_global_mmlu), device=device, dtype=dtype)
    if do_probes:
        res["probes"] = run_probes(model, tokenizer, device=device, dtype=dtype)
    if extrap_lengths:
        res["length_extrap"] = run_length_extrap(model, tokenizer, MANIFEST, lengths=extrap_lengths,
                                                 train_len=seq_len, device=device, dtype=dtype)
    if was_training:
        model.train()

    flat = {f"eval/bpb_{k}": v for k, v in res["bpb"]["per_language"].items()}
    flat["eval/bpb_overall"] = res["bpb"]["overall"]
    flat.update({f"eval/acc_{k}": v for k, v in res["mcq"]["per_language"].items()})
    if "probes" in res:
        flat.update({f"eval/probe_{k}": v for k, v in res["probes"]["per_language"].items()})
    if "length_extrap" in res:
        flat.update({f"eval/extrap_degradation_{k}": v["degradation"]
                     for k, v in res["length_extrap"].items()})
    return res, flat


def summarize(flat):
    """One-line en/hi summary for the training log."""
    g = lambda k: flat.get(k, float("nan"))
    return (f"bpb hi={g('eval/bpb_hi'):.3f} en={g('eval/bpb_en'):.3f} | "
            f"acc hi={g('eval/acc_hi'):.3f} en={g('eval/acc_en'):.3f}")
