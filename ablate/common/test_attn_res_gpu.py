"""The fused AR kernel, INSIDE the model, against the eager torch path.

test_attn_res runs on CPU, where `apply_attention_residual` takes its torch branch and the
kernel never executes -- so it has never actually checked the thing that runs in training.
This flips `_HAS_FUSED_AR` on the same weights and compares logits and gradients end to end.

Also reports the residual-stream dtype per layer, because that is what made the AttnRes arms
non-comparable to the baseline: the block-boundary reset drops the fp32 embedding out of the
stream, so AttnRes runs bf16 where the control runs fp32.

    python -m ablate.common.test_attn_res_gpu
"""
from . import _paths  # noqa: F401

import torch

import exp.modeling_bibo as E
from .models import build_arm
from .configs import swa_block_pattern, SHARED
from . import patches as patchmod


def _stream_dtypes(model, ids):
    seen, hooks = [], []
    for i, lyr in enumerate(model.model.layers):
        hooks.append(lyr.register_forward_hook(
            lambda m, inp, out, i=i: seen.append(
                str((out[0] if isinstance(out, (tuple, list)) else out).dtype).replace("torch.", ""))))
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    return seen


def main():
    assert torch.cuda.is_available(), "needs a GPU: this is the path that runs in training"
    assert E._HAS_FUSED_AR, "fused AR kernel not importable -- nothing to compare against"
    patchmod.apply(["liger_norm", "liger_rope", "moe", "xsa"])
    pat = swa_block_pattern(SHARED["num_hidden_layers"])
    ids = torch.randint(0, SHARED["vocab_size"], (2, 512), device="cuda")

    for tag, (ar, sites, carry) in (("b3 sites=2", ("3", 2, False)),
                                    ("b3 sites=1 carry", ("3", 1, True)),
                                    ("b1 sites=1 carry", ("1", 1, True))):
        torch.manual_seed(42069)
        model, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32,
                             num_experts=8, top_k=2, special_pairs=0, use_xsa=True,
                             hybrid_layer_pattern=pat, sliding_window=128,
                             attn_res=ar, attn_res_sites=sites, attn_res_carry=carry)
        model.train()

        def run():
            for p in model.parameters():
                p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(input_ids=ids, use_cache=False)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            loss = h.float().square().mean()
            loss.backward()
            named = {n: p.grad.detach().float().clone()
                     for n, p in model.named_parameters() if p.grad is not None}
            return h.detach().float(), named

        E._HAS_FUSED_AR = True
        h_k, g_k = run()
        E._HAS_FUSED_AR = False
        h_e, g_e = run()
        E._HAS_FUSED_AR = True

        d_out = (h_k - h_e).abs().max().item() / h_e.abs().max().item()
        worst, wname = 0.0, ""
        for n in g_e:
            den = g_e[n].abs().max().item()
            if den < 1e-12:
                continue
            r = (g_k[n] - g_e[n]).abs().max().item() / den
            if r > worst:
                worst, wname = r, n
        ok = d_out < 2e-2 and worst < 5e-2
        print(f"{tag:<20} hidden {d_out:.2e} | worst grad {worst:.2e} ({wname[:44]})"
              + ("  ok" if ok else "  <-- FAIL"))
        assert ok, f"{tag}: kernel and eager disagree inside the model"
        del model
        torch.cuda.empty_cache()

    print()
    print("residual-stream dtype per layer (autocast bf16):")
    for tag, (ar, sites, carry) in (("no AttnRes", ("control", 2, False)),
                                    ("b3 carry", ("3", 1, True))):
        torch.manual_seed(42069)
        m, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32, num_experts=8,
                         top_k=2, special_pairs=0, use_xsa=True, hybrid_layer_pattern=pat,
                         sliding_window=128, attn_res=ar, attn_res_sites=sites,
                         attn_res_carry=carry)
        print(f"  {tag:<12}{_stream_dtypes(m.train(), ids)}")
        del m
        torch.cuda.empty_cache()
    print("PASS")


if __name__ == "__main__":
    main()
