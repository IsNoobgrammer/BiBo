"""The fused residual-add kernel, INSIDE the model, against the eager torch path.

parity_check/parity_residual_add.py grades the kernel standalone on random tensors and it passes:
forward matches eager's error to the digit in bf16/fp16, and on the real dtype layout it is
30,000-47,000x closer to fp64 truth because eager rounds the learned scalar to bf16. None of that
proves the thing that matters, which is that swapping it into exp/modeling_bibo changes the model
ONLY in the way parity predicts.

It matters because the measured result went the wrong way: the same config scored bpb hi 0.5061
eager and 0.5098 on the kernel -- 2.6x the noise floor WORSE, from a change that is strictly more
accurate in isolation. Either the loss genuinely prefers eager's bf16 rounding of the scalar, or
something differs between the two paths that standalone parity never exercised. This separates
those: if kernel and eager agree here to the level parity predicts, the kernel is sound and the
bpb gap is a real (and surprising) preference. If they do not, there is a bug.

    python -m ablate.common.test_res_add_gpu
"""
from . import _paths  # noqa: F401

import torch

import exp.modeling_bibo as E
from .models import build_arm
from .configs import swa_block_pattern, SHARED
from . import patches as patchmod


def main():
    assert torch.cuda.is_available(), "needs a GPU: this is the path that runs in training"
    assert E._HAS_FUSED_RES_ADD, "fused residual-add kernel not importable -- nothing to compare"
    patchmod.apply(["liger_norm", "liger_rope", "moe", "xsa"])
    pat = swa_block_pattern(SHARED["num_hidden_layers"])
    ids = torch.randint(0, SHARED["vocab_size"], (2, 512), device="cuda")

    for tag, (cs, emb, es) in (("carry unbounded", ("unbounded", False, "none")),
                               ("carry+emb raw", ("unbounded", True, "none")),
                               ("carry+emb 2sigmoid", ("sigmoid", True, "2sigmoid")),
                               ("carry fixed c=1", ("none", False, "none"))):
        torch.manual_seed(42069)
        model, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32,
                             num_experts=8, top_k=2, special_pairs=0, use_xsa=True,
                             hybrid_layer_pattern=pat, sliding_window=128,
                             attn_res="3", attn_res_sites=1, attn_res_carry=True,
                             attn_res_fp32_stream=True, attn_res_carry_scale=cs,
                             attn_res_emb_term=emb, attn_res_emb_scale=es)
        model.train()
        # give the scalars non-trivial values -- at theta=0 the carry is exactly 1.0 and d is 0 or
        # 1, which is the one setting where a scale bug cannot show up.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "attn_res_carry_theta" in n:
                    p.fill_(0.6)
                if "attn_res_emb_theta" in n:
                    p.fill_(0.4)

        # the MoE mutates gate.bias on every TRAINING forward, so restore all buffers before each
        # run or this measures bias drift instead of the kernel
        snap = {n: b.detach().clone() for n, b in model.named_buffers()}

        def run():
            with torch.no_grad():
                for n, b in model.named_buffers():
                    b.copy_(snap[n])
            for p in model.parameters():
                p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(input_ids=ids, use_cache=False)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            h.float().square().mean().backward()
            return (h.detach().float(),
                    {n: p.grad.detach().float().clone()
                     for n, p in model.named_parameters() if p.grad is not None})

        E._HAS_FUSED_RES_ADD = True
        h_k, g_k = run()
        h_k2, _ = run()
        d_self = (h_k - h_k2).abs().max().item()
        assert d_self == 0.0, f"{tag}: two identical kernel runs differ by {d_self:.2e}"
        E._HAS_FUSED_RES_ADD = False
        h_e, g_e = run()
        E._HAS_FUSED_RES_ADD = True

        d_out = (h_k - h_e).abs().max().item() / h_e.abs().max().item()
        worst, wname = 0.0, ""
        for n in g_e:
            den = g_e[n].abs().max().item()
            if den < 1e-12:
                continue
            r = (g_k[n] - g_e[n]).abs().max().item() / den
            if r > worst:
                worst, wname = r, n
        # bf16 through 10 layers: eager and kernel differ by the bf16 scalar rounding this kernel
        # removes, compounded down the stack. A few percent is expected; an order of magnitude is a
        # bug, and a gradient that disagrees far more than the hidden state is the signature of one.
        ok = d_out < 5e-2 and worst < 2e-1
        print(f"{tag:<22} hidden {d_out:.2e} | worst grad {worst:.2e} ({wname[:40]})"
              + ("  ok" if ok else "  <-- FAIL"))
        # the scalar gradients are the kernel's own arithmetic, so call them out separately
        for key in ("attn_res_carry_theta", "attn_res_emb_theta"):
            hits = [(n, g_e[n], g_k[n]) for n in g_e if key in n]
            if hits:
                r = max((k_ - e_).abs().max().item() / max(e_.abs().max().item(), 1e-12)
                        for _, e_, k_ in hits)
                print(f"{'':<22}   {key}: worst rel {r:.2e} over {len(hits)} layers")
        assert ok, f"{tag}: kernel and eager disagree inside the model"
        del model
        torch.cuda.empty_cache()
    print("PASS")


if __name__ == "__main__":
    main()
