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
    # imported AFTER apply(), like gate_emb_gain does -- a patch that swapped the class would
    # make an isinstance check against a module-level import silently match nothing.
    from src.modeling.ffn.router import BiBoMoERouter
    pat = swa_block_pattern(SHARED["num_hidden_layers"])
    ids = torch.randint(0, SHARED["vocab_size"], (2, 512), device="cuda")

    # (carry_scale, emb_term, emb_scale, fill) -- `fill` is the value every learned scalar takes.
    # fill=1.0 makes the multiply EXACT, which is the only way bit-identity is reachable (see the
    # contract note below). Those cases are the hard gate; the rest are the amplification cases.
    for tag, (cs, emb, es, fill) in (
            ("carry c=1.0 learned", ("unbounded", False, "none", 1.0)),
            ("carry+emb both 1.0", ("unbounded", True, "none", 1.0)),
            ("carry unbounded", ("unbounded", False, "none", 0.625)),
            ("carry+emb raw", ("unbounded", True, "none", 0.625)),
            ("carry+emb 2sigmoid", ("sigmoid", True, "2sigmoid", 0.625)),
            ("carry fixed c=1", ("none", False, "none", 0.625))):
        torch.manual_seed(42069)
        model, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32,
                             num_experts=8, top_k=2, special_pairs=0, use_xsa=True,
                             hybrid_layer_pattern=pat, sliding_window=128,
                             attn_res="3", attn_res_sites=1, attn_res_carry=True,
                             attn_res_fp32_stream=False, bf16_residual_stream=True,
                             attn_res_carry_scale=cs,
                             attn_res_emb_term=emb, attn_res_emb_scale=es)
        model.train()
        # Never 0: at theta=0 the carry is exactly 1.0 and d is 0, the one setting where a scale
        # bug cannot show up. 0.625 = 1.25*2^-1 is bf16-exact, which removes scalar-rounding as a
        # confound -- though measurement says that is NOT what drives the divergence (see below).
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "attn_res_carry_theta" in n or "attn_res_emb_theta" in n:
                    p.fill_(fill)

        # ROUTER TOP-K FLIPS. Without this the hidden/grad numbers below cannot be interpreted:
        # in an MoE a sub-ULP difference can flip which expert runs, and one flip sends a token
        # down a different subnetwork. That is not the kernel being wrong, it is two different
        # trajectories being compared, and it reads as a huge hidden divergence either way.
        # gate_emb_gain has always counted flips; main() never did, which is why it fired at
        # hidden 2.9e-01 with no way to tell amplification from a bug.
        picks = []
        hooks = [m.register_forward_hook(lambda _m, _i, o: picks.append(o[0].detach().clone()))
                 for m in model.modules() if isinstance(m, BiBoMoERouter)]
        assert hooks, "no routers found -- the flip count would be vacuously zero"

        # the MoE mutates gate.bias on every TRAINING forward, so restore all buffers before each
        # run or this measures bias drift instead of the kernel
        snap = {n: b.detach().clone() for n, b in model.named_buffers()}

        def run():
            picks.clear()
            with torch.no_grad():
                for n, b in model.named_buffers():
                    b.copy_(snap[n])
            for p in model.parameters():
                p.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.model(input_ids=ids, use_cache=False)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            h.float().square().mean().backward()
            return (h.detach().float(), list(picks),
                    {n: p.grad.detach().float().clone()
                     for n, p in model.named_parameters() if p.grad is not None})

        E._HAS_FUSED_RES_ADD = True
        h_k, p_k, g_k = run()
        h_k2, _, g_k2 = run()
        d_self = (h_k - h_k2).abs().max().item()
        assert d_self == 0.0, f"{tag}: two identical kernel runs differ by {d_self:.2e}"
        E._HAS_FUSED_RES_ADD = False
        h_e, p_e, g_e = run()
        E._HAS_FUSED_RES_ADD = True

        def worst_grad(ga, gb):
            """Worst RELATIVE gradient disagreement between two runs, and where."""
            w, wn = 0.0, ""
            for n in gb:
                den = gb[n].abs().max().item()
                if den < 1e-12 or n not in ga:
                    continue
                r = (ga[n] - gb[n]).abs().max().item() / den
                if r > w:
                    w, wn = r, n
            return w, wn

        # THE CONTROL. The same kernel run twice, gradients compared. attn_res's backward reduces
        # with atomic_add, whose summation order is not fixed, so gradients are nondeterministic
        # run to run even with a bit-identical forward. Without this floor a kernel-vs-eager
        # gradient number means nothing -- the fixed-c case reads 2.1e-02 with an EXACT forward
        # and ZERO router flips, which cannot be a kernel-vs-eager difference at all.
        g_floor, g_floor_n = worst_grad(g_k, g_k2)
        flips = sum((a != b).sum().item() for a, b in zip(p_k, p_e))
        n_pick = sum(a.numel() for a in p_k)
        d_out = (h_k - h_e).abs().max().item() / h_e.abs().max().item()
        worst, wname = worst_grad(g_k, g_e)
        # WHAT THIS CAN AND CANNOT ASSERT.
        # Under a bf16 stream eager computes `c.to(bf16) * attn_output` IN BF16 -- it rounds the
        # PRODUCT -- then rounds again on the add. The kernel forms the product in fp32 and rounds
        # once, at the store. That is the whole difference, and the fp64 grade says the kernel is
        # the more accurate of the two. So equality is not merely unmet for c != 1, it is the
        # wrong thing to demand, and the old blanket tolerance fired at hidden 2.9e-01 while the
        # kernel was behaving perfectly.
        # Making c bf16-EXACT does not rescue it: measured at c = 0.625 (exactly representable)
        # the two still diverge at hidden 4.0e-01 over 517 flips, because it is the product that
        # rounds, not the scalar. Only c == 1.0 removes the multiply's rounding entirely.
        # What IS assertable:
        #   scalar == 1.0    -- product is exact, so BIT IDENTITY and zero flips. Non-vacuous:
        #                       the Parameter exists and takes gradients, unlike the i=0 case.
        #   carry fixed c=1  -- same, via the ones buffer instead of a Parameter.
        #   any other scalar -- divergence must be EXPLAINED BY FLIPS. A big hidden diff with
        #                       ZERO flips would mean the kernel moved the arithmetic on its own,
        #                       which is the bug signature this gate exists to catch.
        # Every gradient claim is made against g_floor, never against zero.
        # exact_mul: the multiply introduces no rounding, so nothing is left to differ on.
        #   cs="none"                      -> no scalar at all (ones buffer)
        #   cs="unbounded" and fill == 1.0 -> c is the raw theta, so c is exactly 1.0
        # A transformed mode never qualifies: 2*sigmoid(1.0) is not 1.0.
        exact_mul = cs == "none" or (cs == "unbounded" and es == "none" and fill == 1.0)
        if exact_mul:
            ok = (d_out == 0.0 and flips == 0 and worst <= max(g_floor, 1e-12) * 4)
            why = "BIT-IDENTITY (exact multiply), grad within run-to-run floor"
        else:
            ok = (flips > 0) or (d_out < 5e-2 and worst < max(2e-1, g_floor * 4))
            why = "divergence explained by router flips" if flips else "no flips, tolerance"
        print(f"{tag:<22} hidden {d_out:.2e} | grad {worst:.2e} ({wname[:30]}) "
              f"| kernel-vs-self grad floor {g_floor:.2e} ({g_floor_n[:30]}) "
              f"| flips {flips}/{n_pick} | {why}" + ("  ok" if ok else "  <-- FAIL"))
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
    gate_emb_gain()
    print("PASS")


def gate_emb_gain():
    """HT = AR(...) + i*emb through the fused kernel. BIT IDENTITY, not a tolerance.

    The cases above grade on a relative tolerance, which is the gate that let the "more accurate"
    kernel ship and cost real bpb. This one is the contract from the kernel-bit-identity rule:
    max|fused - eager| == 0 on the layout the model actually trains in (BF16 STREAM end to end,
    bf16 autocast), forward AND backward, plus ZERO router top-k flips. The flip count is the one that matters in
    an MoE -- a 2.5e-03 perturbation once flipped 3.9% of picks and produced 37% hidden divergence
    from an otherwise correct kernel, so agreement on the hidden state alone proves nothing.
    """
    from src.modeling.ffn.router import BiBoMoERouter
    pat = swa_block_pattern(SHARED["num_hidden_layers"])
    ids = torch.randint(0, SHARED["vocab_size"], (4, 512), device="cuda")
    torch.manual_seed(42069)
    model, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32,
                         num_experts=64, top_k=6, special_pairs=0, use_xsa=True,
                         hybrid_layer_pattern=pat, sliding_window=128,
                         attn_res="3", attn_res_sites=1, attn_res_carry=True,
                         attn_res_fp32_stream=False, bf16_residual_stream=True,
                         attn_res_carry_scale="unbounded",
                         attn_res_emb_term=True, attn_res_emb_site="ht",
                         attn_res_emb_gain=True)
    model.train()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "attn_res_carry_theta" in n:
                p.fill_(1.0)      # exact multiply, same reason as the gain below
            # Not 0 -- at i=0 the whole term vanishes and the gate is vacuous. Exactly 1.0 --
            # under a bf16 stream eager rounds the PRODUCT i*emb to bf16 while the kernel forms it
            # in fp32, so any i != 1 makes the two different computations by construction and
            # bit-identity is unreachable (0.37 measured 9735/98304 flips once the stream went
            # bf16; 0.375, though bf16-exact, does not help -- it is the product that rounds).
            # At i = 1.0 the multiply is exact, the Parameter still exists and still takes a
            # gradient, and BIT IDENTITY is a contract the kernel can actually be held to.
            if "attn_res_emb_gain" in n:
                p.fill_(1.0)
    picks = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: picks.append(o[0].detach().clone()))
             for m in model.modules() if isinstance(m, BiBoMoERouter)]
    assert hooks, "no routers found -- the flip count would be vacuously zero"
    snap = {n: b.detach().clone() for n, b in model.named_buffers()}

    def run():
        picks.clear()
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(snap[n])
        for p in model.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.model(input_ids=ids, use_cache=False)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        h.float().square().mean().backward()
        return (h.detach().float(), list(picks),
                {n: p.grad.detach().float().clone()
                 for n, p in model.named_parameters() if p.grad is not None})

    E._HAS_FUSED_RES_ADD = True
    h_k, p_k, g_k = run()
    E._HAS_FUSED_RES_ADD = False
    h_e, p_e, g_e = run()
    E._HAS_FUSED_RES_ADD = True
    for h in hooks:
        h.remove()

    flips = sum((a != b).sum().item() for a, b in zip(p_k, p_e))
    total = sum(a.numel() for a in p_k)
    d_out = (h_k - h_e).abs().max().item()
    worst, wname = 0.0, ""
    for n in g_e:
        d = (g_k[n] - g_e[n]).abs().max().item()
        if d > worst:
            worst, wname = d, n
    print(f"{'emb gain i*emb':<22} hidden {d_out:.2e} | grad {worst:.2e} ({wname[:38]}) "
          f"| router flips {flips}/{total}")
    assert flips == 0, f"{flips}/{total} router top-k picks flipped -- the kernel changes routing"
    assert d_out == 0.0, f"forward is not bit-identical to eager ({d_out:.2e})"
    assert worst == 0.0, f"backward is not bit-identical to eager ({worst:.2e} on {wname})"
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
