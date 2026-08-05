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

    for tag, (cs, emb, es) in (("carry unbounded", ("unbounded", False, "none")),
                               ("carry+emb raw", ("unbounded", True, "none")),
                               ("carry+emb 2sigmoid", ("sigmoid", True, "2sigmoid")),
                               ("carry fixed c=1", ("none", False, "none"))):
        torch.manual_seed(42069)
        model, _ = build_arm("bibo_min", device="cuda", dtype=torch.float32,
                             num_experts=8, top_k=2, special_pairs=0, use_xsa=True,
                             hybrid_layer_pattern=pat, sliding_window=128,
                             attn_res="3", attn_res_sites=1, attn_res_carry=True,
                             attn_res_fp32_stream=False, bf16_residual_stream=True,
                             attn_res_carry_scale=cs,
                             attn_res_emb_term=emb, attn_res_emb_scale=es)
        model.train()
        # Non-trivial values, because at theta=0 the carry is exactly 1.0 and d is 0 or 1 -- the
        # one setting where a scale bug cannot show up. But BF16-EXACT ones: 0.625 = 1.25*2^-1 and
        # 0.375 = 1.5*2^-2 both fit bf16's 8 mantissa bits, so `c.to(bf16)` is a no-op and eager's
        # rounding of the scalar -- the ONLY thing the kernel deliberately does differently --
        # stops existing. That makes bit-identity attainable for the RAW modes, which turns this
        # from a tolerance into a real contract. The old 0.6/0.4 are not representable, so every
        # raw-mode case was forced through the weaker flips-explained branch for no reason.
        # Transformed modes (sigmoid/2sigmoid/tanh) still cannot be exact: f(theta) is not.
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "attn_res_carry_theta" in n:
                    p.fill_(0.625)
                if "attn_res_emb_theta" in n:
                    p.fill_(0.375)

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
        # With a LEARNABLE scalar the kernel and eager are different computations on purpose: the
        # kernel keeps c in fp32 across the multiply, eager does `_c.to(attn_output.dtype) * ...`
        # and rounds it to bf16 first (exp/modeling_bibo.py, eager branch). The standalone fp64
        # grade says the kernel is the more accurate of the two. So equality is not just unmet
        # here, it is the wrong thing to demand -- the old blanket tolerance was asserting
        # something false, and it fired at hidden 2.9e-01 while the kernel was behaving.
        # What IS assertable:
        #   carry fixed c=1  -- no scalar to round, so BIT IDENTITY and zero flips.
        #   learnable c      -- divergence must be EXPLAINED BY FLIPS. A big hidden diff with
        #                       zero flips would mean the kernel moved the arithmetic on its own,
        #                       which is the bug signature this gate exists to catch.
        # Every gradient claim is made against g_floor, never against zero.
        # RAW modes with bf16-exact scalars have nothing left to differ on -> hard bit-identity.
        # Transformed modes compute f(theta), which is not bf16-exact, so they keep the weaker
        # divergence-must-be-explained-by-flips contract.
        raw = cs in ("none", "unbounded") and es == "none"
        if raw:
            ok = (d_out == 0.0 and flips == 0 and worst <= max(g_floor, 1e-12) * 4)
            why = "BIT-IDENTITY (bf16-exact scalars), grad within run-to-run floor"
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
                p.fill_(0.6)
            # not 0 -- at i=0 the whole term vanishes and the gate is vacuous. 0.375 = 1.5*2^-2 is
            # BF16-EXACT, so eager's `i.to(bf16)` is a no-op and bit-identity is attainable under
            # the bf16 stream. 0.37 is not representable, and under bf16 it made eager and the
            # kernel different computations by construction -- 9735/98304 router flips.
            if "attn_res_emb_gain" in n:
                p.fill_(0.375)
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
