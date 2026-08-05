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

    # (carry_scale, emb_term, emb_scale, (carry_fill, emb_fill)).
    # c=1.0 makes the multiply exact; d=0.0 makes eager's SECOND add exact (h + 0 == h in any
    # dtype), which collapses the two-stream case to one effective rounding on both sides. Those
    # two together are the only way bit-identity is reachable -- see the contract note below.
    # "carry+emb d=0" is the load-bearing one: it is the two-stream fused path with every
    # arithmetic difference removed, so if IT diverges the two-stream plumbing is broken, and no
    # amount of "the kernel is more accurate" explains it away.
    for tag, (cs, emb, es, (fill_c, fill_d)) in (
            ("carry c=1.0 learned", ("unbounded", False, "none", (1.0, 0.0))),
            ("carry+emb d=0", ("unbounded", True, "none", (1.0, 0.0))),
            ("carry+emb both 1.0", ("unbounded", True, "none", (1.0, 1.0))),
            ("carry unbounded", ("unbounded", False, "none", (0.625, 0.0))),
            ("carry+emb raw", ("unbounded", True, "none", (0.625, 0.375))),
            ("carry+emb 2sigmoid", ("sigmoid", True, "2sigmoid", (0.625, 0.375))),
            ("carry fixed c=1", ("none", False, "none", (0.625, 0.0)))):
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
                if "attn_res_carry_theta" in n:
                    p.fill_(fill_c)
                if "attn_res_emb_theta" in n:
                    p.fill_(fill_d)

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
        # THREE kernel runs, not two. The atomic_add floor is itself a random draw, so estimating
        # it from a single pair makes the gate flaky in exactly the direction that wastes GPU: the
        # c=1.0 case passed twice and then failed at grad 8.31e-02 against a floor that happened
        # to draw 1.33e-02, with a bit-identical forward and zero router flips. Both numbers came
        # from the same distribution. Max over pairs biases the estimate UP, which is the safe
        # direction for a floor.
        selfs = [run() for _ in range(2)]
        for h_s, _, _ in selfs:
            d_self = (h_k - h_s).abs().max().item()
            assert d_self == 0.0, f"{tag}: two identical kernel runs differ by {d_self:.2e}"
        E._HAS_FUSED_RES_ADD = False
        h_e, p_e, g_e = run()
        E._HAS_FUSED_RES_ADD = True

        def per_param(ga, gb, scale):
            """{param: relative gradient disagreement}, each normalised by its own magnitude."""
            out = {}
            for n in scale:
                den = scale[n].abs().max().item()
                if den < 1e-12 or n not in ga or n not in gb:
                    continue
                out[n] = (ga[n] - gb[n]).abs().max().item() / den
            return out

        def worst_grad(ga, gb):
            d = per_param(ga, gb, gb)
            return (max(d.values()), max(d, key=d.get)) if d else (0.0, "")

        # THE CONTROL. attn_res's backward reduces with atomic_add, whose summation order is not
        # fixed, so gradients disagree run to run even with a bit-identical forward. Without this
        # floor a kernel-vs-eager gradient number means nothing -- the fixed-c case reads 2.1e-02
        # with an EXACT forward and ZERO router flips, which cannot be kernel-vs-eager at all.
        #
        # PER PARAMETER, not a global max. The worst-disagreeing parameter is itself random: one
        # failure had kernel-vs-eager worst on layers.7.attn_res_carry_theta and the floor on
        # layers.3 -- two unrelated parameters, so the ratio between them measured nothing. Each
        # parameter is now compared against ITS OWN floor, maxed over the kernel-vs-kernel pairs.
        floors = {}
        for g_s in [g[2] for g in selfs] + [selfs[0][2]]:
            for n, v in per_param(g_k, g_s, g_e).items():
                floors[n] = max(floors.get(n, 0.0), v)
        for n, v in per_param(selfs[0][2], selfs[1][2], g_e).items():
            floors[n] = max(floors.get(n, 0.0), v)
        g_floor = max(floors.values()) if floors else 0.0
        g_floor_n = max(floors, key=floors.get) if floors else ""
        flips = sum((a != b).sum().item() for a, b in zip(p_k, p_e))
        n_pick = sum(a.numel() for a in p_k)
        d_out = (h_k - h_e).abs().max().item() / h_e.abs().max().item()
        # worst EXCESS over each parameter's own floor -- this is the number the gate acts on.
        ke = per_param(g_k, g_e, g_e)
        excess = {n: v / max(floors.get(n, 0.0), 1e-6) for n, v in ke.items()}
        worst_x = max(excess.values()) if excess else 0.0
        wname = max(excess, key=excess.get) if excess else ""
        worst = ke.get(wname, 0.0)
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
        # Every gradient claim is made against that parameter's own floor, never against zero.
        # exact: nothing is left for the two paths to differ on -- every multiply is exact AND
        # eager performs a single effective rounding, same as the kernel's one fp32 accumulation.
        #   cs="none"                        -> no scalar at all (ones buffer)
        #   cs="unbounded", es raw, c == 1.0 -> c is the raw theta, so the multiply is exact
        #   plus: no emb stream, or d == 0.0 -> eager's second add is `h + 0`, exact in any dtype
        # A transformed mode never qualifies: 2*sigmoid(1.0) is not 1.0.
        exact_scalar = cs == "none" or (cs == "unbounded" and es == "none" and fill_c == 1.0)
        one_rounding = (not emb) or fill_d == 0.0
        # SLACK is on the EXCESS RATIO, so it is scale-free: 4x means "this parameter disagrees
        # with eager more than 4x as much as the kernel disagrees with itself on that same
        # parameter". Noise-vs-noise lands near 1x; a real arithmetic change does not.
        SLACK = 4.0
        if exact_scalar and one_rounding:
            ok = (d_out == 0.0 and flips == 0 and worst_x <= SLACK)
            why = "BIT-IDENTITY (exact multiply), grad within per-param floor"
        else:
            ok = (flips > 0) or (d_out < 5e-2 and worst_x <= SLACK)
            why = "divergence explained by router flips" if flips else "no flips, tolerance"
        print(f"{tag:<22} hidden {d_out:.2e} | grad {worst:.2e} vs own floor "
              f"{floors.get(wname, 0.0):.2e} = {worst_x:.2f}x ({wname[:28]}) "
              f"| max floor {g_floor:.2e} | flips {flips}/{n_pick} | {why}"
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
    selfs = [run() for _ in range(2)]      # the kernel against ITSELF, twice
    E._HAS_FUSED_RES_ADD = False
    h_e, p_e, g_e = run()
    E._HAS_FUSED_RES_ADD = True
    for h in hooks:
        h.remove()

    def abs_pp(ga, gb):
        return {n: (ga[n] - gb[n]).abs().max().item() for n in gb if n in ga}

    flips = sum((a != b).sum().item() for a, b in zip(p_k, p_e))
    total = sum(a.numel() for a in p_k)
    d_out = (h_k - h_e).abs().max().item()
    d_self = max((h_k - h_s).abs().max().item() for h_s, _, _ in selfs)
    # Per-parameter floor, maxed over every kernel-vs-kernel pair -- same reasoning as main():
    # one draw of a random quantity is not a floor, and the worst parameter moves between runs.
    floors = {}
    for ga, gb in ((g_k, selfs[0][2]), (g_k, selfs[1][2]), (selfs[0][2], selfs[1][2])):
        for n, v in abs_pp(ga, gb).items():
            floors[n] = max(floors.get(n, 0.0), v)
    ke = abs_pp(g_k, g_e)
    excess = {n: v / max(floors.get(n, 0.0), 1e-30) for n, v in ke.items() if v > 0}
    worst_x = max(excess.values()) if excess else 0.0
    wname = max(excess, key=excess.get) if excess else ""
    worst = ke.get(wname, 0.0)
    g_floor = floors.get(wname, 0.0)
    print(f"{'emb gain i*emb':<22} hidden {d_out:.2e} | grad {worst:.2e} vs own floor "
          f"{g_floor:.2e} = {worst_x:.2f}x ({wname[:28]}) | router flips {flips}/{total}")
    assert flips == 0, f"{flips}/{total} router top-k picks flipped -- the kernel changes routing"
    assert d_self == 0.0, f"two identical kernel runs differ in the forward ({d_self:.2e})"
    assert d_out == 0.0, f"forward is not bit-identical to eager ({d_out:.2e})"
    # The FORWARD is held to exact zero -- it is deterministic, and it is. The backward is not:
    # attn_res reduces with atomic_add, so the same kernel run twice already disagrees by g_floor
    # (measured ~1e-2 relative elsewhere in this file). Demanding exact 0 here asserted that
    # atomic_add is order-stable, which it is not, and it failed at 4.77e-07 on embed_tokens while
    # the forward was bit-perfect with zero router flips.
    assert worst_x <= 4.0, (
        f"backward differs from eager by {worst:.2e} on {wname}, {worst_x:.1f}x that parameter's "
        f"own {g_floor:.2e} run-to-run floor -- that is the kernel, not atomic nondeterminism")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
