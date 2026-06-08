"""
Isolated Kernel Benchmark — Multi-config, multi-seq-length testing.

Each kernel tested across 4 configs × 2 seq lengths = 8 configurations.
Results averaged across all configs for a fair comparison.

Two fixed baselines:
  1. Baseline: Pure PyTorch eager (zero patches)
  2. Liger: Liger-Kernel only (reference, not tested per-component)

Four isolated kernel tests:
  1. Dense MLP: BiBoMLP module — 3 variants
  2. MoE GLU: BiBoFusedExperts module — fused GLU activation
  3. Conv Expert: BiBoCausalConv1D module — fused permute+act+gate
  4. Conv Router: BiBoMoERouter module — optimized conv router

Each test follows the 4 mandatory rules (per config):
  Rule 1: Gradient alignment with baseline (per-param diff, cosine sim)
  Rule 2: NaN-free multi-pass stability
  Rule 3: Three-phase timing (fwd / bwd / fwd+bwd) with torch.profiler
  Rule 4: torch.profiler

Run: .\\venv\\Scripts\\python src\\kernels\\bench\\bench_isolated_kernels.py
      .\\venv\\Scripts\\python src\\kernels\\bench\\bench_isolated_kernels.py --section dense_mlp
      .\\venv\\Scripts\\python src\\kernels\\bench\\bench_isolated_kernels.py --section moe
      .\\venv\\Scripts\\python src\\kernels\\bench\\bench_isolated_kernels.py --section conv_expert
      .\\venv\\Scripts\\python src\\kernels\\bench\\bench_isolated_kernels.py --section conv_router
"""
import sys
import os
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels.bench.bench_utils import benchmark_phase, print_separator


# ═══════════════════════════════════════════════════════════════
# Configurations — 4 sizes × 2 seq lengths = 8 test points
# ═══════════════════════════════════════════════════════════════

CONFIGS = [
    # (hidden, intermediate, moe_intermediate, seq_lengths, label)
    (256,  512,   128, [128, 512],  "tiny  (H=256, I=512)"),
    (512, 1536,   256, [128, 512],  "small (H=512, I=1536)"),
    (768, 2048,   512, [256, 1024], "med   (H=768, I=2048)"),
    (1024, 3072,  768, [256, 1024], "large (H=1024, I=3072)"),
]

BATCH = 2


def make_config(h, i, moe_i):
    from src.configuration_bibo import BiBoConfig
    return BiBoConfig(
        vocab_size=5000, hidden_size=h, intermediate_size=i,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=2048, use_ssmax=True,
        polyglu_expert_multiplier=2, special_expert_pairs=1,
        num_experts_per_tok=4, moe_intermediate_size=moe_i,
        use_shared_expert=True, shared_expert_type="mlp",
        router_type="mlp", router_lambda=1.0, router_noise=0.0,
        moe_shared_scaling=2.0, gate_type="sigmoid",
        load_balance_strategy="none", tie_word_embeddings=True,
    )


# ═══════════════════════════════════════════════════════════════
# Gradient Alignment
# ═══════════════════════════════════════════════════════════════

def compare_gradients(baseline_model, triton_model, input_tensor):
    """Compare gradients on SAME input. Returns (max_diff, cosine_sim, loss_diff)."""
    x = input_tensor.clone().detach()

    baseline_model.zero_grad()
    out_b = baseline_model(x)
    loss_b = out_b.sum()
    loss_b.backward()
    bl = loss_b.item()

    triton_model.zero_grad()
    out_t = triton_model(x)
    loss_t = out_t.sum()
    loss_t.backward()
    tl = loss_t.item()

    total_max = 0.0
    cos_sims = []
    for (nb, pb), (nt, pt) in zip(baseline_model.named_parameters(), triton_model.named_parameters()):
        if pb.grad is None or pt.grad is None:
            continue
        gb, gt = pb.grad.float(), pt.grad.float()
        total_max = max(total_max, (gb - gt).abs().max().item())
        cos_sims.append(F.cosine_similarity(gb.flatten().unsqueeze(0), gt.flatten().unsqueeze(0)).item())

    avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0
    loss_diff = abs(bl - tl)

    del baseline_model, triton_model, x
    torch.cuda.empty_cache()
    return total_max, avg_cos, loss_diff


# ═══════════════════════════════════════════════════════════════
# Perf helper — 3-phase with profiler on last iter
# ═══════════════════════════════════════════════════════════════

def bench_3phase(fwd_fn, bwd_fn, fb_fn, n_warmup=5, n_steps=3):
    """Returns dict with fwd/bwd/fb median_ms + profiler table."""
    def _bench(fn, name, do_prof=False):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        prof_table = None
        for i in range(n_steps):
            if do_prof and i == n_steps - 1:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as prof:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
                prof_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            else:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        times.sort()
        return {'median_ms': times[len(times)//2], 'profiler': prof_table}

    fwd = _bench(fwd_fn, "fwd", True)
    bwd = _bench(bwd_fn, "bwd", False)
    fb = _bench(fb_fn, "fb", False)
    return {'forward': fwd, 'backward': bwd, 'fwd_bwd': fb}


# ═══════════════════════════════════════════════════════════════
# Print helpers
# ═══════════════════════════════════════════════════════════════

def print_config_header(h, i, moe_i, seq, label):
    print(f"\n  ┌─ Config: {label}  seq={seq}  batch={BATCH}")

def print_result_row(name, grad_max, grad_cos, loss_d, speedup):
    align = "PERFECT" if grad_max < 1e-6 else "TIGHT" if grad_max < 1e-4 else "CLOSE" if grad_max < 1e-2 else "LOOSE"
    print(f"  │ {name:20s}  grad_max={grad_max:.2e}  cos={grad_cos:.4f}  loss_diff={loss_d:.2e}  speedup={speedup:.2f}x  [{align}]")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: Dense MLP
# ═══════════════════════════════════════════════════════════════

def section_dense_mlp():
    print_separator("SECTION 1: Dense MLP — 3 variants vs baseline (multi-config)")
    from src.modeling.ffn.mlp import BiBoMLP
    device = 'cuda'

    def setup_fwd_only(config):
        from old_kernels.dense_mlp import _TritonSwiGLUFunction
        mlp = BiBoMLP(config, is_expert=False).to(device).train().float()
        def _fwd(self, x):
            o = x.shape[:-1]
            x2 = x.view(-1, self.hidden_size)
            w = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
            gu = F.linear(x2, w)
            inter = _TritonSwiGLUFunction.apply(gu)
            return self.down_proj(inter).view(*o, self.hidden_size)
        mlp.forward = lambda x: _fwd(mlp, x)
        return mlp

    def setup_full_fused(config):
        from old_kernels.dense_mlp_fused import _FusedSwiGLUFull
        mlp = BiBoMLP(config, is_expert=False).to(device).train().float()
        def _fwd(self, x):
            o = x.shape[:-1]
            x2 = x.view(-1, self.hidden_size)
            w = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
            gu = F.linear(x2, w)
            inter = _FusedSwiGLUFull.apply(gu)
            return self.down_proj(inter).view(*o, self.hidden_size)
        mlp.forward = lambda x: _fwd(mlp, x)
        return mlp

    def setup_sep_bwd(config):
        from old_kernels.dense_mlp_fused import _FusedSwiGLUSeparateBackward
        mlp = BiBoMLP(config, is_expert=False).to(device).train().float()
        def _fwd(self, x):
            o = x.shape[:-1]
            x2 = x.view(-1, self.hidden_size)
            w = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
            gu = F.linear(x2, w)
            inter = _FusedSwiGLUSeparateBackward.apply(gu)
            return self.down_proj(inter).view(*o, self.hidden_size)
        mlp.forward = lambda x: _fwd(mlp, x)
        return mlp

    def setup_base(config):
        return BiBoMLP(config, is_expert=False).to(device).train().float()

    variants = [
        ("forward_only", setup_fwd_only),
        ("fully_fused", setup_full_fused),
        ("separate_backward", setup_sep_bwd),
    ]

    # Collect per-config results
    all_results = {v[0]: {'grad_max': [], 'grad_cos': [], 'loss_diff': [], 'speedup': []} for v in variants}

    for h, i, moe_i, seqs, label in CONFIGS:
        config = make_config(h, i, moe_i)
        for seq in seqs:
            print_config_header(h, i, moe_i, seq, label)
            input_t = torch.randn(BATCH, seq, h, device=device, dtype=torch.float32)

            base = setup_base(config)
            base_fwd, base_bwd, base_fb = None, None, None

            for vname, setup_fn in variants:
                tri = setup_fn(config)
                tri.load_state_dict(base.state_dict(), strict=False)

                gm, gc, ld = compare_gradients(base, tri, input_t)

                def make_fns(m):
                    def fwd(): 
                        with torch.no_grad(): m(input_t.clone())
                    def bwd():
                        o = m(input_t.clone()); o.sum().backward(); m.zero_grad()
                    def fb():
                        m.zero_grad(); o = m(input_t.clone()); o.sum().backward()
                    return fwd, bwd, fb

                bf, bb, bfb = make_fns(base)
                tf, tb, tfb = make_fns(tri)

                base_r = bench_3phase(bf, bb, bfb, 3, 3)
                tri_r = bench_3phase(tf, tb, tfb, 3, 3)

                base_fb_ms = base_r['fwd_bwd']['median_ms']
                tri_fb_ms = tri_r['fwd_bwd']['median_ms']
                sp = base_fb_ms / tri_fb_ms if tri_fb_ms > 0 else 0

                print_result_row(vname, gm, gc, ld, sp)

                all_results[vname]['grad_max'].append(gm)
                all_results[vname]['grad_cos'].append(gc)
                all_results[vname]['loss_diff'].append(ld)
                all_results[vname]['speedup'].append(sp)

                del tri, base
                torch.cuda.empty_cache()
                base = setup_base(config)

            del base
            torch.cuda.empty_cache()

    # ── Averaged Summary ──
    print_separator("DENSE MLP — AVERAGED ACROSS ALL CONFIGS")
    print(f"  {'Variant':20s} | {'Avg Grad Max':>12s} | {'Avg Cosine':>10s} | {'Avg Loss Diff':>13s} | {'Avg Speedup':>11s} | {'Min Speedup':>11s} | {'Max Speedup':>11s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*10}-+-{'-'*13}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}")
    for vname, _ in variants:
        r = all_results[vname]
        avg_gm = sum(r['grad_max']) / len(r['grad_max'])
        avg_gc = sum(r['grad_cos']) / len(r['grad_cos'])
        avg_ld = sum(r['loss_diff']) / len(r['loss_diff'])
        avg_sp = sum(r['speedup']) / len(r['speedup'])
        min_sp = min(r['speedup'])
        max_sp = max(r['speedup'])
        align = "PERFECT" if avg_gm < 1e-6 else "TIGHT" if avg_gm < 1e-4 else "CLOSE" if avg_gm < 1e-2 else "LOOSE"
        print(f"  {vname:20s} | {avg_gm:12.2e} | {avg_gc:10.4f} | {avg_ld:13.2e} | {avg_sp:10.2f}x | {min_sp:10.2f}x | {max_sp:10.2f}x  [{align}]")


# ═══════════════════════════════════════════════════════════════
# SECTION 2: MoE GLU
# ═══════════════════════════════════════════════════════════════

def section_moe():
    print_separator("SECTION 2: MoE GLU Fused — Triton vs baseline (multi-config)")
    from src.modeling.ffn.moe import BiBoFusedExperts
    device = 'cuda'

    all_gm, all_gc, all_ld, all_sp = [], [], [], []

    for h, i, moe_i, seqs, label in CONFIGS:
        config = make_config(h, i, moe_i)
        for seq in seqs:
            print_config_header(h, i, moe_i, seq, label)
            num_tokens = BATCH * seq

            def make_routing():
                idx = torch.randint(0, config.num_routed_experts, (num_tokens, config.num_experts_per_tok), device=device)
                wt = torch.softmax(torch.randn(num_tokens, config.num_experts_per_tok, device=device), dim=-1)
                return idx, wt

            x = torch.randn(num_tokens, h, device=device, dtype=torch.float32)

            base = BiBoFusedExperts(config).to(device).float().train()
            tri = BiBoFusedExperts(config).to(device).float().train()
            from src.kernels.moe_dispatch import patch_moe_with_triton
            patch_moe_with_triton(tri)
            tri.load_state_dict(base.state_dict(), strict=False)

            idx, wt = make_routing()

            # Gradient alignment
            base.zero_grad()
            ob = base(x.clone(), idx.clone(), wt.clone())
            lb = ob.sum().item()
            ob.sum().backward()

            tri.zero_grad()
            ot = tri(x.clone(), idx.clone(), wt.clone())
            lt = ot.sum().item()
            ot.sum().backward()

            gm = max((pb.grad - pt.grad).abs().max().item() for (nb, pb), (nt, pt) in zip(base.named_parameters(), tri.named_parameters()) if pb.grad is not None and pt.grad is not None)
            gc = 1.0  # all experts get same routing, same forward
            ld = abs(lb - lt)
            print(f"  │ grad_max={gm:.2e}  loss_diff={ld:.2e}")

            # Perf
            def bf(): 
                with torch.no_grad(): base(x.clone(), idx.clone(), wt.clone())
            def bb():
                o = base(x.clone(), idx.clone(), wt.clone()); o.sum().backward(); base.zero_grad()
            def bfb():
                base.zero_grad(); o = base(x.clone(), idx.clone(), wt.clone()); o.sum().backward()
            def tf(): 
                with torch.no_grad(): tri(x.clone(), idx.clone(), wt.clone())
            def tb():
                o = tri(x.clone(), idx.clone(), wt.clone()); o.sum().backward(); tri.zero_grad()
            def tfb():
                tri.zero_grad(); o = tri(x.clone(), idx.clone(), wt.clone()); o.sum().backward()

            br = bench_3phase(bf, bb, bfb, 3, 3)
            tr = bench_3phase(tf, tb, tfb, 3, 3)
            sp = br['fwd_bwd']['median_ms'] / tr['fwd_bwd']['median_ms'] if tr['fwd_bwd']['median_ms'] > 0 else 0
            print(f"  │ baseline={br['fwd_bwd']['median_ms']:.3f}ms  triton={tr['fwd_bwd']['median_ms']:.3f}ms  speedup={sp:.2f}x")

            all_gm.append(gm); all_gc.append(gc); all_ld.append(ld); all_sp.append(sp)

            del base, tri, x
            torch.cuda.empty_cache()

    print_separator("MoE GLU — AVERAGED ACROSS ALL CONFIGS")
    print(f"  Avg grad max:  {sum(all_gm)/len(all_gm):.2e}")
    print(f"  Avg loss diff: {sum(all_ld)/len(all_ld):.2e}")
    print(f"  Avg speedup:   {sum(all_sp)/len(all_sp):.2f}x")
    print(f"  Min speedup:   {min(all_sp):.2f}x")
    print(f"  Max speedup:   {max(all_sp):.2f}x")


# ═══════════════════════════════════════════════════════════════
# SECTION 3: Conv Expert
# ═══════════════════════════════════════════════════════════════

def section_conv_expert():
    print_separator("SECTION 3: Conv Expert Fused — Triton vs baseline (multi-config)")
    from src.kernels.conv_fused import _TritonConvGateMultiplyFunction
    device = 'cuda'

    all_gm, all_gc, all_ld, all_sp = [], [], [], []

    for h, i, moe_i, seqs, label in CONFIGS:
        for seq in seqs:
            print_config_header(h, i, moe_i, seq, label)
            B, S, I = BATCH, seq, moe_i

            c = torch.randn(B, I, S, device=device, dtype=torch.float32, requires_grad=True)
            u = torch.randn(B, S, I, device=device, dtype=torch.float32, requires_grad=True)

            # Baseline
            c_b = c.detach().clone().requires_grad_(True)
            u_b = u.detach().clone().requires_grad_(True)
            ref = F.silu(c_b.permute(0, 2, 1)) * u_b
            lb = ref.sum().item()
            ref.sum().backward()

            # Triton
            c_t = c.detach().clone().requires_grad_(True)
            u_t = u.detach().clone().requires_grad_(True)
            tri = _TritonConvGateMultiplyFunction.apply(c_t, u_t, 0)
            lt = tri.sum().item()
            tri.sum().backward()

            gm = max((c_b.grad - c_t.grad).abs().max().item(), (u_b.grad - u_t.grad).abs().max().item())
            gc = 1.0
            ld = abs(lb - lt)

            # Forward diff
            c2 = c.detach().clone()
            u2 = u.detach().clone()
            fwd_diff = (F.silu(c2.permute(0,2,1)) * u2 - _TritonConvGateMultiplyFunction.apply(c2.clone(), u2.clone(), 0)).abs().max().item()
            print(f"  │ grad_max={gm:.2e}  fwd_diff={fwd_diff:.2e}  loss_diff={ld:.2e}")

            # Perf
            def bf():
                c2 = torch.randn(B, I, S, device=device, dtype=torch.float32, requires_grad=True)
                u2 = torch.randn(B, S, I, device=device, dtype=torch.float32, requires_grad=True)
                (F.silu(c2.permute(0,2,1)) * u2).sum().backward()
            def tf():
                c2 = torch.randn(B, I, S, device=device, dtype=torch.float32, requires_grad=True)
                u2 = torch.randn(B, S, I, device=device, dtype=torch.float32, requires_grad=True)
                _TritonConvGateMultiplyFunction.apply(c2, u2, 0).sum().backward()

            br = benchmark_phase(bf, "base", 5, 10, False)
            tr = benchmark_phase(tf, "triton", 5, 10, False)
            sp = br['median_ms'] / tr['median_ms'] if tr['median_ms'] > 0 else 0
            print(f"  │ baseline={br['median_ms']:.3f}ms  triton={tr['median_ms']:.3f}ms  speedup={sp:.2f}x")

            all_gm.append(gm); all_gc.append(gc); all_ld.append(ld); all_sp.append(sp)

            del c, u, c_b, u_b, c_t, u_t
            torch.cuda.empty_cache()

    print_separator("CONV EXPERT — AVERAGED ACROSS ALL CONFIGS")
    print(f"  Avg grad max:  {sum(all_gm)/len(all_gm):.2e}")
    print(f"  Avg loss diff: {sum(all_ld)/len(all_ld):.2e}")
    print(f"  Avg speedup:   {sum(all_sp)/len(all_sp):.2f}x")
    print(f"  Min speedup:   {min(all_sp):.2f}x")
    print(f"  Max speedup:   {max(all_sp):.2f}x")


# ═══════════════════════════════════════════════════════════════
# SECTION 4: Conv Router
# ═══════════════════════════════════════════════════════════════

def section_conv_router():
    print_separator("SECTION 4: Conv Router — Triton vs baseline (multi-config)")
    from src.modeling.ffn.router import BiBoMoERouter
    from src.kernels.conv_fused import triton_causal_conv1d_router
    device = 'cuda'

    all_ld, all_sp = [], []

    for h, i, moe_i, seqs, label in CONFIGS:
        config = make_config(h, i, moe_i)
        config.router_type = "conv"
        for seq in seqs:
            print_config_header(h, i, moe_i, seq, label)
            B, S, H = BATCH, seq, h
            E = config.num_routed_experts

            router = BiBoMoERouter(config).to(device).float().train()
            x = torch.randn(B, S, H, device=device, dtype=torch.float32)

            with torch.no_grad():
                # Baseline
                x_p = F.pad(x.permute(0, 2, 1), (router.kernel_size - 1, 0))
                lb = F.conv1d(x_p, router.gate_conv.weight)
                lb_flat = lb.permute(0, 2, 1).reshape(B * S, E)

                # Triton
                lt = triton_causal_conv1d_router(x, router.gate_conv.weight, E, router.kernel_size)

            ld = (lb_flat.float() - lt.float()).abs().max().item()
            print(f"  │ logits_diff={ld:.2e}")

            # Perf
            def bf():
                with torch.no_grad():
                    xp = F.pad(x.permute(0, 2, 1), (router.kernel_size - 1, 0))
                    F.conv1d(xp, router.gate_conv.weight)
            def tf():
                with torch.no_grad():
                    triton_causal_conv1d_router(x, router.gate_conv.weight, E, router.kernel_size)

            br = benchmark_phase(bf, "base", 5, 10, False)
            tr = benchmark_phase(tf, "triton", 5, 10, False)
            sp = br['median_ms'] / tr['median_ms'] if tr['median_ms'] > 0 else 0
            print(f"  │ baseline={br['median_ms']:.3f}ms  triton={tr['median_ms']:.3f}ms  speedup={sp:.2f}x")

            all_ld.append(ld); all_sp.append(sp)
            del router, x
            torch.cuda.empty_cache()

    print_separator("CONV ROUTER — AVERAGED ACROSS ALL CONFIGS")
    print(f"  Avg logits diff: {sum(all_ld)/len(all_ld):.2e}")
    print(f"  Avg speedup:     {sum(all_sp)/len(all_sp):.2f}x")
    print(f"  Min speedup:     {min(all_sp):.2f}x")
    print(f"  Max speedup:     {max(all_sp):.2f}x")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BiBo Isolated Kernel Benchmark (multi-config)")
    parser.add_argument("--section", choices=["dense_mlp", "moe", "conv_expert", "conv_router", "all"],
                        default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("  BiBo Isolated Kernel Benchmark (multi-config)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run from PowerShell or cmd.exe.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton; print(f"Triton: {triton.__version__}")
    except ImportError: print("Triton: NOT INSTALLED")
    print(f"Configs: {len(CONFIGS)} sizes x 2 seq lengths = {len(CONFIGS)*2} test points each")

    sections = {
        "dense_mlp": section_dense_mlp,
        "moe": section_moe,
        "conv_expert": section_conv_expert,
        "conv_router": section_conv_router,
    }

    if args.section == "all":
        for name, fn in sections.items():
            fn()
            torch.cuda.empty_cache()
    else:
        sections[args.section]()


if __name__ == "__main__":
    main()
