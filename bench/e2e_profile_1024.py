"""torch.profiler op breakdown of the full BiBo step at B1xS1024 (the small regime where all+CE
is comparable to baseline/std on TIME but saves memory). Shows WHERE time goes: kernels save on
attn/MoE/norm; fused CE adds back the loss-path GEMMs (3-GEMM backward) — net ~comparable, less mem.
Fresh process per config (no fragmentation). CPU+CUDA activities (torch quirk)."""
import os, sys
os.environ["WANDB_MODE"] = "disabled"; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _REPO)

def prof_one(level, ce):
    import yaml, torch, torch.profiler as P
    from bench.models import build_model_from_config
    cfg = yaml.safe_load(open(os.path.join(_REPO, "bench/configs/bibo.yaml")))
    torch.manual_seed(0); m, c = build_model_from_config(cfg)
    if level == "all":
        from src.kernels.patch import patch_bibo_with_triton
        from src.kernels.moe_grouped import patch_moe_auto
        from src.kernels.dense_mlp import patch_dense_mlp_with_triton
        from src.kernels.conv_fused import patch_conv_router_with_triton, patch_conv_expert_with_triton
        patch_bibo_with_triton(m); patch_moe_auto(m); patch_dense_mlp_with_triton(m)
        patch_conv_router_with_triton(m); patch_conv_expert_with_triton(m)
    m.config.use_fused_linear_ce = bool(ce); m = m.cuda().train()
    x = torch.randint(0, 81000, (1, 1024), device="cuda"); lab = torch.randint(0, 81000, (1, 1024), device="cuda")
    def step():
        for p in m.parameters(): p.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(x, labels=lab); loss = out.loss
        loss.backward()
    for _ in range(6): step()
    torch.cuda.synchronize()
    with P.profile(activities=[P.ProfilerActivity.CPU, P.ProfilerActivity.CUDA]) as pr:
        step(); torch.cuda.synchronize()
    tag = f"{level}/{'fusedCE' if ce else 'stdCE'}"
    print(f"\n===== {tag}  B1xS1024  fwd+bwd =====")
    print(pr.key_averages().table(sort_by="self_cuda_time_total", row_limit=16))

if len(sys.argv) > 1 and sys.argv[1] == "--one":
    prof_one(sys.argv[2], int(sys.argv[3])); sys.exit(0)

import subprocess
for level, ce in (("baseline", 0), ("all", 1)):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", level, str(ce)], text=True, capture_output=True)
    out = "\n".join(l for l in p.stdout.splitlines() if "===" in l or "%" in l or "self_cuda" in l.lower()
                     or any(k in l for k in ("aten::", "_FLCE", "Self CUDA", "Name", "----")))
    print(out)
    if p.returncode != 0: print("ERR:", p.stderr[-800:])
