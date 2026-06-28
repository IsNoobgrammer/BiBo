"""E2E full-model (BiBo) matrix: {baseline, liger-only, all-kernels} x {std CE, fused CE-optmaxx}
across token budgets {1024 x2, 2048 x2, 4096 x2, 8192 x1} as (batch, seq) splits (seq<=2048).

Each cell runs in its OWN subprocess (fresh CUDA allocator) — a single-process run accumulates
allocator fragmentation that poisons later cells (measured: a cell that is 388ms isolated read
5921ms after 30 prior cells). Subprocess isolation is mandatory for trustworthy numbers AND lets
each cell OOM independently. Measures fwd+bwd step ms (cuda.Event, warmup+median) + peak mem.
Optimizer excluded (identical across variants). Clock should be locked (nvidia-smi -lgc).

On the 4GB RTX 3050, >=4096-token cells swap to host memory (numbers meaningless / OOM) — run on
the T4 (16GB) for those. <=2048-token cells are clean here.

Run:    .venv/Scripts/python -u bench/e2e_ce_matrix.py
Worker: .venv/Scripts/python bench/e2e_ce_matrix.py --cell <level> <0|1> <B> <S>   (internal)
"""
import os, sys, subprocess, statistics
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

SHAPES = [(1,1024),(2,512),(1,2048),(2,1024),(2,2048),(4,1024),(8,1024)]
LEVELS = ("baseline", "liger", "all")
# On the 4GB 3050, cap to the clean (non-swapping) shapes: BENCH_MAXTOK=2048. Unset on the T4.
_MAXTOK = int(os.environ.get("BENCH_MAXTOK", "0"))
if _MAXTOK:
    SHAPES = [(b, s) for (b, s) in SHAPES if b * s <= _MAXTOK]


def run_cell(level, ce, B, S):
    """Worker: build one model variant, measure one shape, print 'RESULT <ms> <mem>' or 'RESULT OOM'."""
    import yaml, torch
    from bench.models import build_model_from_config
    cfg = yaml.safe_load(open(os.path.join(_REPO, "bench/configs/bibo.yaml")))
    torch.manual_seed(0)
    m, c = build_model_from_config(cfg)
    if level == "liger":
        from src.kernels.patch import patch_bibo_with_triton
        patch_bibo_with_triton(m)
    elif level == "all":
        from src.kernels.patch import patch_bibo_with_triton
        from src.kernels.moe_grouped import patch_moe_auto
        from src.kernels.dense_mlp import patch_dense_mlp_with_triton
        patch_bibo_with_triton(m); patch_moe_auto(m); patch_dense_mlp_with_triton(m)
    m.config.use_fused_linear_ce = bool(ce)
    m = m.cuda().train()
    x = torch.randint(0, 81000, (B, S), device="cuda")
    lab = torch.randint(0, 81000, (B, S), device="cuda")
    def step():
        for p in m.parameters(): p.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(x, labels=lab)
            loss = out.loss if hasattr(out, "loss") else (out[0] if isinstance(out, tuple) else out)
            if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
    def t1(fn):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)
    try:
        for _ in range(5): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ms = statistics.median([t1(step) for _ in range(12)])
        print(f"RESULT {ms:.2f} {torch.cuda.max_memory_allocated()/1e6:.0f}")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
        print(f"RESULT OOM {type(ex).__name__}")


if len(sys.argv) > 1 and sys.argv[1] == "--cell":
    run_cell(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    sys.exit(0)

# ---- parent: dispatch one subprocess per cell ----
res = {}
for level in LEVELS:
    for ce in (0, 1):
        for (B, S) in SHAPES:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--cell", level, str(ce), str(B), str(S)],
                               capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), "RESULT OOM nostdout")
            tok = line.split()
            ms = float(tok[1]) if tok[1] != "OOM" else None
            mem = float(tok[2]) if (ms is not None and len(tok) > 2) else None
            res[(level, ce, B, S)] = (ms, mem)
            tag = f"{level}/{'fusedCE' if ce else 'stdCE'}"
            print(f"  {tag:20s} B{B}xS{S} ({B*S}tok): " + (f"{ms:8.2f}ms {mem:7.0f}MB" if ms else "OOM"), flush=True)

def c(level, ce, B, S, mem=False):
    ms, mm = res[(level, ce, B, S)]
    v = mm if mem else ms
    return f"{v:6.0f}" if (mem and v) else (f"{v:6.1f}" if v else "  OOM ")

print("\n" + "="*104)
print("STEP TIME (ms, fwd+bwd) | cols: baseline/liger/all x std/fusedCE | last: all+fusedCE vs baseline+stdCE")
print(f"{'BxS (tok)':>13s} | {'base/std':>8s}{'base/CE':>8s} | {'lig/std':>8s}{'lig/CE':>8s} | {'all/std':>8s}{'all/CE':>8s} | {'allCE/baseStd':>13s}")
for (B,S) in SHAPES:
    bstd=res[('baseline',0,B,S)][0]; allce=res[('all',1,B,S)][0]
    spd = f"{bstd/allce:.2f}x" if (bstd and allce) else ("base OOM" if not bstd else "—")
    print(f"{B}x{S}({B*S:5d}) | {c('baseline',0,B,S):>8s}{c('baseline',1,B,S):>8s} | "
          f"{c('liger',0,B,S):>8s}{c('liger',1,B,S):>8s} | {c('all',0,B,S):>8s}{c('all',1,B,S):>8s} | {spd:>13s}")
print("\nPEAK MEMORY (MB)")
print(f"{'BxS (tok)':>13s} | {'base/std':>8s}{'base/CE':>8s} | {'lig/std':>8s}{'lig/CE':>8s} | {'all/std':>8s}{'all/CE':>8s}")
for (B,S) in SHAPES:
    print(f"{B}x{S}({B*S:5d}) | {c('baseline',0,B,S,1):>8s}{c('baseline',1,B,S,1):>8s} | "
          f"{c('liger',0,B,S,1):>8s}{c('liger',1,B,S,1):>8s} | {c('all',0,B,S,1):>8s}{c('all',1,B,S,1):>8s}")
