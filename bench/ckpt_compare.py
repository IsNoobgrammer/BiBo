"""Gradient-checkpointing comparison for the full BiBo model (all kernels + fused CE).
(1) grad-equivalence: ckpt-ON vs ckpt-OFF on the SAME weights/seed/input -> max|dgrad| (must be ~0
    now that router noise is commented out + dropout=0 -> deterministic recompute).
(2) time + peak-memory sweep across (batch, seq): checkpointing trades recompute time for activation
    memory. Subprocess-isolated per cell (fresh CUDA allocator). use_reentrant=True (matches train.py).
"""
import os, sys, subprocess, statistics
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

SHAPES = [(2,512),(2,1024),(2,2048),(4,1024),(8,1024)]   # 1024 .. 8192 tok, seq<=2048

def _build():
    import yaml, torch
    from bench.models import build_model_from_config
    from src.kernels.patch import patch_bibo_with_triton
    from src.kernels.moe_grouped import patch_moe_auto
    from src.kernels.dense_mlp import patch_dense_mlp_with_triton
    cfg = yaml.safe_load(open(os.path.join(_REPO, "bench/configs/bibo.yaml")))
    torch.manual_seed(0)
    m, c = build_model_from_config(cfg)
    patch_bibo_with_triton(m); patch_moe_auto(m); patch_dense_mlp_with_triton(m)
    m.config.use_fused_linear_ce = True; m.config.use_cache = False
    return m.cuda().train()

def _step_fn(m, x, lab):
    import torch
    def step():
        for p in m.parameters(): p.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(input_ids=x, labels=lab, use_cache=False)
            loss = out.loss
        loss.backward(); return loss
    return step

def gradcheck():
    import torch
    m = _build(); B, S = 2, 1024
    torch.manual_seed(1)
    x = torch.randint(0, 81000, (B, S), device="cuda")
    lab = torch.randint(0, 81000, (B, S), device="cuda")
    def grads():
        s = _step_fn(m, x, lab); loss = s()
        return loss.item(), {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.grad is not None}
    m.gradient_checkpointing_disable()
    l0, g0 = grads()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    l1, g1 = grads()
    maxd = 0.0; worst = ""
    for k in g0:
        d = (g0[k] - g1[k]).abs().max().item()
        if d > maxd: maxd, worst = d, k
    print(f"GRADCHECK B{B}xS{S}: loss off={l0:.5f} on={l1:.5f} (dloss {abs(l0-l1):.2e}) | "
          f"max|dgrad| {maxd:.2e} @ {worst} | {len(g0)} param tensors")

def time_cell(B, S, ckpt):
    import torch
    m = _build()
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    x = torch.randint(0, 81000, (B, S), device="cuda")
    lab = torch.randint(0, 81000, (B, S), device="cuda")
    step = _step_fn(m, x, lab)
    def t1(fn):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)
    try:
        for _ in range(4): step()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ms = statistics.median([t1(step) for _ in range(8)])
        print(f"RESULT {ms:.2f} {torch.cuda.max_memory_allocated()/1e6:.0f}")
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        print("RESULT OOM")

if len(sys.argv) > 1:
    if sys.argv[1] == "--gradcheck":
        gradcheck()
    elif sys.argv[1] == "--time":
        time_cell(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    sys.exit(0)

# ---- parent ----
print("=== gradient equivalence (ckpt ON vs OFF, same weights) ===", flush=True)
subprocess.run([sys.executable, os.path.abspath(__file__), "--gradcheck"])

print("\n=== time + peak-memory sweep ===", flush=True)
res = {}
for (B, S) in SHAPES:
    for ckpt in (0, 1):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--time", str(B), str(S), str(ckpt)],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT")), "RESULT OOM")
        tok = line.split()
        ms = float(tok[1]) if tok[1] != "OOM" else None
        mem = float(tok[2]) if ms is not None else None
        res[(B, S, ckpt)] = (ms, mem)
        print(f"  B{B}xS{S} ({B*S}tok) ckpt={ckpt}: " + (f"{ms:8.2f}ms {mem:7.0f}MB" if ms else "OOM"), flush=True)

print("\n" + "="*78)
print(f"{'BxS (tok)':>13s} | {'off ms':>8s} {'on ms':>8s} {'time x':>7s} | {'off MB':>7s} {'on MB':>7s} {'mem x':>6s}")
for (B, S) in SHAPES:
    o, on = res[(B,S,0)], res[(B,S,1)]
    oms, omem = o; nms, nmem = on
    tx = f"{nms/oms:.2f}" if (oms and nms) else "—"
    mx = f"{nmem/omem:.2f}" if (omem and nmem) else "—"
    f = lambda v: f"{v:.0f}" if v else "OOM"
    print(f"{B}x{S}({B*S:5d}) | {f(oms):>8s} {f(nms):>8s} {tx:>7s} | {f(omem):>7s} {f(nmem):>7s} {mx:>6s}")
print("time x >1 = checkpointing slower (recompute); mem x <1 = checkpointing saves memory")
