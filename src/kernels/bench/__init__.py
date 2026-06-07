"""
BiBo Kernel Benchmarks — Correctness + Performance validation.

All benchmarks follow the 4 mandatory rules (see AGENTS.md Triton Kernels section):
1. Gradient equivalence vs baseline (original PyTorch, not Triton-patched)
2. NaN-free multi-pass stability (>=2 fwd+bwd passes)
3. Three-phase timing: forward-only, backward-only, forward+backward (>=3 steps each)
4. torch.profiler for all benchmarking (never time.time())

Run individual benchmarks:
    .\\venv\\Scripts\\python -m src.kernels.bench.bench_dense_mlp    # Dense MLP (3-variant head-to-head)
    .\\venv\\Scripts\\python -m src.kernels.bench.bench_moe          # MoE layer
    .\\venv\\Scripts\\python -m src.kernels.bench.bench_conv         # Conv fusion
    .\\venv\\Scripts\\python -m src.kernels.bench.bench_moe_fwdbwd   # MoE full fwd+bwd
    .\\venv\\Scripts\\python -m src.kernels.bench.verify_e2e         # E2E correctness
    .\\venv\\Scripts\\python -m src.kernels.bench.verify_grads       # Gradient verification
    .\\venv\\Scripts\\python -m src.kernels.bench.profile_benchmark  # torch.profiler 4-way
"""
