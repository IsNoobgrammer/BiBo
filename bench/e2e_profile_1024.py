"""torch.profiler op breakdown + timing of a REAL training step (config batch x seq, fwd+bwd).

Profiles the ACTUAL training path — model built from the YAML, all Triton kernels + fused CE
applied (or eager via --baseline), at the config's real batch_size x seq_len — so the numbers
represent the real use case, not a B1 microbench. Reports: model size, per-step median time,
tokens/sec, MFU, peak memory, and the per-op CUDA-time table (where the time actually goes).

Run on T4 (the training GPU) — batch 16 x hidden 512 will OOM a 4GB local card.

Usage:
  python bench/e2e_profile_1024.py                          # bibo, real shape, all kernels+CE
  python bench/e2e_profile_1024.py --config bench/configs/qwen3moe.yaml
  python bench/e2e_profile_1024.py --baseline               # eager (no kernels, std CE) A/B
  python bench/e2e_profile_1024.py --batch 8 --compile      # override batch; with torch.compile
Compile note: with --compile the op table lumps into inductor kernels (less attributable) but
matches real wall-time; without it (default) ops attribute cleanly (eager fuses fewer elementwise,
so memory-bound ops read slightly higher than a compiled run).
"""
import os, sys, time, argparse
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "bench"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="bench/configs/bibo.yaml")
    ap.add_argument("--batch", type=int, default=None, help="override config batch_size")
    ap.add_argument("--seq", type=int, default=None, help="override config seq_len")
    ap.add_argument("--baseline", action="store_true", help="eager (no Triton kernels, standard CE)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (real wall-time)")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--timed", type=int, default=10)
    ap.add_argument("--rows", type=int, default=20)
    args = ap.parse_args()

    import yaml, torch, torch.profiler as P
    from bench.models import build_model_from_config, count_params, apply_triton_kernels
    from metrics import estimate_mfu, format_count

    cfg = yaml.safe_load(open(os.path.join(_REPO, args.config)))
    B = args.batch or cfg["training"]["batch_size"]
    S = args.seq or cfg["training"].get("seq_len", 1024)
    torch.manual_seed(0)
    m, c = build_model_from_config(cfg)
    p = count_params(m, c)

    if not args.baseline:
        apply_triton_kernels(m, c, no_triton=False, use_fused_ce=True)   # the real training path
    m = m.cuda().train()
    if args.compile:
        m = torch.compile(m)

    V = c.vocab_size
    x = torch.randint(0, V, (B, S), device="cuda")
    lab = torch.randint(0, V, (B, S), device="cuda")

    def step():
        for prm in m.parameters():
            prm.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(input_ids=x, labels=lab, use_cache=False)
            loss = out.loss
        loss.backward()
        return loss

    tag = f"{'baseline-eager' if args.baseline else 'all+fusedCE'}{'+compile' if args.compile else ''}"
    print(f"\n===== {os.path.basename(args.config)} | {tag} | B{B} x S{S} = {B*S} tok/step | fwd+bwd =====")
    print(f"params: {p['total_m']:.1f}M total / {p['active_m']:.1f}M active | "
          f"hidden={c.hidden_size} heads={c.num_attention_heads} head_dim={c.hidden_size//c.num_attention_heads}")

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # timing: median over `timed` steps
    times = []
    for _ in range(args.timed):
        torch.cuda.synchronize(); t0 = time.time()
        step()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    times.sort()
    med = times[len(times) // 2]
    tps = B * S / med
    mfu = estimate_mfu(p["active"], tps, "cuda")
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nstep: median {med*1e3:.1f} ms | tps {format_count(tps)}/s | MFU {mfu*100:.1f}% | peak {peak:.1f} GB")

    # one profiled step
    with P.profile(activities=[P.ProfilerActivity.CPU, P.ProfilerActivity.CUDA]) as pr:
        step(); torch.cuda.synchronize()
    print(pr.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.rows))


if __name__ == "__main__":
    main()
