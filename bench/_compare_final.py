"""BiBo vs Qwen3MoE: 300 steps, no compile, identical settings."""
import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, time, torch, gc, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda"
STEPS = 300
BS = 16
ACCUM = 1
SEQ = 1024

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
print("Steps: %d, Batch: %d, Accum: %d, Seq: %d" % (STEPS, BS, ACCUM, SEQ))
print("=" * 60)

# Create data
def make_data():
    torch.manual_seed(42)
    x = torch.randint(0, 81000, (BS, SEQ)).to(DEVICE)
    labels = torch.randint(0, 81000, (BS, SEQ)).to(DEVICE)
    return x, labels

x, labels = make_data()

def build_and_train(name, build_fn, steps=STEPS):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    gc.collect()
    torch.cuda.empty_cache()

    model = build_fn()
    model = model.to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print("%s: %.1fM params" % (name, params / 1e6))

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler()

    # Warmup 3
    for i in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, labels=labels)
            loss_val = out.loss if hasattr(out, "loss") else (out[0] if isinstance(out, tuple) else out)
            if isinstance(loss_val, tuple): loss_val = loss_val[0]
            loss = loss_val / ACCUM
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Timed training
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total_tok = 0
    losses = []
    times = []

    for i in range(steps):
        step_t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, labels=labels)
            loss_val = out.loss if hasattr(out, "loss") else (out[0] if isinstance(out, tuple) else out)
            if isinstance(loss_val, tuple): loss_val = loss_val[0]
            loss = loss_val / ACCUM
            losses.append(loss_val.item())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        total_tok += x.numel()
        torch.cuda.synchronize()
        times.append(time.time() - step_t0)

        if (i + 1) % 50 == 0:
            avg_tps = total_tok / (time.time() - t0)
            print("  step %d/%d | loss=%.4f | tps=%.0f" % (i + 1, steps, losses[-1], avg_tps))

    elapsed = time.time() - t0
    tps = total_tok / elapsed
    vram = torch.cuda.max_memory_allocated() / 1e9
    avg_step = sum(times) / len(times)

    print("  DONE: %.0f tps | %.2fGB | loss %.4f->%.4f | avg_step=%.3fs" % (
        tps, vram, losses[0], losses[-1], avg_step))

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    return {"tps": tps, "vram": vram, "losses": losses, "times": times,
            "loss_s": losses[0], "loss_e": losses[-1], "elapsed": elapsed}


# Build BiBo
def build_bibo():
    import yaml
    with open("bench/configs/bibo.yaml") as f:
        cfg = yaml.safe_load(f)
    from bench.models import build_model_from_config, apply_triton_kernels
    model, config = build_model_from_config(cfg)
    apply_triton_kernels(model, config, no_triton=False)
    return model

# Build Qwen
def build_qwen():
    import yaml
    with open("bench/configs/qwen3moe.yaml") as f:
        cfg = yaml.safe_load(f)
    from bench.models import build_model_from_config, apply_triton_kernels
    model, config = build_model_from_config(cfg)
    apply_triton_kernels(model, config, no_triton=False)
    return model

print("\n--- BiBo Training ---")
bibo_r = build_and_train("BiBo", build_bibo)

print("\n--- Qwen3MoE Training ---")
qwen_r = build_and_train("Qwen3MoE", build_qwen)

# Save results
results = {
    "bibo": {"tps": bibo_r["tps"], "vram": bibo_r["vram"], "loss_s": bibo_r["loss_s"], "loss_e": bibo_r["loss_e"], "losses": bibo_r["losses"]},
    "qwen": {"tps": qwen_r["tps"], "vram": qwen_r["vram"], "loss_s": qwen_r["loss_s"], "loss_e": qwen_r["loss_e"], "losses": qwen_r["losses"]},
}

with open("bench/_comparison_results.json", "w") as f:
    json.dump(results, f)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print("BiBo:   %.0f tps | %.2fGB | loss %.4f -> %.4f (reduction: %.4f)" % (
    bibo_r["tps"], bibo_r["vram"], bibo_r["loss_s"], bibo_r["loss_e"], bibo_r["loss_s"] - bibo_r["loss_e"]))
print("Qwen:   %.0f tps | %.2fGB | loss %.4f -> %.4f (reduction: %.4f)" % (
    qwen_r["tps"], qwen_r["vram"], qwen_r["loss_s"], qwen_r["loss_e"], qwen_r["loss_s"] - qwen_r["loss_e"]))
print()
print("TPS diff:   BiBo is %.1f%% %s than Qwen" % (
    abs(bibo_r["tps"] - qwen_r["tps"]) / max(qwen_r["tps"], 1) * 100,
    "faster" if bibo_r["tps"] > qwen_r["tps"] else "slower"))
print("Loss diff:  BiBo reduction %.4f vs Qwen %.4f" % (
    bibo_r["loss_s"] - bibo_r["loss_e"], qwen_r["loss_s"] - qwen_r["loss_e"]))
print("VRAM:       BiBo %.2fGB vs Qwen %.2fGB" % (bibo_r["vram"], qwen_r["vram"]))
print("=" * 60)
print("Results saved to bench/_comparison_results.json")
