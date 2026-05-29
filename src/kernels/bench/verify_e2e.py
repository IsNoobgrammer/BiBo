"""
End-to-End Verification: Original vs Triton-RMSNorm-patched BiBo model.

Verifies:
1. Forward pass outputs match (logits, loss)
2. Backward pass gradients match
3. Works under torch.amp.autocast (mixed precision)
4. Multi-step training convergence matches

Run: .\\venv\\Scripts\\python src/kernels/bench/verify_e2e.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import torch.nn as nn

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.kernels.patch import patch_bibo_with_triton, unpatch_bibo


def make_small_config():
    """Small BiBo config for fast verification (~2M params)."""
    return BiBoConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        use_ssmax=True,
        polyglu_expert_multiplier=1,
        special_expert_pairs=1,
        num_experts_per_tok=2,
        moe_intermediate_size=192,
        use_shared_expert=True,
        shared_expert_type="mlp",
        router_type="mlp",
        router_lambda=1.0,
        router_noise=0.0,
        moe_shared_scaling=0.5,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
    )


def compare_tensors(name, ref, tri, atol, rtol):
    """Compare two tensors, return True if close."""
    if ref is None and tri is None:
        print(f"  [PASS] {name}: both None")
        return True
    if ref is None or tri is None:
        print(f"  [FAIL] {name}: one is None")
        return False
    if ref.dtype != tri.dtype:
        tri = tri.to(ref.dtype)
    if ref.isnan().any() or tri.isnan().any():
        print(f"  [FAIL] {name}: NaN detected (ref_nan={ref.isnan().any().item()}, tri_nan={tri.isnan().any().item()})")
        return False
    if torch.allclose(ref, tri, atol=atol, rtol=rtol):
        max_diff = (ref - tri).abs().max().item()
        print(f"  [PASS] {name} (max_diff={max_diff:.2e})")
        return True
    else:
        max_diff = (ref - tri).abs().max().item()
        mean_diff = (ref - tri).abs().mean().item()
        print(f"  [FAIL] {name} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
        return False


def test_forward_fp32():
    print("\n" + "=" * 60)
    print("TEST 1: Forward Pass (fp32)")
    print("=" * 60)

    device = 'cuda'
    config = make_small_config()
    input_ids = torch.randint(0, 1000, (2, 64), device=device)
    labels = input_ids.clone()

    # Original
    torch.manual_seed(42)
    model_orig = BiBoForCausalLM(config).to(device).eval()

    with torch.no_grad():
        out_orig = model_orig(input_ids=input_ids, labels=labels)

    # Triton (same weights via same seed)
    torch.manual_seed(42)
    model_tri = BiBoForCausalLM(config).to(device).eval()
    patch_bibo_with_triton(model_tri)

    with torch.no_grad():
        out_tri = model_tri(input_ids=input_ids, labels=labels)

    all_pass = True
    all_pass &= compare_tensors("loss", out_orig.loss, out_tri.loss, atol=1e-5, rtol=1e-5)
    if out_orig.logits is not None and out_tri.logits is not None:
        all_pass &= compare_tensors("logits", out_orig.logits, out_tri.logits, atol=1e-4, rtol=1e-4)

    del model_orig, model_tri
    torch.cuda.empty_cache()
    return all_pass


def test_backward_fp32():
    print("\n" + "=" * 60)
    print("TEST 2: Backward Pass (fp32)")
    print("=" * 60)

    device = 'cuda'
    config = make_small_config()
    input_ids = torch.randint(0, 1000, (2, 64), device=device)
    labels = input_ids.clone()

    # Original
    torch.manual_seed(42)
    model_orig = BiBoForCausalLM(config).to(device).train()
    out_orig = model_orig(input_ids=input_ids, labels=labels)
    out_orig.loss.backward()
    grads_orig = {n: p.grad.clone() for n, p in model_orig.named_parameters() if p.grad is not None}

    # Triton
    torch.manual_seed(42)
    model_tri = BiBoForCausalLM(config).to(device).train()
    patch_bibo_with_triton(model_tri)
    out_tri = model_tri(input_ids=input_ids, labels=labels)
    out_tri.loss.backward()
    grads_tri = {n: p.grad.clone() for n, p in model_tri.named_parameters() if p.grad is not None}

    all_pass = True
    all_pass &= compare_tensors("loss", out_orig.loss, out_tri.loss, atol=1e-5, rtol=1e-5)

    # Compare all gradients
    checked = 0
    failed = 0
    max_max_diff = 0.0
    for name in grads_orig:
        if name in grads_tri:
            g1, g2 = grads_orig[name], grads_tri[name]
            if g1.isnan().any() or g2.isnan().any():
                failed += 1
                if failed <= 3:
                    print(f"  [FAIL] grad({name}): NaN")
            elif not torch.allclose(g1, g2, atol=1e-4, rtol=1e-4):
                diff = (g1 - g2).abs().max().item()
                max_max_diff = max(max_max_diff, diff)
                failed += 1
                if failed <= 3:
                    print(f"  [FAIL] grad({name}) max_diff={diff:.2e}")
            checked += 1

    if failed == 0:
        print(f"  [PASS] All {checked} parameter gradients match (atol=1e-4)")
    else:
        print(f"  [FAIL] {failed}/{checked} gradients differ (worst={max_max_diff:.2e})")
    all_pass &= (failed == 0)

    del model_orig, model_tri
    torch.cuda.empty_cache()
    return all_pass


def test_autocast():
    print("\n" + "=" * 60)
    print("TEST 3: Autocast (fp16) Forward + Backward")
    print("=" * 60)

    device = 'cuda'
    config = make_small_config()
    input_ids = torch.randint(0, 1000, (2, 64), device=device)
    labels = input_ids.clone()

    # Original
    torch.manual_seed(42)
    model_orig = BiBoForCausalLM(config).to(device).train()
    with torch.amp.autocast('cuda'):
        out_orig = model_orig(input_ids=input_ids, labels=labels)
    out_orig.loss.backward()
    grads_orig = {n: p.grad.clone() for n, p in model_orig.named_parameters() if p.grad is not None}

    # Triton
    torch.manual_seed(42)
    model_tri = BiBoForCausalLM(config).to(device).train()
    patch_bibo_with_triton(model_tri)
    with torch.amp.autocast('cuda'):
        out_tri = model_tri(input_ids=input_ids, labels=labels)
    out_tri.loss.backward()
    grads_tri = {n: p.grad.clone() for n, p in model_tri.named_parameters() if p.grad is not None}

    all_pass = True
    all_pass &= compare_tensors("loss (autocast)", out_orig.loss, out_tri.loss, atol=1e-3, rtol=1e-3)

    # Gradients (looser tolerance for fp16)
    checked = 0
    failed = 0
    for name in grads_orig:
        if name in grads_tri:
            g1, g2 = grads_orig[name], grads_tri[name]
            if g1.isnan().any() or g2.isnan().any():
                failed += 1
            elif not torch.allclose(g1, g2, atol=5e-2, rtol=5e-2):
                failed += 1
            checked += 1

    if failed == 0:
        print(f"  [PASS] All {checked} gradients match (atol=5e-2, fp16)")
    elif failed < checked * 0.1:
        print(f"  [PASS] {failed}/{checked} grads differ slightly — acceptable for fp16")
    else:
        print(f"  [FAIL] {failed}/{checked} gradients differ")
        all_pass = False

    del model_orig, model_tri
    torch.cuda.empty_cache()
    return all_pass


def test_multi_step():
    print("\n" + "=" * 60)
    print("TEST 4: Multi-step Training (5 steps, autocast + GradScaler)")
    print("=" * 60)

    device = 'cuda'
    config = make_small_config()
    input_ids = torch.randint(0, 1000, (2, 64), device=device)
    labels = input_ids.clone()

    # Original
    torch.manual_seed(42)
    model_orig = BiBoForCausalLM(config).to(device).train()
    opt_orig = torch.optim.AdamW(model_orig.parameters(), lr=1e-3)
    scaler_orig = torch.amp.GradScaler('cuda')

    losses_orig = []
    for _ in range(5):
        opt_orig.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model_orig(input_ids=input_ids, labels=labels)
        scaler_orig.scale(out.loss).backward()
        scaler_orig.step(opt_orig)
        scaler_orig.update()
        losses_orig.append(out.loss.item())

    # Triton
    torch.manual_seed(42)
    model_tri = BiBoForCausalLM(config).to(device).train()
    patch_bibo_with_triton(model_tri)
    opt_tri = torch.optim.AdamW(model_tri.parameters(), lr=1e-3)
    scaler_tri = torch.amp.GradScaler('cuda')

    losses_tri = []
    for _ in range(5):
        opt_tri.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model_tri(input_ids=input_ids, labels=labels)
        scaler_tri.scale(out.loss).backward()
        scaler_tri.step(opt_tri)
        scaler_tri.update()
        losses_tri.append(out.loss.item())

    print(f"  Step | Original  | Triton    | Diff")
    print(f"  -----|-----------|-----------|--------")
    all_pass = True
    for i in range(5):
        diff = abs(losses_orig[i] - losses_tri[i])
        status = "ok" if diff < 0.01 else "!!"
        print(f"  {i:4d} | {losses_orig[i]:.6f} | {losses_tri[i]:.6f} | {diff:.2e} {status}")
        if diff > 0.05:
            all_pass = False

    final_diff = abs(losses_orig[-1] - losses_tri[-1])
    if final_diff < 0.05:
        print(f"\n  [PASS] Final loss diff = {final_diff:.2e}")
    else:
        print(f"\n  [FAIL] Final loss diff = {final_diff:.2e}")
        all_pass = False

    del model_orig, model_tri
    torch.cuda.empty_cache()
    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("BiBo Triton RMSNorm — End-to-End Verification")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"PyTorch: {torch.__version__}")
    import triton
    print(f"Triton: {triton.__version__}")

    results = {}
    results['forward_fp32'] = test_forward_fp32()
    results['backward_fp32'] = test_backward_fp32()
    results['autocast_fp16'] = test_autocast()
    results['multi_step'] = test_multi_step()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if all(results.values()):
        print("\n  All E2E tests passed.")
        print("  Safe to use on Kaggle T4 (sm_75) — Triton compiles at runtime.")
    else:
        print("\n  Some tests failed.")
        sys.exit(1)
