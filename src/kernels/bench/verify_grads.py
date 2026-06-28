"""
Gradient Equivalence Verification — P0 Action Item

Compares weight gradients between Triton-patched and unpatched BiBo models.
This is the test that would have caught the register_buffer + autograd opacity bugs.

FAIL criteria:
  1. Any parameter has grad=None in patched but not in unpatched
  2. Any parameter has grad.norm()==0 in patched but >0 in unpatched
  3. Max gradient difference > 1e-3 (fp32) or > 5e-2 (fp16)

Run:
    .\\venv\\Scripts\\python src/kernels/bench/verify_grads.py
"""

import os
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

from src.configuration_bibo import BiBoConfig
from src.modeling.models import BiBoForCausalLM
from src.kernels.patch import patch_bibo_with_triton, unpatch_bibo
from src.kernels.moe_dispatch import patch_moe_with_triton, unpatch_moe


def make_test_model(device='cuda'):
    """Create a small BiBo model for gradient testing."""
    cfg = BiBoConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        moe_intermediate_size=128,
        polyglu_expert_multiplier=2,
        special_expert_pairs=1,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        moe_shared_scaling=2.0,
        use_ssmax=True,
    )
    model = BiBoForCausalLM(cfg).to(device).train()
    return cfg, model


def test_gradient_equivalence_fp32():
    """Test gradient equivalence in fp32 (strict tolerance)."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Equivalence (fp32, atol=1e-3)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Build reference model
    _, model_ref = make_test_model(device)

    # Build patched model (same weights)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model_tri = make_test_model(device)
    model_tri.load_state_dict(model_ref.state_dict())

    # Apply all patches
    patch_bibo_with_triton(model_tri)
    patch_moe_with_triton(model_tri)

    # Same input
    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (2, 64)).to(device)

    # Forward + backward (reference)
    out_ref = model_ref(input_ids=input_ids, labels=input_ids)
    out_ref.loss.backward()

    # Forward + backward (triton)
    out_tri = model_tri(input_ids=input_ids, labels=input_ids)
    out_tri.loss.backward()

    # Compare
    loss_diff = abs(out_ref.loss.item() - out_tri.loss.item())
    print(f"  Loss: ref={out_ref.loss.item():.6f}, tri={out_tri.loss.item():.6f}, diff={loss_diff:.2e}")

    failures = []
    max_grad_diff = 0.0
    zero_grad_params = []
    missing_grad_params = []
    checked = 0

    for (name_ref, p_ref), (name_tri, p_tri) in zip(
        model_ref.named_parameters(), model_tri.named_parameters()
    ):
        assert name_ref == name_tri, f"Name mismatch: {name_ref} vs {name_tri}"

        if p_ref.grad is None and p_tri.grad is None:
            continue

        checked += 1

        # FAIL 1: grad exists in ref but not in patched
        if p_ref.grad is not None and p_tri.grad is None:
            missing_grad_params.append(name_tri)
            failures.append(f"  FAIL: {name_tri} has grad=None (ref has grad)")
            continue

        # FAIL 2: grad is zero in patched but non-zero in ref
        if p_ref.grad is not None and p_tri.grad is not None:
            ref_norm = p_ref.grad.norm().item()
            tri_norm = p_tri.grad.norm().item()
            if tri_norm == 0 and ref_norm > 0:
                zero_grad_params.append(name_tri)
                failures.append(f"  FAIL: {name_tri} has zero grad (ref norm={ref_norm:.4e})")
                continue

            # FAIL 3: gradient difference too large
            diff = (p_ref.grad - p_tri.grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
            if diff > 1e-3:
                failures.append(f"  FAIL: {name_tri} grad diff={diff:.2e} > 1e-3")

    print(f"  Parameters checked: {checked}")
    print(f"  Max gradient difference: {max_grad_diff:.2e}")
    print(f"  Missing grads: {len(missing_grad_params)}")
    print(f"  Zero grads: {len(zero_grad_params)}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures[:10]:
            print(f"    {f}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
        print(f"\n  RESULT: FAIL")
        return False
    else:
        print(f"\n  RESULT: PASS (all gradients match within 1e-3)")
        return True


def test_gradient_equivalence_fp16():
    """Test gradient equivalence under autocast fp16 (relaxed tolerance)."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Equivalence (fp16 autocast, atol=5e-2)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  SKIP: fp16 test requires CUDA")
        return True

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model_ref = make_test_model(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model_tri = make_test_model(device)
    model_tri.load_state_dict(model_ref.state_dict())

    patch_bibo_with_triton(model_tri)
    patch_moe_with_triton(model_tri)

    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (2, 64)).to(device)

    # Reference with autocast
    with torch.autocast('cuda', dtype=torch.float16):
        out_ref = model_ref(input_ids=input_ids, labels=input_ids)
    out_ref.loss.backward()

    # Triton with autocast
    with torch.autocast('cuda', dtype=torch.float16):
        out_tri = model_tri(input_ids=input_ids, labels=input_ids)
    out_tri.loss.backward()

    loss_diff = abs(out_ref.loss.item() - out_tri.loss.item())
    print(f"  Loss: ref={out_ref.loss.item():.6f}, tri={out_tri.loss.item():.6f}, diff={loss_diff:.2e}")

    failures = []
    max_grad_diff = 0.0
    checked = 0

    for (name_ref, p_ref), (name_tri, p_tri) in zip(
        model_ref.named_parameters(), model_tri.named_parameters()
    ):
        if p_ref.grad is None and p_tri.grad is None:
            continue
        checked += 1

        if p_ref.grad is not None and p_tri.grad is None:
            failures.append(f"  FAIL: {name_tri} has grad=None")
            continue

        if p_ref.grad is not None and p_tri.grad is not None:
            ref_norm = p_ref.grad.norm().item()
            tri_norm = p_tri.grad.norm().item()
            if tri_norm == 0 and ref_norm > 0:
                failures.append(f"  FAIL: {name_tri} has zero grad")
                continue

            diff = (p_ref.grad - p_tri.grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
            if diff > 5e-2:
                failures.append(f"  FAIL: {name_tri} grad diff={diff:.2e} > 5e-2")

    print(f"  Parameters checked: {checked}")
    print(f"  Max gradient difference: {max_grad_diff:.2e}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures[:10]:
            print(f"    {f}")
        print(f"\n  RESULT: FAIL")
        return False
    else:
        print(f"\n  RESULT: PASS (all gradients match within 5e-2 for fp16)")
        return True


def test_no_frozen_params():
    """Verify no parameter has zero gradient that shouldn't (frozen weight detection)."""
    print("\n" + "=" * 60)
    print("TEST: No Frozen Parameters (all params receive gradients)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model = make_test_model(device)

    patch_bibo_with_triton(model)
    patch_moe_with_triton(model)

    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (4, 64)).to(device)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()

    frozen = []
    nan_params = []
    total_params = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total_params += 1

        if p.grad is None:
            frozen.append(f"{name} (grad=None)")
        elif p.grad.norm().item() == 0:
            frozen.append(f"{name} (grad=0)")
        elif p.grad.isnan().any():
            nan_params.append(name)

    print(f"  Total trainable parameters: {total_params}")
    print(f"  Frozen (zero/None grad): {len(frozen)}")
    print(f"  NaN gradients: {len(nan_params)}")

    if frozen:
        print(f"\n  FROZEN PARAMS:")
        for f in frozen:
            print(f"    {f}")
        print(f"\n  RESULT: FAIL")
        return False

    if nan_params:
        print(f"\n  NaN PARAMS:")
        for n in nan_params:
            print(f"    {n}")
        print(f"\n  RESULT: FAIL")
        return False

    print(f"\n  RESULT: PASS (all {total_params} params have non-zero, finite gradients)")
    return True


def test_no_stale_buffers():
    """Verify patching doesn't register buffers where parameters should be."""
    print("\n" + "=" * 60)
    print("TEST: No Stale Buffers (no _fused_gate_up_weight)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model = make_test_model(device)

    patch_bibo_with_triton(model)
    patch_moe_with_triton(model)

    bad_buffers = []
    for name, _ in model.named_buffers():
        if '_fused_gate_up_weight' in name:
            bad_buffers.append(name)

    if bad_buffers:
        print(f"  FAIL: Found stale buffers: {bad_buffers}")
        return False

    print(f"  RESULT: PASS (no _fused_gate_up_weight buffers)")
    return True


def test_multi_step_convergence():
    """Verify patched model converges at same rate as unpatched over 30 steps."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Step Convergence (30 steps, loss within 5%)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reference model
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model_ref = make_test_model(device)
    opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3)

    # Patched model (same init)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    _, model_tri = make_test_model(device)
    model_tri.load_state_dict(model_ref.state_dict())
    patch_bibo_with_triton(model_tri)
    patch_moe_with_triton(model_tri)
    opt_tri = torch.optim.AdamW(model_tri.parameters(), lr=1e-3)

    # Fixed data for overfitting
    torch.manual_seed(999)
    input_ids = torch.randint(0, 1000, (4, 32)).to(device)

    losses_ref = []
    losses_tri = []

    for step in range(30):
        # Reference step
        opt_ref.zero_grad()
        out_ref = model_ref(input_ids=input_ids, labels=input_ids)
        out_ref.loss.backward()
        opt_ref.step()
        losses_ref.append(out_ref.loss.item())

        # Triton step
        opt_tri.zero_grad()
        out_tri = model_tri(input_ids=input_ids, labels=input_ids)
        out_tri.loss.backward()
        opt_tri.step()
        losses_tri.append(out_tri.loss.item())

    print(f"  Step  0: ref={losses_ref[0]:.4f}, tri={losses_tri[0]:.4f}, diff={abs(losses_ref[0]-losses_tri[0]):.2e}")
    print(f"  Step 14: ref={losses_ref[14]:.4f}, tri={losses_tri[14]:.4f}, diff={abs(losses_ref[14]-losses_tri[14]):.2e}")
    print(f"  Step 29: ref={losses_ref[29]:.4f}, tri={losses_tri[29]:.4f}, diff={abs(losses_ref[29]-losses_tri[29]):.2e}")

    # Check final loss within 5%
    final_ref = losses_ref[-1]
    final_tri = losses_tri[-1]
    pct_diff = abs(final_ref - final_tri) / max(final_ref, 1e-8) * 100

    print(f"  Final loss difference: {pct_diff:.2f}%")

    if pct_diff > 5.0:
        print(f"  RESULT: FAIL (>{5}% divergence)")
        return False

    # Check both models actually learned
    if losses_tri[-1] >= losses_tri[0] * 0.9:
        print(f"  RESULT: FAIL (Triton model didn't learn — loss didn't decrease)")
        return False

    print(f"  RESULT: PASS (loss within {pct_diff:.2f}%, both models converge)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT EQUIVALENCE VERIFICATION")
    print("Post-incident P0 action item — catches register_buffer")
    print("and autograd opacity bugs before they reach training.")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU (limited testing)")

    results = {}
    results['fp32_grads'] = test_gradient_equivalence_fp32()
    results['fp16_grads'] = test_gradient_equivalence_fp16()
    results['no_frozen'] = test_no_frozen_params()
    results['no_buffers'] = test_no_stale_buffers()
    results['convergence'] = test_multi_step_convergence()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: {status}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)
