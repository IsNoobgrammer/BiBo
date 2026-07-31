"""Training-path smoke: gradients reach everything, autocast is NaN-free, loss actually moves.

Deliberately capped at 5 optimizer steps — this is a wiring check, not a benchmark.
"""
import math

import pytest
import torch
from conftest import DEVICE, make_model, tokens

HYBRID = [0, 1, 1, 0]


@pytest.mark.parametrize("norm", ["sum", "softmax", False])
@pytest.mark.parametrize("hybrid", [None, HYBRID], ids=["global", "hybrid"])
def test_every_trainable_param_receives_a_gradient(norm, hybrid):
    m = make_model(norm_topk_prob=norm, hybrid_layer_pattern=hybrid,
                   use_ssmax=(hybrid is None), use_xsa=True, use_shared_expert=True)
    x = tokens(2, 8)
    m(x, labels=x).loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient for {missing[:5]}"
    # radial_theta is a real parameter now (it used to be injected by ablate as `situ_alpha`), so it
    # has to be in that sweep — an inert exponent would leave the model stuck at normsilu's floor.
    assert any("radial_theta" in n for n, _ in m.named_parameters())


@pytest.mark.parametrize("shared_type", ["mlp", "conv"])
def test_shared_expert_variants_train(shared_type):
    m = make_model(use_shared_expert=True, shared_expert_type=shared_type)
    x = tokens(2, 8)
    out = m(x, labels=x)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    conv = [n for n, _ in m.named_parameters() if "gate_conv" in n]
    assert bool(conv) == (shared_type == "conv"), "conv shared expert wiring"


@pytest.mark.gpu
@pytest.mark.skipif(DEVICE != "cuda", reason="autocast(bf16) needs CUDA")
def test_bf16_autocast_forward_backward_is_nan_free():
    """Training precision is bf16 (fp16 was removed 2026-07-08 — Muon + fp16 overflowed experts)."""
    m = make_model(use_ssmax=True, use_xsa=True, hybrid_layer_pattern=HYBRID)
    x = tokens(2, 16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x, labels=x).loss
    loss.backward()
    assert torch.isfinite(loss), f"bf16 loss is {loss}"
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite bf16 grads: {bad[:5]}"


def test_five_step_overfit_decreases_loss_and_runs_the_balancer():
    torch.manual_seed(0)
    m = make_model(bias_update_threshold=32).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = tokens(2, 16, seed=11)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = m(x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses), losses
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    moved = any(l.mlp.gate.bias.abs().sum().item() > 0
                for l in m.model.layers if hasattr(l.mlp, "gate"))
    assert moved, "the load balancer never updated the router bias"


def test_eval_mode_skips_the_balancer():
    m = make_model(bias_update_threshold=1).eval()
    m(tokens(2, 8))
    assert all(l.mlp.gate.bias.abs().max() == 0
               for l in m.model.layers if hasattr(l.mlp, "gate")), \
        "bias must not update outside training (it would also deadlock the DDP all_reduce)"


def test_single_token_forward_produces_finite_logits():
    """The decode path: one token in, usable logits out (no labels — see the next test for why)."""
    m = make_model().eval()
    logits = m(tokens(1, 1)).logits
    assert logits.shape == (1, 1, m.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_shortest_trainable_sequence_is_two_tokens():
    """seq_len=1 with labels is NaN BY CONSTRUCTION, not a bug: next-token loss shifts labels, so a
    1-token sequence yields shift_logits of shape (B, 0, V) and cross_entropy over an empty tensor.
    The Qwen3MoE baseline returns nan here too. seq_len=2 is the first trainable length."""
    m = make_model()
    one = tokens(1, 1)
    assert torch.isnan(m(one, labels=one).loss), "expected the empty-label-set NaN"
    two = tokens(1, 2)
    assert torch.isfinite(m(two, labels=two).loss)
