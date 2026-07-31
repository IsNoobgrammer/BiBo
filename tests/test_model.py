"""Model level: KV cache, tying, layer placement, generate, checkpoint round-trip.

The `from_pretrained` tests here are REGRESSION GUARDS for two severe bugs fixed 2026-07-26 in
`BiBoPreTrainedModel._init_weights`. Both were silent (no missing keys, no warning):
  1. Raw `param.data.normal_()` writes ignored the `_is_hf_initialized` flag that transformers >=5
     uses to protect loaded weights -> 25 of 54 tensors were re-randomized on every load.
  2. No `RotaryEmbedding` branch, so the `persistent=False` inv_freq buffers came back as
     uninitialized memory. With `original_inv_freq` zeroed, dynamic-NTK returns zeros in-window,
     cos=1/sin=0, and RoPE degenerates to the identity -> the model loses positional encoding.
If either of these regresses, the tests below fail loudly instead of silently ruining checkpoints.
"""
import os
import tempfile

import pytest
import torch
from conftest import DEVICE, make_config, make_model, tokens
from safetensors.torch import load_file

from src.modeling.ffn.mlp import BiBoMLP
from src.modeling.models import BiBoForCausalLM


# ── KV cache ─────────────────────────────────────────────────────────────────
def test_incremental_decode_matches_full_forward():
    torch.manual_seed(0)
    m = make_model().eval()
    x = tokens(1, 8, seed=3)
    full = m(x).logits
    past, steps = None, []
    for t in range(x.shape[1]):
        out = m(x[:, t:t + 1], past_key_values=past, use_cache=True)
        past, _ = out.past_key_values, steps.append(out.logits)
    assert (full - torch.cat(steps, 1)).abs().max().item() < 2e-4


def test_swa_layers_use_a_window_evicted_cache():
    """Windowed layers must hold O(sliding_window) KV during decode, not O(total_len)."""
    W, S = 4, 16
    m = make_model(hybrid_layer_pattern=[0, 1, 1, 0], sliding_window=W, use_ssmax=False).eval()
    cache = m(tokens(1, S, seed=4), use_cache=True).past_key_values
    swa_len = cache.layers[1].keys.shape[-2]
    global_len = cache.layers[0].keys.shape[-2]
    assert swa_len <= W + 1, f"SWA layer kept {swa_len} keys for W={W}"
    assert global_len == S, f"global layer should keep all {S}, kept {global_len}"


# ── structure ────────────────────────────────────────────────────────────────
def test_embedding_and_lm_head_tying():
    tied = make_model(tie_word_embeddings=True)
    assert tied.lm_head.weight.data_ptr() == tied.model.embed_tokens.weight.data_ptr()
    untied = make_model(tie_word_embeddings=False)
    assert untied.lm_head.weight.data_ptr() != untied.model.embed_tokens.weight.data_ptr()


def test_mlp_only_layers_places_dense_mlp_at_the_ends():
    m = make_model(num_hidden_layers=5, mlp_only_layers=[0, 4])
    kinds = ["dense" if isinstance(l.mlp, BiBoMLP) else "moe" for l in m.model.layers]
    assert kinds == ["dense", "moe", "moe", "moe", "dense"]


def test_exp_post_embed_norm_costs_nothing_when_off():
    off, on = make_model(exp_post_embed_norm=False), make_model(exp_post_embed_norm=True)
    assert off.model.embed_norm is None and on.model.embed_norm is not None
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on - n_off == off.config.hidden_size


def test_output_attentions_and_hidden_states():
    m = make_model()
    out = m(tokens(1, 6, seed=5), output_attentions=True, output_hidden_states=True)
    assert len(out.attentions) == m.config.num_hidden_layers
    assert out.attentions[0].shape[-2:] == (6, 6)
    assert len(out.hidden_states) == m.config.num_hidden_layers + 1


def test_generate_with_kv_cache():
    m = make_model().eval()
    out = m.generate(tokens(2, 4, seed=6), max_new_tokens=6, do_sample=False, pad_token_id=0)
    assert out.shape == (2, 10)


def test_selective_gradient_checkpointing_preserves_gradients():
    torch.manual_seed(0)
    a = make_model().train()
    x = tokens(2, 8, seed=7)
    a(x, labels=x).loss.backward()
    grads = {n: p.grad.clone() for n, p in a.named_parameters() if p.grad is not None}

    b = make_model().train()
    b.load_state_dict(a.state_dict())
    enable = (getattr(b, "enable_selective_gradient_checkpointing", None)
              or getattr(b.model, "enable_selective_gradient_checkpointing", None))
    assert enable is not None, "no selective-checkpointing entry point"
    enable()
    b.zero_grad()
    b(x, labels=x).loss.backward()
    after = {n: p.grad for n, p in b.named_parameters() if p.grad is not None}
    assert set(grads) == set(after)
    assert max((grads[n] - after[n]).abs().max().item() for n in grads) < 1e-5


# ── checkpoint round-trip: the regression guards ─────────────────────────────
def _save_and_reload(**overrides):
    m = make_model(**overrides).eval()
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        disk = load_file(os.path.join(d, "model.safetensors"))
        reloaded = BiBoForCausalLM.from_pretrained(d).to(DEVICE).eval()
    return m, reloaded, disk


def test_save_pretrained_writes_the_in_memory_weights():
    """Isolates save from load: the file itself must match memory."""
    m, _, disk = _save_and_reload()
    sd = m.state_dict()
    for k, v in disk.items():
        assert (v.to(DEVICE).float() - sd[k].float()).abs().max().item() == 0, f"{k} saved wrong"


@pytest.mark.parametrize("norm", ["sum", "softmax"])
def test_from_pretrained_does_not_reinitialize_loaded_weights(norm):
    """REGRESSION: `.data.normal_()` in _init_weights re-randomized 25/54 loaded tensors."""
    _, reloaded, disk = _save_and_reload(norm_topk_prob=norm)
    sd = reloaded.state_dict()
    clobbered = [k for k in disk
                 if (disk[k].to(DEVICE).float() - sd[k].float()).abs().max().item() > 0]
    assert not clobbered, (f"{len(clobbered)}/{len(disk)} checkpoint tensors were overwritten by "
                           f"_init_weights, e.g. {clobbered[:3]}")


def test_from_pretrained_rebuilds_non_persistent_buffers():
    """REGRESSION: inv_freq is persistent=False, so a missing RotaryEmbedding branch in
    _init_weights left it as uninitialized memory and silently disabled RoPE."""
    m, reloaded, _ = _save_and_reload()
    before, after = dict(m.named_buffers()), dict(reloaded.named_buffers())
    assert set(before) == set(after)
    for k in before:
        d = (before[k].float() - after[k].float()).abs().max().item()
        assert d == 0, f"buffer {k} differs after load by {d:.3e}"
    r = reloaded.model.rotary_emb
    assert r.original_inv_freq.abs().max() > 0, "zeroed inv_freq => cos=1/sin=0 => RoPE is a no-op"
    assert r.original_inv_freq.flatten()[0].item() == 1.0


def test_save_load_round_trip_is_logit_exact():
    m, reloaded, _ = _save_and_reload(norm_topk_prob="softmax")
    x = tokens(1, 6, seed=8)
    assert (m(x).logits - reloaded(x).logits).abs().max().item() == 0


def test_fresh_initialization_is_seed_reproducible():
    """The flag-aware init rewrite must not change FRESH init: with nothing flagged, the helpers
    fall through to the identical torch calls."""
    torch.manual_seed(1234)
    a = make_model()
    torch.manual_seed(1234)
    b = make_model()
    assert max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters())) == 0
    std = a.model.layers[1].self_attn.q_proj.weight.std().item()
    assert abs(std - a.config.initializer_range) / a.config.initializer_range < 0.25
    norms = [l.input_layernorm.weight for l in a.model.layers]
    assert all(torch.allclose(w, torch.ones_like(w)) for w in norms), "RMSNorm gain must init to 1.0"


def test_plain_load_state_dict_also_round_trips():
    """run_eval.py uses this path rather than from_pretrained."""
    m = make_model().eval()
    x = tokens(1, 6, seed=9)
    before = m(x).logits
    other = make_model().eval()
    other.load_state_dict(m.state_dict())
    assert (before - other(x).logits).abs().max().item() == 0
