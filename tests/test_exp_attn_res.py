"""Kimi K3 Block Attention Residuals in the isolated ``exp`` package."""

import ast
import pathlib
import subprocess
import sys
import tempfile

import pytest
import torch

from exp.configuration_bibo import BiBoConfig as ExperimentalBiBoConfig
from exp.modeling_bibo import (
    BiBoForCausalLM as ExperimentalBiBoForCausalLM,
    apply_attention_residual,
)
from src.configuration_bibo import BiBoConfig as StableBiBoConfig
from src.modeling.models import BiBoForCausalLM as StableBiBoForCausalLM
from src.modeling.norm import BiBoRMSNorm

from conftest import BASE, DEVICE, tokens


def make_exp_config(**overrides):
    return ExperimentalBiBoConfig(**{**BASE, "attn_res_block_size": 2, **overrides})


def make_exp_model(device=DEVICE, **overrides):
    return ExperimentalBiBoForCausalLM(make_exp_config(**overrides)).to(device)


def test_config_defaults_to_k3_block_size_and_round_trips():
    assert ExperimentalBiBoConfig().attn_res_block_size == 12
    with tempfile.TemporaryDirectory() as directory:
        config = make_exp_config(attn_res_block_size=3)
        config.save_pretrained(directory)
        loaded = ExperimentalBiBoConfig.from_pretrained(directory)
    assert loaded.attn_res_block_size == 3
    assert loaded.model_type == "bibo_attn_res"


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "12"])
def test_config_rejects_invalid_block_sizes(value):
    with pytest.raises(ValueError, match="attn_res_block_size"):
        make_exp_config(attn_res_block_size=value)


def test_attention_residual_matches_direct_reference():
    torch.manual_seed(0)
    tokens_count, blocks, hidden = 6, 3, 16
    prefix = torch.randn(tokens_count, hidden)
    completed = torch.randn(tokens_count, blocks, hidden)
    norm = BiBoRMSNorm(hidden)
    projection = torch.nn.Linear(hidden, 1, bias=False)

    values = torch.cat((completed, prefix.unsqueeze(1)), dim=1)
    keys = norm(values)
    scores = projection(keys).squeeze(-1)
    expected = (scores.softmax(-1).unsqueeze(-1) * values).sum(dim=1)
    actual = apply_attention_residual(prefix, completed, projection, norm)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_zero_query_is_an_exact_uniform_depth_average():
    prefix = torch.tensor([[7.0, 8.0]])
    completed = torch.tensor([[[1.0, 2.0], [4.0, 5.0]]])
    norm = BiBoRMSNorm(2)
    projection = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(projection.weight)

    actual = apply_attention_residual(prefix, completed, projection, norm)
    expected = torch.tensor([[4.0, 5.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_block_boundaries_match_k3_layer_index_semantics():
    model = make_exp_model(attn_res_block_size=2).eval()
    block_counts = []
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: block_counts.append(output[1].shape[1])
        )
        for layer in model.model.layers
    ]
    try:
        model(tokens(1, 5, seed=10))
    finally:
        for hook in hooks:
            hook.remove()

    # Boundaries are layer 0, 2, ...; layer zero stores the token embedding.
    assert block_counts == [1, 1, 2, 2]


def test_parameter_overhead_is_two_queries_and_norms_per_layer_plus_output():
    stable = StableBiBoForCausalLM(StableBiBoConfig(**BASE))
    experimental = make_exp_model(device="cpu")
    difference = sum(p.numel() for p in experimental.parameters()) - sum(
        p.numel() for p in stable.parameters()
    )
    expected = (4 * experimental.config.num_hidden_layers + 2) * experimental.config.hidden_size
    assert difference == expected


def test_forward_backward_reaches_depth_mixing_parameters():
    model = make_exp_model().train()
    x = tokens(2, 8, seed=11)
    output = model(x, labels=x, output_attentions=True, output_hidden_states=True)
    output.loss.backward()

    assert output.logits.shape == (2, 8, model.config.vocab_size)
    assert len(output.attentions) == model.config.num_hidden_layers
    assert len(output.hidden_states) == model.config.num_hidden_layers + 1
    assert model.model.output_attn_res_proj.weight.grad is not None
    assert model.model.layers[0].mlp_res_proj.weight.grad is not None
    assert model.model.layers[1].self_attention_res_proj.weight.grad is not None
    assert torch.isfinite(model.model.output_attn_res_proj.weight.grad).all()


def test_full_gradient_checkpointing_preserves_gradients():
    torch.manual_seed(13)
    reference = make_exp_model().train()
    checkpointed = make_exp_model().train()
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.gradient_checkpointing_enable()
    x = tokens(2, 8, seed=14)

    reference(x, labels=x, use_cache=False).loss.backward()
    checkpointed(x, labels=x, use_cache=False).loss.backward()
    reference_grads = {
        name: parameter.grad
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    checkpointed_grads = {
        name: parameter.grad
        for name, parameter in checkpointed.named_parameters()
        if parameter.grad is not None
    }

    assert reference_grads.keys() == checkpointed_grads.keys()
    for name, gradient in reference_grads.items():
        torch.testing.assert_close(
            gradient,
            checkpointed_grads[name],
            rtol=0,
            atol=0,
            msg=lambda message: f"{name}: {message}",
        )


def test_incremental_decode_matches_full_forward():
    torch.manual_seed(0)
    model = make_exp_model().eval()
    x = tokens(1, 8, seed=12)
    full = model(x).logits
    past, steps = None, []
    for index in range(x.shape[1]):
        output = model(x[:, index:index + 1], past_key_values=past, use_cache=True)
        past = output.past_key_values
        steps.append(output.logits)
    incremental = torch.cat(steps, dim=1)
    assert (full - incremental).abs().max().item() < 2e-4


def test_generate_uses_the_experimental_kv_cache_path():
    model = make_exp_model().eval()
    prompt = tokens(2, 4, seed=15)
    generated = model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=0,
    )
    assert generated.shape == (2, 7)


def test_attn_res_disabled_is_bit_exact_with_stable_model():
    torch.manual_seed(123)
    stable = StableBiBoForCausalLM(StableBiBoConfig(**BASE)).eval()
    torch.manual_seed(123)
    control = ExperimentalBiBoForCausalLM(
        ExperimentalBiBoConfig(**BASE, attn_res_block_size=None)
    ).eval()

    assert stable.state_dict().keys() == control.state_dict().keys()
    for name, value in stable.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)
    x = torch.randint(0, BASE["vocab_size"], (1, 6))
    torch.testing.assert_close(stable(x).logits, control(x).logits, rtol=0, atol=0)


def test_save_load_round_trip_is_logit_exact():
    model = make_exp_model(device="cpu").eval()
    x = torch.randint(0, BASE["vocab_size"], (1, 6))
    expected = model(x).logits
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        loaded = ExperimentalBiBoForCausalLM.from_pretrained(directory).eval()
    torch.testing.assert_close(expected, loaded(x).logits, rtol=0, atol=0)


def test_src_does_not_import_or_probe_exp():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "exp" or alias.name.startswith("exp.")
                    for alias in node.names
                ):
                    offenders.append(path)
            elif isinstance(node, ast.ImportFrom):
                if node.module == "exp" or (node.module and node.module.startswith("exp.")):
                    offenders.append(path)
    assert not offenders

    code = (
        "import sys; import src; "
        "assert not any(name == 'exp' or name.startswith('exp.') for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
