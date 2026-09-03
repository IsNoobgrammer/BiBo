"""Typed thought/memory streams layered onto K3 Block AttnRes."""

import math
import sys
import tempfile
import types

import pytest
import torch

from exp.configuration_bibo import BiBoConfig
from exp.modeling_bibo import (
    BiBoForCausalLM,
    apply_innovation_memory_write,
    apply_typed_attention_residual,
    update_fast_slow_memory,
)
from ablate.common.configs import make_bibo_min_config
from ablate.common.report_ckpt import _arch_kwargs
from src.modeling.norm import BiBoRMSNorm

from conftest import BASE, DEVICE, tokens


def make_config(**overrides):
    return BiBoConfig(
        **{
            **BASE,
            "attn_res_block_size": 2,
            "attn_res_sites": 2,
            "use_typed_attn_res": True,
            **overrides,
        }
    )


def make_model(device=DEVICE, **overrides):
    return BiBoForCausalLM(make_config(**overrides)).to(device)


def make_full_memory_model(device=DEVICE, **overrides):
    return make_model(
        device=device,
        use_typed_attn_res_fast_slow_memory=True,
        use_typed_attn_res_innovation_write=True,
        **overrides,
    )


def test_config_is_opt_in_validated_and_round_trips():
    assert BiBoConfig().use_typed_attn_res is False
    assert BiBoConfig().typed_attn_res_long_memory is True
    assert BiBoConfig().typed_attn_res_extra_init == 0.01
    assert BiBoConfig().use_typed_attn_res_fast_slow_memory is False
    assert BiBoConfig().use_typed_attn_res_innovation_write is False

    with pytest.raises(ValueError, match="attn_res_block_size"):
        make_config(attn_res_block_size=None)
    with pytest.raises(ValueError, match="attn_res_sites=2"):
        make_config(attn_res_sites=1)
    for value in (0.0, -0.1, 1.1, True, "0.1"):
        with pytest.raises(ValueError, match="typed_attn_res_extra_init"):
            make_config(typed_attn_res_extra_init=value)
    with pytest.raises(ValueError, match="slow_decay_init must be >"):
        make_config(
            use_typed_attn_res_fast_slow_memory=True,
            typed_attn_res_fast_decay_init=0.9,
            typed_attn_res_slow_decay_init=0.5,
        )
    with pytest.raises(ValueError, match="requires use_typed_attn_res"):
        BiBoConfig(use_typed_attn_res_fast_slow_memory=True)
    with pytest.raises(ValueError, match="requires use_typed_attn_res"):
        BiBoConfig(use_typed_attn_res_innovation_write=True)

    with tempfile.TemporaryDirectory() as directory:
        config = make_config(
            typed_attn_res_long_memory=False,
            typed_attn_res_extra_init=0.05,
            use_typed_attn_res_fast_slow_memory=True,
            typed_attn_res_fast_decay_init=0.4,
            typed_attn_res_slow_decay_init=0.9,
            use_typed_attn_res_innovation_write=True,
            typed_attn_res_innovation_init=0.2,
        )
        config.save_pretrained(directory)
        loaded = BiBoConfig.from_pretrained(directory)
    assert loaded.use_typed_attn_res is True
    assert loaded.typed_attn_res_long_memory is False
    assert loaded.typed_attn_res_extra_init == 0.05
    assert loaded.use_typed_attn_res_fast_slow_memory is True
    assert loaded.typed_attn_res_fast_decay_init == 0.4
    assert loaded.typed_attn_res_slow_decay_init == 0.9
    assert loaded.use_typed_attn_res_innovation_write is True
    assert loaded.typed_attn_res_innovation_init == 0.2


def test_ablation_and_eval_rebuild_plumbing_preserve_typed_architecture():
    config = make_bibo_min_config(
        attn_res="2",
        attn_res_sites=2,
        use_typed_attn_res=True,
        typed_attn_res_long_memory=False,
        typed_attn_res_extra_init=0.05,
        use_typed_attn_res_fast_slow_memory=True,
        typed_attn_res_fast_decay_init=0.4,
        typed_attn_res_slow_decay_init=0.9,
        use_typed_attn_res_innovation_write=True,
        typed_attn_res_innovation_init=0.2,
    )
    assert config.use_typed_attn_res is True
    assert config.typed_attn_res_long_memory is False
    assert config.typed_attn_res_extra_init == 0.05
    assert config.use_typed_attn_res_fast_slow_memory is True
    assert config.typed_attn_res_fast_decay_init == 0.4
    assert config.typed_attn_res_slow_decay_init == 0.9
    assert config.use_typed_attn_res_innovation_write is True
    assert config.typed_attn_res_innovation_init == 0.2

    rebuilt = _arch_kwargs(
        {
            "attn_res": "2",
            "moe_override": "0:8:8:576",
            "bf16_residual_stream": True,
            "typed_attn_res": True,
            "typed_attn_res_long_memory": False,
            "typed_attn_res_extra_init": 0.05,
            "typed_attn_res_fast_slow_memory": True,
            "typed_attn_res_fast_decay_init": 0.4,
            "typed_attn_res_slow_decay_init": 0.9,
            "typed_attn_res_innovation_write": True,
            "typed_attn_res_innovation_init": 0.2,
        }
    )
    assert rebuilt["moe_overrides"][0] == dict(
        num_routed_experts=8, num_experts_per_tok=8, moe_intermediate_size=576)
    assert rebuilt["bf16_residual_stream"] is True
    assert rebuilt["use_typed_attn_res"] is True
    assert rebuilt["typed_attn_res_long_memory"] is False
    assert rebuilt["typed_attn_res_extra_init"] == 0.05
    assert rebuilt["use_typed_attn_res_fast_slow_memory"] is True
    assert rebuilt["typed_attn_res_fast_decay_init"] == 0.4
    assert rebuilt["typed_attn_res_slow_decay_init"] == 0.9
    assert rebuilt["use_typed_attn_res_innovation_write"] is True
    assert rebuilt["typed_attn_res_innovation_init"] == 0.2

    with pytest.raises(ValueError, match="integer attn_res block size"):
        make_bibo_min_config(attn_res="off", use_typed_attn_res=True)


def test_typed_read_can_select_thought_or_memory_as_distinct_inputs():
    hidden = 4
    prefix = torch.zeros(1, hidden)
    thought = torch.full_like(prefix, 2.0)
    memory = torch.full_like(prefix, 7.0)
    empty = torch.zeros(1, 0, hidden)
    projection = torch.nn.Linear(hidden, 1, bias=False)
    torch.nn.init.zeros_(projection.weight)
    norm = BiBoRMSNorm(hidden)
    controller = torch.zeros(5, hidden)
    bias = torch.full((5,), -80.0)

    bias[3] = 80.0
    actual = apply_typed_attention_residual(
        prefix,
        empty,
        thought,
        memory,
        empty,
        projection,
        norm,
        controller,
        bias,
        prefix,
    )
    torch.testing.assert_close(actual, thought, rtol=0, atol=0)

    bias[3], bias[4] = -80.0, 80.0
    actual = apply_typed_attention_residual(
        prefix,
        empty,
        thought,
        memory,
        empty,
        projection,
        norm,
        controller,
        bias,
        prefix,
    )
    torch.testing.assert_close(actual, memory, rtol=0, atol=0)


def test_controller_source_changes_the_selected_residual_type_per_token():
    hidden = 2
    prefix = torch.zeros(2, hidden)
    thought = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    memory = torch.tensor([[9.0, 9.0], [9.0, 9.0]])
    source = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    empty = torch.zeros(2, 0, hidden)
    projection = torch.nn.Linear(hidden, 1, bias=False)
    torch.nn.init.zeros_(projection.weight)
    norm = BiBoRMSNorm(hidden)
    controller = torch.zeros(5, hidden)
    controller[3, 0] = 20.0
    controller[4, 0] = -20.0
    bias = torch.zeros(5)
    bias[2] = -80.0

    actual = apply_typed_attention_residual(
        prefix,
        empty,
        thought,
        memory,
        empty,
        projection,
        norm,
        controller,
        bias,
        source,
    )
    torch.testing.assert_close(actual[0], thought[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1], memory[1], rtol=0, atol=1e-6)


def test_sixth_typed_candidate_can_select_slow_memory():
    hidden = 4
    prefix = torch.zeros(1, hidden)
    thought = torch.full_like(prefix, 2.0)
    fast = torch.full_like(prefix, 5.0)
    slow = torch.full_like(prefix, 11.0)
    empty = torch.zeros(1, 0, hidden)
    projection = torch.nn.Linear(hidden, 1, bias=False)
    torch.nn.init.zeros_(projection.weight)
    norm = BiBoRMSNorm(hidden)
    controller = torch.zeros(6, hidden)
    bias = torch.full((6,), -80.0)
    bias[5] = 80.0

    actual = apply_typed_attention_residual(
        prefix,
        empty,
        thought,
        fast,
        empty,
        projection,
        norm,
        controller,
        bias,
        prefix,
        slow_memory_residual=slow,
    )
    torch.testing.assert_close(actual, slow, rtol=0, atol=0)


def test_innovation_write_removes_only_the_thought_parallel_component():
    thought = torch.tensor([[[2.0, 0.0]]])
    mlp = torch.tensor([[[3.0, 4.0]]])
    almost_full = torch.tensor(20.0)
    innovation = apply_innovation_memory_write(mlp, thought, almost_full, 1e-6)
    torch.testing.assert_close(
        innovation,
        torch.tensor([[[0.0, 4.0]]]),
        rtol=0,
        atol=1e-6,
    )

    almost_off = torch.tensor(-20.0)
    unchanged = apply_innovation_memory_write(mlp, thought, almost_off, 1e-6)
    torch.testing.assert_close(unchanged, mlp, rtol=0, atol=1e-6)


def test_fast_slow_update_has_ordered_decay_and_attention_write_gate():
    write = torch.tensor([[[2.0, 4.0]]])
    fast = torch.tensor([[[10.0, 10.0]]])
    slow = torch.tensor([[[20.0, 20.0]]])
    attention = torch.tensor([[[1.0, -1.0]]])
    fast_logit = torch.tensor(math.log(0.5 / 0.5))
    # gap=.8 gives slow=.5 + .5*.8 = .9
    gap_logit = torch.tensor(math.log(0.8 / 0.2))
    controller = torch.zeros(2)
    bias = torch.zeros(1)

    new_fast, new_slow = update_fast_slow_memory(
        fast,
        slow,
        write,
        attention,
        fast_logit,
        gap_logit,
        controller,
        bias,
        1e-6,
    )
    torch.testing.assert_close(new_fast, 0.5 * fast + write)
    torch.testing.assert_close(new_slow, 0.9 * slow + write)

    reset_fast, _ = update_fast_slow_memory(
        None,
        slow,
        write,
        attention,
        fast_logit,
        gap_logit,
        controller,
        bias,
        1e-6,
    )
    torch.testing.assert_close(reset_fast, write)


def test_model_fast_resets_at_boundaries_while_slow_persists_exactly():
    model = make_full_memory_model().eval()
    captured = {index: {} for index in range(model.config.num_hidden_layers)}
    hooks = []
    for index, layer in enumerate(model.model.layers):
        hooks.extend(
            [
                layer.self_attn.register_forward_hook(
                    lambda _module, _inputs, output, i=index: captured[i].update(
                        attention=output[0]
                    )
                ),
                layer.mlp.register_forward_hook(
                    lambda _module, _inputs, output, i=index: captured[i].update(
                        mlp=output
                    )
                ),
                layer.register_forward_hook(
                    lambda _module, _inputs, output, i=index: captured[i].update(
                        layer=output
                    )
                ),
            ]
        )
    try:
        model(tokens(1, 5, seed=146))
    finally:
        for hook in hooks:
            hook.remove()

    previous_fast = None
    previous_slow = torch.zeros_like(captured[0]["layer"][5])
    for index, layer in enumerate(model.model.layers):
        output = captured[index]["layer"]
        thought, actual_fast, actual_slow = output[2], output[3], output[5]
        write = apply_innovation_memory_write(
            captured[index]["mlp"],
            thought,
            layer.typed_attn_res_innovation_logit,
            layer.post_attention_layernorm.variance_epsilon,
        )
        if index % model.config.attn_res_block_size == 0:
            previous_fast = None
        expected_fast, expected_slow = update_fast_slow_memory(
            previous_fast,
            previous_slow,
            write,
            captured[index]["attention"],
            layer.typed_attn_res_fast_decay_logit,
            layer.typed_attn_res_slow_decay_gap_logit,
            layer.typed_attn_res_slow_write_controller,
            layer.typed_attn_res_slow_write_bias,
            layer.post_attention_layernorm.variance_epsilon,
        )
        torch.testing.assert_close(actual_fast, expected_fast, rtol=0, atol=0)
        torch.testing.assert_close(actual_slow, expected_slow, rtol=0, atol=0)
        previous_fast, previous_slow = actual_fast, actual_slow


def test_attention_and_mlp_outputs_have_single_typed_destinations():
    model = make_model().eval()
    layer = model.model.layers[0]
    captured = {}

    def capture_attention(_module, _inputs, output):
        captured["attention"] = output[0]

    def capture_mlp(_module, _inputs, output):
        captured["mlp"] = output

    def capture_layer(_module, _inputs, output):
        captured["layer"] = output

    hooks = [
        layer.self_attn.register_forward_hook(capture_attention),
        layer.mlp.register_forward_hook(capture_mlp),
        layer.register_forward_hook(capture_layer),
    ]
    try:
        model(tokens(1, 5, seed=40))
    finally:
        for hook in hooks:
            hook.remove()

    prefix, _, thought, memory, _ = captured["layer"][:5]
    torch.testing.assert_close(thought, captured["attention"], rtol=0, atol=0)
    torch.testing.assert_close(memory, captured["mlp"], rtol=0, atol=0)
    torch.testing.assert_close(prefix, thought + memory, rtol=0, atol=0)


def test_memory_only_state_is_committed_at_the_next_block_boundary():
    model = make_model().eval()
    records = []
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: records.append(
                (output[3].detach(), output[4].detach())
            )
        )
        for layer in model.model.layers
    ]
    try:
        model(tokens(1, 5, seed=41))
    finally:
        for hook in hooks:
            hook.remove()

    assert [block_memory.shape[1] for _, block_memory in records] == [0, 0, 1, 1]
    previous_memory = records[1][0].reshape(-1, BASE["hidden_size"])
    torch.testing.assert_close(records[2][1][:, 0], previous_memory, rtol=0, atol=0)


@pytest.mark.parametrize("fast_slow", [False, True])
def test_long_memory_can_be_disabled_without_disabling_current_typed_streams(fast_slow):
    model = make_model(
        typed_attn_res_long_memory=False,
        use_typed_attn_res_fast_slow_memory=fast_slow,
    ).eval()
    counts = []
    slow_norms = []

    def capture(_module, _inputs, output):
        counts.append(output[4].shape[1])
        if fast_slow:
            slow_norms.append(output[5].norm().item())

    hooks = [
        layer.register_forward_hook(capture)
        for layer in model.model.layers
    ]
    try:
        model(tokens(1, 5, seed=42))
    finally:
        for hook in hooks:
            hook.remove()
    assert counts == [0, 0, 0, 0]
    if fast_slow:
        assert all(value > 0 for value in slow_norms)


@pytest.mark.parametrize(
    ("fast_slow", "innovation"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_fast_slow_and_innovation_axes_compose_independently(fast_slow, innovation):
    model = make_model(
        use_typed_attn_res_fast_slow_memory=fast_slow,
        use_typed_attn_res_innovation_write=innovation,
    ).train()
    x = tokens(1, 6, seed=147)
    loss = model(x, labels=x).loss
    loss.backward()
    assert torch.isfinite(loss)
    names = set(dict(model.named_parameters()))
    assert any("typed_attn_res_fast_decay_logit" in name for name in names) == fast_slow
    assert any("typed_attn_res_innovation_logit" in name for name in names) == innovation


def test_parameter_overhead_is_only_type_controllers_and_biases():
    torch.manual_seed(43)
    control = BiBoForCausalLM(
        BiBoConfig(**BASE, attn_res_block_size=2, attn_res_sites=2)
    )
    torch.manual_seed(43)
    typed = make_model(device="cpu")

    difference = sum(parameter.numel() for parameter in typed.parameters()) - sum(
        parameter.numel() for parameter in control.parameters()
    )
    sites = 2 * typed.config.num_hidden_layers + 1
    assert difference == sites * (5 * typed.config.hidden_size + 5)

    expected_bias = torch.tensor([0.0, math.log(0.01), 0.0, math.log(0.01), math.log(0.01)])
    for layer in typed.model.layers:
        assert torch.count_nonzero(layer.self_attention_res_type_controller) == 0
        torch.testing.assert_close(layer.self_attention_res_type_bias, expected_bias)


def test_fast_slow_and_innovation_parameter_overhead_is_exact():
    typed = make_model(device="cpu")
    full = make_full_memory_model(device="cpu")
    hidden = typed.config.hidden_size
    layers = typed.config.num_hidden_layers
    read_sites = 2 * layers + 1
    # One extra slow-memory row+bias at every typed read. Per layer: fast logit,
    # slow-gap logit, slow-write vector+bias, and innovation-strength logit.
    expected = read_sites * (hidden + 1) + layers * (hidden + 4)
    actual = sum(p.numel() for p in full.parameters()) - sum(
        p.numel() for p in typed.parameters()
    )
    assert actual == expected


@pytest.mark.parametrize("full_memory", [False, True])
def test_typed_read_parameters_use_zero_decay_adamw_not_muon(monkeypatch, full_memory):
    class FakeMuon(torch.optim.Optimizer):
        def __init__(self, params, **kwargs):
            super().__init__(params, {"lr": kwargs["lr"]})

        def step(self, closure=None):
            return None

    kernels = types.ModuleType("kernels")
    sm120 = types.ModuleType("kernels.sm120")
    muon = types.ModuleType("kernels.sm120.muon")
    muon.FusedMuon = FakeMuon
    monkeypatch.setitem(sys.modules, "kernels", kernels)
    monkeypatch.setitem(sys.modules, "kernels.sm120", sm120)
    monkeypatch.setitem(sys.modules, "kernels.sm120.muon", muon)

    from ablate.common.optim import build_optimizers

    model = (
        make_full_memory_model(device="cpu")
        if full_memory
        else make_model(device="cpu")
    )
    optimizers, _, _ = build_optimizers(model)
    muon_ids = {
        id(parameter)
        for group in optimizers[0].param_groups
        for parameter in group["params"]
    }
    adam_groups = {
        id(parameter): group
        for group in optimizers[1].param_groups
        for parameter in group["params"]
    }
    typed = {
        name: parameter
        for name, parameter in model.named_parameters()
        if (
            "_res_type_controller" in name
            or "_res_type_bias" in name
            or "typed_attn_res_" in name
        )
    }
    assert typed
    for name, parameter in typed.items():
        assert id(parameter) not in muon_ids, f"{name} was incorrectly assigned to Muon"
        assert adam_groups[id(parameter)]["weight_decay"] == 0.0


def test_forward_backward_reaches_all_typed_read_parameters():
    model = make_model().train()
    x = tokens(2, 8, seed=44)
    output = model(x, labels=x, output_attentions=True, output_hidden_states=True)
    output.loss.backward()

    assert output.logits.shape == (2, 8, BASE["vocab_size"])
    assert len(output.attentions) == model.config.num_hidden_layers
    assert len(output.hidden_states) == model.config.num_hidden_layers + 1
    parameters = [
        # Layer zero's attention read has only the embedding candidate, so its
        # controller correctly receives zero gradient on that first site.
        model.model.layers[1].self_attention_res_type_controller,
        model.model.layers[1].mlp_res_type_controller,
        model.model.layers[2].mlp_res_type_bias,
        model.model.output_attn_res_type_controller,
        model.model.output_attn_res_type_bias,
    ]
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_full_memory_forward_backward_reaches_new_controllers():
    model = make_full_memory_model().train()
    x = tokens(2, 8, seed=145)
    output = model(x, labels=x, output_attentions=True)
    output.loss.backward()

    layer = model.model.layers[1]
    parameters = [
        layer.self_attention_res_type_controller,
        layer.typed_attn_res_fast_decay_logit,
        layer.typed_attn_res_slow_decay_gap_logit,
        layer.typed_attn_res_slow_write_controller,
        layer.typed_attn_res_slow_write_bias,
        layer.typed_attn_res_innovation_logit,
        model.model.output_attn_res_type_controller,
    ]
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("full_memory", [False, True])
def test_incremental_decode_and_generate_use_typed_state_path(full_memory):
    torch.manual_seed(45)
    model = (make_full_memory_model() if full_memory else make_model()).eval()
    x = tokens(1, 8, seed=46)
    full = model(x).logits
    past, steps = None, []
    for index in range(x.shape[1]):
        output = model(x[:, index : index + 1], past_key_values=past, use_cache=True)
        past = output.past_key_values
        steps.append(output.logits)
    incremental = torch.cat(steps, dim=1)
    assert (full - incremental).abs().max().item() < 2e-4

    generated = model.generate(
        x[:, :4],
        attention_mask=torch.ones_like(x[:, :4]),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=0,
    )
    assert generated.shape == (1, 6)


@pytest.mark.parametrize("full_memory", [False, True])
def test_full_gradient_checkpointing_preserves_typed_gradients(full_memory):
    torch.manual_seed(47)
    factory = make_full_memory_model if full_memory else make_model
    reference = factory().train()
    checkpointed = factory().train()
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.gradient_checkpointing_enable()
    x = tokens(2, 8, seed=48)

    reference(x, labels=x, use_cache=False).loss.backward()
    checkpointed(x, labels=x, use_cache=False).loss.backward()
    expected = dict(reference.named_parameters())
    actual = dict(checkpointed.named_parameters())
    assert expected.keys() == actual.keys()
    for name in expected:
        if expected[name].grad is not None:
            assert actual[name].grad is not None
            torch.testing.assert_close(
                expected[name].grad,
                actual[name].grad,
                rtol=0,
                atol=0,
                msg=lambda message: f"{name}: {message}",
            )


@pytest.mark.parametrize("full_memory", [False, True])
def test_save_load_round_trip_is_exact_for_typed_model(full_memory):
    model = (
        make_full_memory_model(device="cpu")
        if full_memory
        else make_model(device="cpu")
    ).eval()
    with torch.no_grad():
        bias = model.model.layers[1].mlp_res_type_bias
        bias.add_(torch.linspace(0.1, 0.1 * bias.numel(), bias.numel()))
    x = torch.randint(0, BASE["vocab_size"], (1, 6))
    expected = model(x).logits
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        loaded = BiBoForCausalLM.from_pretrained(directory).eval()
    torch.testing.assert_close(expected, loaded(x).logits, rtol=0, atol=0)
    torch.testing.assert_close(
        model.model.layers[1].mlp_res_type_bias,
        loaded.model.layers[1].mlp_res_type_bias,
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_typed_reads_keep_fp32_math_under_autocast_and_diagnostics_are_detached():
    torch.manual_seed(7)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    blocks = torch.randn(32, 3, 64, device="cuda", dtype=torch.bfloat16)
    projection = torch.nn.Linear(64, 1, bias=False).cuda()
    norm = BiBoRMSNorm(64).cuda()
    controller = torch.randn(5, 64, device="cuda", requires_grad=True)
    bias = torch.randn(5, device="cuda", requires_grad=True)
    args = (x, blocks, x * .7, x * .3, blocks, projection, norm, controller, bias, x)
    expected, probs, _ = apply_typed_attention_residual(*args, return_details=True)
    diagnostics = {}
    with torch.autocast("cuda", torch.bfloat16):
        actual, amp_probs, _ = apply_typed_attention_residual(
            *args, return_details=True, diagnostics=diagnostics)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(amp_probs, probs, rtol=0, atol=0)
    torch.testing.assert_close(sum(diagnostics[f"read/type_mass_{i}"] for i in range(5)),
                               torch.ones((), device="cuda"))
    assert all(not value.requires_grad for value in diagnostics.values())
    actual.float().square().mean().backward()
    assert controller.grad is not None and torch.isfinite(controller.grad).all()


@pytest.mark.parametrize("options", [
    {"attn_res_carry": True}, {"attn_res_carry_scale": "sigmoid"},
    {"attn_res_carry_per_dim": True, "attn_res_carry": True, "attn_res_carry_scale": "sigmoid"},
    {"attn_res_score": "signorm"},
    {"attn_res_topk": 2}, {"attn_res_emb_term": True},
])
def test_typed_model_rejects_inert_ordinary_residual_options(options):
    with pytest.raises(ValueError, match="typed AttnRes requires"):
        make_config(**options)
