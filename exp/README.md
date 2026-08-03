# Experimental BiBo: Kimi K3 Attention Residuals

This package is the staging area for architectural experiments. It is deliberately
one-way: code in `exp/` may reuse stable components from `src/`, but `src/` never
imports, probes, or conditionally enables anything from `exp/`.

The current model implements Kimi K3's Block Attention Residuals (AttnRes). It was
ported from Moonshot AI's official sources rather than inferred from a third-party
implementation:

- [Kimi K3 Hugging Face model code (pinned revision)](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py)
- [Kimi K3 Hugging Face configuration (pinned revision)](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json)
- [Moonshot AI Attention Residuals repository](https://github.com/MoonshotAI/Attention-Residuals)
- [Attention Residuals paper](https://arxiv.org/abs/2603.15031)

Use the experimental classes explicitly:

```python
from exp.configuration_bibo import BiBoConfig
from exp.modeling_bibo import BiBoForCausalLM

config = BiBoConfig(attn_res_block_size=12)
model = BiBoForCausalLM(config)
```

`attn_res_block_size` counts complete decoder layers, matching Kimi K3. Pass
`None` to construct a standard-residual control using the same experimental
modeling entry point.

## Typed thought/memory residuals

BiBo's opt-in extension preserves the original K3 depth-content score and adds
a small, token-conditioned type score. A read can choose among:

1. completed canonical block states;
2. completed memory-only block states;
3. the current canonical prefix;
4. the current attention-produced thought stream; and
5. the current MLP-produced memory stream.

Attention and MLP still each emit one stream: attention writes only to thought,
and the MLP writes only to memory. Both are also added to the canonical prefix,
which prevents a learned gate from deleting the ordinary residual route. At a
block boundary the canonical state is archived, thought/memory are reset, and
the previous memory-only state is optionally archived for longer-lived recall.
The attention output drives the MLP site's type controller, so the current
attention computation can decide which kind of prior state the MLP accepts.

```python
from exp.configuration_bibo import BiBoConfig
from exp.modeling_bibo import BiBoForCausalLM

config = BiBoConfig(
    attn_res_block_size=3,
    attn_res_sites=2,
    use_typed_attn_res=True,
    typed_attn_res_long_memory=True,
    typed_attn_res_extra_init=0.01,
)
model = BiBoForCausalLM(config)
```

The type controllers are zero-initialized. Extra typed candidates begin at a
relative prior of `typed_attn_res_extra_init` against each canonical candidate;
this makes initialization a conservative extension of K3 instead of treating
empty/new typed streams as equally important on step zero.

The training harness exposes the same knobs:

```bash
python -m ablate.common.train --arm bibo_min --attn_res 3 --typed_attn_res
```

## Fast/slow depth memory

Fast/slow mode adds a sixth residual type without changing the canonical path:

```text
fast[l+1] = fast_decay[l] * fast[l] + memory_write[l]
slow[l+1] = slow_decay[l] * slow[l] + slow_gain(attention[l]) * memory_write[l]
```

Fast memory resets at each AttnRes block boundary. Slow memory persists across
the entire layer stack. The parameterization guarantees
`0 < fast_decay < slow_decay < 1`, even after training. The attention-conditioned
slow gain is `2 * sigmoid(.)`, so its zero controller initialization gives an
exact gain of one rather than suppressing the initial slow path.

```python
config = BiBoConfig(
    attn_res_block_size=3,
    attn_res_sites=2,
    use_typed_attn_res=True,
    use_typed_attn_res_fast_slow_memory=True,
    typed_attn_res_fast_decay_init=0.5,
    typed_attn_res_slow_decay_init=0.95,
)
```

`typed_attn_res_long_memory` remains independent: when enabled, completed fast
block states are also archived as individually addressable candidates. Disable
it to compare the compressed persistent slow state against block snapshots.

## Innovation-only typed memory writes

Innovation mode filters only the value written into typed memory:

```text
alpha = sigmoid(innovation_logit)
memory_write = mlp_output - alpha * projection(mlp_output onto thought)
```

The projection is per token and computed in fp32. The unfiltered MLP output is
always added to the canonical residual, so this gate cannot delete the standard
Transformer route. The default `alpha=0.01` starts close to the typed-memory
control while retaining a usable gradient.

```python
config = BiBoConfig(
    attn_res_block_size=3,
    attn_res_sites=2,
    use_typed_attn_res=True,
    use_typed_attn_res_innovation_write=True,
    typed_attn_res_innovation_init=0.01,
)
```

Run both extensions together with:

```bash
python -m ablate.common.train \
  --arm bibo_min \
  --attn_res 3 \
  --typed_attn_res \
  --typed_attn_res_fast_slow_memory \
  --typed_attn_res_innovation_write
```
