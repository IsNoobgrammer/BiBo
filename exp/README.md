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
