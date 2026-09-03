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

## Training diagnostics and the current baseline

The `some_mhc` branch includes the training stack from baseline revision
`51a2469`: packed Parquet input, held-out validation, global NoPE/local full
RoPE, per-layer expert overrides, bf16 residual streams, and per-layer router
and parameter/gradient diagnostics.

Typed reads require `--attn_res_sites 2`. They replace the baseline's one-site
carry path; leave `--attn_res_carry false`, `--attn_res_carry_per_dim false`, and
`--attn_res_carry_scale none`. Incompatible carry, embedding-skip, sparse-read,
and non-softmax options raise instead of silently doing nothing. Typed scores,
controllers, and aggregation remain fp32 under bf16 autocast.

With W&B enabled, `train/` contains loss, gradient norm, learning rate, throughput,
and memory; `val/` contains the fixed held-out loss. `interp/typed/` records:

- Per-layer attention/MLP reads and the final output read: probability mass for
  each residual type, candidate entropy, output RMS, and canonical, thought,
  fast-memory, and slow-memory RMS.
- Per-layer learned fast/slow decay, innovation strength, controller RMS, and
  type biases; `interp/typed_*` also contains aggregate min/mean/max values.
- Actual slow-write gain and the RMS of raw, filtered, and removed memory writes.

Type IDs are `0=full block`, `1=archived block memory`, `2=current prefix`,
`3=thought`, `4=current/fast memory`, and `5=slow memory`. Type mass sums to one;
absent candidates have zero mass. Forward diagnostics sample the last training
microbatch of each logged optimizer step and are disabled before validation.
Captured values are detached scalars, so logging retains no activation graphs.
`grad/norm/*` and `params/norm/*` retain the training harness's per-tensor
health metrics. `report_ckpt` rebuilds the typed flags and layer-0 expert override
from the saved result configuration for strict checkpoint loading.

For a 100-step smoke, use `--max_steps 100 --log_every 10 --val_every 20
--val_start 20 --final_report false` with a distinct `--run_tag`. This compresses
the warmup and cosine schedule into 100 steps, exercising the peak learning rate;
it is a functional test rather than a prefix of a 2,000-step learning curve.
Keep the full run's data, seed, sequence length, and effective batch fixed. Extra
states and two read sites increase activation memory, so reduce microbatch size
and increase gradient accumulation together if needed.

Verified on 2026-09-03 with PyTorch 2.11.0+cu130 on an RTX PRO 6000 Blackwell:
155 pytest checks passed, followed by 100 real-data training steps at batch 32,
grad accumulation 8, sequence length 1024, and seed 2026. All 11 history rows
were finite; validation ran at steps 20/40/60/80/99. Final training loss was
6.009633, final 20-step running loss 6.000733, and held-out loss 6.1989984512.
Strict checkpoint reload reproduced that held-out loss exactly. Peak allocated
memory was 56.8 GB and steady throughput about 67.7k tokens/s. These smoke
numbers validate execution and logging; they do not establish an improvement
over the 2,000-step baseline.


## Fused typed reads

The CUDA path uses the sibling TKF repository's
`kernels.sm120.typed_attn_res` when available. Validated kernel revision:
[`b74cb90`](https://github.com/adi-kmt/triton-kernel-fused/commit/b74cb905a494395e00c23f68dd9a128833ff75bc)
on the `typed-attn-res` branch. It reads each residual stream
in place, keeping the content/type scores and mixing in fp32 and providing an
analytic backward. Probability diagnostics remain available at every read site.
CPU and installations without that kernel retain the eager implementation.

Set `BIBO_TYPED_AR_IMPL=fused` to require the kernel during GPU training, or
`BIBO_TYPED_AR_IMPL=eager` for a matched reference run; the default is `auto`.
W&B records this selection as `typed_ar_impl`. No model parameters or checkpoint
keys change. Tiny fp32 reduction-order differences are expected; fp32 parity
means numerical agreement, not bit-identical training trajectories through
routing decisions and reduced-precision matrix operations.

The TKF gate `python -m parity_check.parity_typed_attn_res` checks the read and
all gradients against the same fp32 reference, including bf16 quantization
error, strided/mixed inputs, empty blocks, zero states, and aliasing. BiBo's
`test_fused_typed_read_matches_eager_through_full_model` additionally checks
that all read sites dispatch to the kernel and compares full-model logits,
loss, and parameter gradients. The full CUDA suite passed 157 tests, including a dtype guard at every typed read.


The fused 100-step smoke at batch 32 × accumulation 8 completed on 2026-09-03:
training loss 6.0044, final running loss about 5.9950, validation 6.199862,
steady throughput about 109k tokens/s, peak allocation 46.6 GB. The preceding
eager typed smoke used the same data, seed, schedule, and effective batch and
measured 67.7k tokens/s and 56.8 GB: about 1.6x training throughput.

Checkpoint keys load strictly. Validation is numerically close, not bit-exact:
reload audits gave 6.199658–6.200111. Replaying the existing MoE atomic scatter
on identical operands showed fp32 differences up to 1.9e-6, occasionally crossing
a bf16 rounding boundary. The reload audit records the difference and uses a
1e-3 loss tolerance; the independent typed-kernel fp32 gradient gate remains
2e-5. These mixed-precision smoke results establish execution and performance,
not better convergence than the earlier baseline.


### Stream dtype guarantee

With `bf16_residual_stream=True`, attention and MLP outputs rejoin the residual
stream as bf16. The cast is after the expert ensemble, whose accumulation can
remain fp32. This also covers the eager MoE fallback: fp32 routing weights must
not silently promote all later archived, thought, or memory states. Routing
parameters and AttnRes weighted-mixture accumulation remain fp32, and optimizer
master parameters remain fp32.

The sm120 tuning work dumps both Inductor forward and backward Triton, profiles
actual CUDA kernels (excluding duplicate annotation ranges), and checks 344
stage/full-operation launch configurations. See TKF's typed-attn-res kernel guide
and `bench/results/typed_attn_res_optimization_sm120.json` for the retained
configurations and rejected candidate-load experiment. This is a bounded tuning
search; it does not establish a global optimum for the read or the full model.


The tuned 100-step smoke
[`1fsit8q8`](https://wandb.ai/ablations-tinycompany-ai/bibo-ideas/runs/1fsit8q8)
completed with all numeric metrics finite and all 21 read sites logged:
training loss 6.001724, running-20 loss 5.992804, validation loss 6.202785,
109.4k tokens/s at the last measurement, and 46.55 GB peak allocation.
Strict checkpoint loading succeeded; replay validation differed by 0.000100,
within the existing 1e-3 MoE-atomic replay audit tolerance. The kernel's separate
fp32 parity gate remains 2e-5; its worst observed error was 5.8e-6.
