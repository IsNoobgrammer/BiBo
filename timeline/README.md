# BiBo ablation timeline

Every axis we have swept, what it settled, and where the evidence lives. Built from 574 W&B runs
across 26 projects (1 May - 15 Aug 2026), 558 commits, and the round notes kept alongside them.

**How to read a verdict.** `SHIPPED` means it is in the default stack. `CLOSED NEGATIVE` means it
was measured and rejected; the code may still exist behind a flag. `PARKED` means nothing cleared
the noise floor, so the axis is neither won nor lost. `OPEN` means it is still running or still
unresolved. A result with no noise floor beside it is not a result -- floors are **per-stack** and
never transfer between them.

## Topics

| | Topic | Span | Headline |
|---|---|---|---|
| 01 | [Optimizer](01-optimizer/) | May - Aug | Muon beats AdamW 2x; the NS coefficient axis is dead, weight decay is the live knob |
| 02 | [MoE routing](02-moe-routing/) | Jul - Aug | Balance is free; gate shape, router norm, temperature and identity experts all closed |
| 03 | [Expert activation](03-expert-activation/) | Jul - Aug | Radial NormSiLU shipped; the axis then saturated at three different mechanisms |
| 04 | [Attention](04-attention/) | Aug | XSA shipped with learnable per-head alpha; AttnRes parked at the noise floor |
| 05 | [Positional encoding](05-positional-encoding/) | Aug | Global NoPE + local full RoPE, fixed in code and no longer configurable |
| 06 | [FFN placement](06-ffn-placement/) | Aug | All-MoE beats dense end-layers by 0.066 val at matched active params, for 4.9% throughput |
| 07 | [Kernels and speed](07-kernels-and-speed/) | May - Aug | 128.9k -> 178.8k tps; bit-identity is the ship gate |
| 08 | [Data and tokenizer](08-data-and-tokenizer/) | Aug | Gigatoken 10.1x, parity-gated; the corpus was padded and we were scoring the padding |
| 09 | [Methodology](09-methodology/) | throughout | The lessons that cost the most: noise floors, ablate-downward, measure don't infer |

## Chronology

| Dates | W&B project | Runs | What was being asked |
|---|---|---|---|
| May 1 | `8bit-test` | 7 | 8-bit quantisation viability |
| May 3-4 | `bibo-benchmark`, `bibo-long-context` | 8 | first end-to-end benchmarks |
| May 15-23 | `bibo-ablation` | 51 | the original broad architecture sweep |
| May 23 | `bibo-vs-qwen` | 8 | first head-to-head against a stock baseline |
| May 27 - Jun 28 | `bibo-bench` | 86 | sustained benchmarking |
| Jul 8-13 | `bibo-qwen-ablate` | 44 | param-matched BiBo-min vs Qwen3MoE |
| Jul 13-16 | `bibo-qwen-ablate-optimizer` | 15 | optimizer axis on the matched pair |
| Jul 22-26 | `polyglu-ablations` x3 | 38 | expert activation, three scales |
| Jul 24-25 | `BiBo-Router` | 13 | router gate shape and input norm |
| Jul 26-30 | `bibo-specials-conv` | 176 | special experts, conv router, temperature |
| Jul 30 | `bibo-manas-*` | 7 | Manas optimizer on a real LM |
| Jul 31 - Aug 1 | `bibo-act-1b` | 6 | does the activation win survive to 1B |
| Aug 1-3 | `bibo-xsa-524m` | 17 | XSA at 524M |
| Aug 3-7 | `bibo-attnres`, `-v2`, `-1b` | 57 | Kimi K3 attention residuals |
| Aug 7 | `bibo-clean-v1`, `bibo-lr-sweep` | 13 | post-purge baseline, Muon LR |
| Aug 8-13 | `bibo-baseline-2k` | 16 | the 2000-step board: carry, rope, noise floor |
| Aug 15 | `bibo-dense-vs-moe-2k` | 5 | dense vs MoE end-layers, and the carry interp |

Raw snapshot of every run: [`_data/wandb.json`](_data/wandb.json).

## The five things worth remembering

1. **A measurement without a noise floor is a story.** Floors are per-stack. The 2k board's val
   floor is 0.0217; several "results" from earlier rounds sit under it.
2. **Ablate downward.** Ask "can this be deleted?" before "how do we improve it?" The largest
   wins of the last month were deletions: the rope knobs, the eval package, the NS coefficient axis.
3. **A learned parameter value tells you nothing about whether it matters.** Only intervention does.
4. **Kernels ship on bit identity**, not on being close, and not on being "more accurate".
5. **Verify the artifact, not the plan.** Grep finds definitions; only the test suite finds callers.
