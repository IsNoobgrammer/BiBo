# BiBo ablation timeline

Every axis we have swept, what it settled, and where the evidence lives.

Built from **574 W&B runs across 26 projects** (1 May - 15 Aug 2026), **558 commits**, and the
round notes kept alongside them. Nine topic folders, each with its verdicts, the numbers behind
them, and the caveats that have to travel with the result.

Start here, then open a topic. Every page exists twice: `README.md` for reading in the repo,
`index.html` for reading in a browser. **The markdown is the source** - see
[Regenerating](#regenerating).

---

## How to read a verdict

| Word | Means |
|---|---|
| **SHIPPED** | in the default stack today |
| **CLOSED NEGATIVE** | measured and rejected; the code may survive behind a flag |
| **PARKED** | nothing cleared the noise floor - neither won nor lost |
| **OPEN** | still running, or still unresolved |

A result quoted without its noise floor is not a result. Floors are **per-stack** and never
transfer between them; the current board's val floor is **0.0217**.

---

## Topics

### [01 - Optimizer](01-optimizer/README.md)
Muon beats AdamW ~2x. Weight decay is the dominant knob (7x any coefficient change), and once wd
was matched the entire Newton-Schulz coefficient family collapsed to a tie. Muon LR 3e-3, 4.3 sigma
over the incumbent. Symmetric-matmul and gram-space NS both shipped on sm120.

### [02 - MoE routing](02-moe-routing/README.md)
Load balance is solved and free (entropy 0.9996, no aux loss). Gate shape, router input norm,
temperature, signed identity experts, top-k reduction, MoE output norm and LatentMoE are all
closed. Temperature has an **interior** optimum at T=2 - sharper is not better in either direction.

### [03 - Expert activation](03-expert-activation/README.md)
Radial NormSiLU shipped and is the default. The axis then **saturated**: three unrelated mechanisms
all land at ~0.6767. The win shrinks 3.6x going from 524M to 1B, so a small-scale activation
advantage is an upper bound, not a preview.

### [04 - Attention](04-attention/README.md)
XSA shipped with learnable per-head alpha, init 0 so the model switches it on itself. AttnRes is
parked after 57 runs - its default was chosen on a **1.5% throughput edge, not quality**. The
carry turned out to be a flat direction: the whole 512-dim vector is replaceable by the scalar 1.

### [05 - Positional encoding](05-positional-encoding/README.md)
Global NoPE + local full RoPE, now **fixed in code rather than configurable**. The global layers
govern extrapolation; local rotary width is irrelevant to it. A dead dynamic-NTK branch had been
silently rescaling the base at eval, making every pre-Aug-13 extrapolation number incomparable.

### [06 - FFN placement](06-ffn-placement/README.md)
All-MoE beats dense end-layers by **0.066 val** at matched active params, monotone across three
arms, for 4.9% throughput. Matching active params is what made it an ablation rather than a
capacity arm in disguise.

### [07 - Kernels and speed](07-kernels-and-speed/README.md)
128.9k to 178.8k tps. Bit identity is the ship gate - a kernel that was *more accurate* than eager
cost real bpb. Microbenchmarks mispredicted one real cost by 13x in both directions.

### [08 - Data and tokenizer](08-data-and-tokenizer/README.md)
Gigatoken is 10.1x faster and shipped on **0 mismatches over 3200 documents**, not on speed. The
bip2 corpus turned out to be padded, not packed - 15.93% of tokens - and we were scoring the
padding.

### [10 - Per-layer interp](10-per-layer-interp/README.md)
The first fully instrumented run. Attention type organises everything: global layers suppress the
carry (0.844) while sliding layers amplify it with depth (up to 1.498). L1 runs XSA backwards and
collapses radial p to pure NormSiLU. The routers saturate - top logits near 200 under a sigmoid
gate - and expert identity churns rather than specialising.

### [09 - Methodology](09-methodology/README.md)
The lessons that cost the most time, all learned the hard way. Noise floors, ablate downward,
a learned parameter value tells you nothing about whether it matters, and durability is verified
rather than set up.

---

## Chronology

| Dates | W&B project | Runs | What was being asked |
|---|---|---|---|
| May 1 | `8bit-test` | 7 | 8-bit quantisation viability |
| May 3-4 | `bibo-benchmark`, `bibo-long-context` | 8 | first end-to-end benchmarks |
| May 15-23 | `bibo-ablation` | 51 | the original broad architecture sweep |
| May 23 | `bibo-vs-qwen` | 8 | first head-to-head against a stock baseline |
| May 27 - Jun 28 | `bibo-bench` | 86 | sustained benchmarking |
| May 29-30 | `latentsig-med-triage-router` | 7 | router work on a separate task |
| Jul 8-13 | `bibo-qwen-ablate` | 44 | param-matched BiBo-min vs Qwen3MoE |
| Jul 13-16 | `bibo-qwen-ablate-optimizer` | 15 | optimizer axis on the matched pair |
| Jul 22-26 | `polyglu-ablations`, `-large`, `-XL` | 38 | expert activation at three scales |
| Jul 24-25 | `BiBo-Router` | 13 | router gate shape and input norm |
| Jul 26-30 | `bibo-specials-conv` | **176** | special experts, conv router, temperature |
| Jul 30 | `bibo-manas-2k`, `-3way`, `-ladder` | 7 | Manas optimizer on a real LM |
| Jul 31 - Aug 1 | `bibo-act-1b` | 6 | does the activation win survive to 1B |
| Aug 1-3 | `bibo-xsa-524m` | 17 | XSA at 524M |
| Aug 3-7 | `bibo-attnres`, `-v2`, `-1b` | 57 | Kimi K3 attention residuals |
| Aug 7 | `bibo-clean-v1`, `bibo-lr-sweep` | 13 | post-purge baseline, Muon LR |
| Aug 8-13 | `bibo-baseline-2k` | 16 | the 2000-step board: carry, rope, noise floor |
| Aug 15 | `bibo-dense-vs-moe-2k` | 5 | dense vs MoE end-layers, and the carry interp |
| Aug 25 | `bibo-dense-vs-moe-2k` | +1 | all-MoE at a second seed, fully instrumented per layer |

Every run indexed locally in [`_data/wandb.json`](_data/wandb.json) - W&B is the system of record,
but boxes and results have been lost before.

**Coverage gap, stated plainly.** The pre-July projects (`bibo-ablation`, `bibo-bench`, `8bit-test`,
`bibo-long-context`, `bibo-vs-qwen`) predate the round notes and logged only `_step` - no readable
loss key. They appear above as *what was being asked*, never as verdicts, because the conclusions
are not reconstructable from what survives.

---

## The five things worth remembering

1. **A measurement without a noise floor is a story.** Floors are per-stack. Several older
   "results" sit under the current board's 0.0217.
2. **Ablate downward.** Ask "can this be deleted?" before "how do we improve it?" The biggest recent
   wins were deletions: the rope knobs, the eval package, the NS coefficient axis.
3. **A learned parameter value tells you nothing about whether it matters.** Only intervention does
   - and even then it measures *this model's* dependence, not the mechanism's necessity.
4. **Kernels ship on bit identity**, not on being close and not on being more accurate.
5. **Verify the artifact, not the plan.** Grep finds definitions; only the test suite finds callers.

---

## Open questions

| Topic | Question | What would settle it |
|---|---|---|
| 04, 06 | Is the AttnRes carry deletable? | one arm at `attn_res_carry_scale=none, per_dim=False`; predicted within 0.003 |
| 04 | Does the L0 carry collapse reproduce on raw? | `carry_ablate` on a raw checkpoint - both raw ones died with their box |
| 06 | Does all-MoE hold at a second seed? | one repeat; the 0.066 is single-seed |
| 01 | Does the Manas champion transfer to BiBo? | 2k-scale arm |
| 01 | `ns8` free compression | 121M A/B |
| 03 | Where does the activation win go at 3B+ | the gap shrank 3.6x from 524M to 1B |

---

## Regenerating

```bash
./.venv/Scripts/python.exe timeline/build.py    # renders every README.md to index.html
```

Edit the markdown, never the HTML - the HTML carries a generated-file header and is overwritten.
Adding a topic means adding `NN-name/README.md` and rerunning; the build picks it up.

To refresh the run index after new experiments, re-run the snapshot in `_data/` against W&B.
