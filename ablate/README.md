# ablate — BiBo-min vs Qwen3MoE, parameter-matched

Isolates whether BiBo's core (PolyGLU expert-activation diversity + partial RoPE) beats a
parameter-matched Qwen3MoE baseline, with **explicit downstream signal** including **Hindi**.
Deliberately does NOT reuse the Kaggle-bench eval (same-distribution val-bpb + near-chance
multiple-choice), which gave zero architectural signal.

## Arms (2, bundled)
- **qwen** — stock Qwen3MoE: SwiGLU experts, full RoPE, softmax router (aux-loss OFF).
- **bibo_min** — Qwen-equivalent EXCEPT: PolyGLU experts (silu/relu²/normsilu cycled) + partial RoPE.
  Stripped: no SWA, no conv router, no XSA, no SSMax, no sinks, no shared expert, no load-balancing.

Both are **137.47M total / 71.41M active, trainable Δ=0 (exact match)** — PolyGLU==SwiGLU in params
and partial-vs-full RoPE is parameter-free, so the match is exact by construction (BiBo carries 72
inert, non-trainable router-bias params).

## Layout
```
common/            hardware-agnostic, modular, swappable
  configs.py       the 2 matched arm configs (every knob a flag)
  models.py        build_arm(name) + count_params
  patches.py       apply({liger_norm,liger_rope,moe}) — patches BOTH arms; CE is a train-loop component
  optim.py         bf16-safe FusedMuon (NS8, aurora-K1) + AdamW
  schedule.py      WSD (warmup 0.05, stable, linear decay)
  data.py          QTK-81K packed instruct stream (+ synthetic mode for smoke)
  train.py         one arm / one seed / one run — trains AND evals PERIODICALLY (every --eval_every
                   steps) + a full final eval, all logged to W&B as curves. Checkpoints + result json.
  evaluate.py      shared evaluate() used by both periodic (cheap) and standalone (full) eval
  run_eval.py      STANDALONE re-eval of a saved checkpoint (full suite on demand); not the main path
  eval/
    manifest.py    bpb text sources (FLORES en/hi, GSM8K en/hi) — Hindi mandatory, swappable
    bpb.py         tokenizer-independent bits-per-byte, per language/domain
    mcq.py         log-likelihood MCQ (Belebele, XNLI, optional Global-MMLU), per language
  smoke.py         local build+patch+param-match+train verification
rtx_6000/run.sh    single GPU, global batch 40, 1B tokens, bf16
t4_x2/run.sh       2x T4, one arm per GPU, micro 10 x accum 4 = same global batch 40
```

## Data
- **Train**: `tinycompany/Instruct-packed-2K-Context-tk-QTK-81K` (own multi-domain, multi-lingual incl
  Hindi instruct amalgamation; disjoint from public benchmarks). Tokenizer `fhai50032/QTK-81K` (vocab 81000).
- **Eval** (public HF, En+Hi parallel): FLORES-200 + GSM8K(-Hindi) for bpb; Belebele + XNLI (+ optional
  Global-MMLU) for LL-MCQ. Hindi is always a separately-reported segment.

## Signal-first eval
1. **Per-language bits-per-byte** on held-out raw text (Hindi vs English), tokenizer-independent.
2. **Length-extrapolation bpb** (train 1024 → eval longer) — probes partial vs full RoPE. *(TODO)*
3. **LL-MCQ accuracy** (pick highest per-token-logprob option; works at 137M where generation doesn't).
4. **Capability probes** (synthetic induction + selective-copy, English AND Hindi).
5. Read run-to-run variance directly in W&B across seeds — no in-code seed aggregation by design.

**All of the above run PERIODICALLY during training** (every `--eval_every` steps, default 500) so you
watch bpb[hi]/bpb[en], acc[hi]/acc[en], and RoPE-extrapolation degradation evolve as curves in W&B —
not just a single number at the end. Periodic eval uses small samples (cheap, cached); a full eval runs
at the end. `run_eval.py` is only for re-scoring a saved checkpoint on demand.

## Run
```bash
# one full run (both arms + eval), one seed:
SEED=0 bash ablate/rtx_6000/run.sh
SEED=0 bash ablate/t4_x2/run.sh
# local smoke (no downloads):
python -m ablate.common.smoke
python -m ablate.common.train --arm bibo_min --data synthetic --max_steps 5 --batch 2 --seq_len 128 --patches liger_norm,liger_rope,ce,moe --ckpt_every 0
```
Precision is **bf16 or fp32 only — never fp16** (fp16 + Muon overflows the MoE PolyGLU experts; see the
fp16-divergence finding). Compare `bpb[hi]`/`bpb[en]` and `acc[hi]`/`acc[en]` between arms in W&B.

## Arguments

`python -m ablate.common.train`
| arg | default | meaning |
|---|---|---|
| `--arm` | (required) | `qwen` \| `bibo_min` |
| `--seed` | 0 | RNG seed (one seed per run; compare seeds in W&B) |
| `--tokens` | 1_000_000_000 | token budget (ignored if `--max_steps>0`) |
| `--batch` | 40 | micro-batch per step |
| `--grad_accum` | 1 | global batch = `batch*grad_accum` (match across hardware) |
| `--seq_len` | 1024 | sequence length |
| `--precision` | bf16 | `bf16` \| `fp32` (never fp16) |
| `--attn` | sdpa | `sdpa` \| `flash_attention_4` (auto-downgrades to sdpa if flash_attn/arch missing) |
| `--patches` | liger_norm,liger_rope,ce,moe | any subset of these 4 |
| `--muon_lr` / `--adam_lr` | 3e-4 | LRs |
| `--wd` | 0.1 | weight decay |
| `--warmup_frac` / `--decay_frac` | 0.05 / 0.20 | WSD phases |
| `--grad_clip` | 1.0 | max grad norm (0=off) |
| `--data` | real | `real` (QTK-81K corpus) \| `synthetic` (smoke) |
| `--max_steps` | 0 | >0 overrides token budget (smoke) |
| `--log_every` / `--ckpt_every` | 20 / 2000 | logging / checkpoint cadence (ckpt 0=only final) |
| `--eval_every` | 500 | **periodic in-training eval → W&B curves** (0 disables; try 200/500/1000) |
| `--eval_mcq_n` / `--eval_bpb_n` | 200 / 200 | cheap periodic eval sample sizes |
| `--eval_extrap` | "" | periodic length-extrap windows (default off; e.g. 1024,2048,4096) |
| `--final_mcq_n` / `--final_extrap` | 500 / 1024,2048,4096 | full final eval |
| `--out` | ../runs | output dir |
| `--wandb` / `--wandb_project` | off / polyglu-ablations | W&B logging |
| `--silu` / `--relu2` / `--normsilu` | 1 / 1 / 1 | PolyGLU act subset (bibo_min): the ENABLED set cycles across the 6 experts (silu only -> 000000, silu+relu2 -> 010101, all -> 012012). Codes: silu=0, relu2=1, normsilu=2. Needs the `moe` patch. |
| `--polyglu_mult` | 2 | GLU experts = mult*3 -> default 6 (LCM of 2,3: any act subset tiles evenly) |

`python -m ablate.common.run_eval`
| arg | default | meaning |
|---|---|---|
| `--arm` | (required) | `qwen` \| `bibo_min` |
| `--ckpt` | (required) | checkpoint path |
| `--precision` | bf16 | `bf16` \| `fp32` |
| `--attn` | sdpa | `sdpa` \| `flash_attention_4` |
| `--patches` | liger_norm,liger_rope,moe | eval-time patches (ce irrelevant) |
| `--seq_len` | 1024 | bpb / mcq context length |
| `--extrap_lengths` | 1024,2048,4096 | length-extrapolation windows (en+hi) |
| `--mcq_n` | 500 | items per MCQ source |
| `--with_global_mmlu` | off | add Global-MMLU (gated, near-chance) |
| `--no_probes` | off | skip capability probes |
| `--out` | ckpt dir | results json dir |
| `--wandb` / `--wandb_project` | off / bibo-qwen-ablate | W&B logging |

Every eval reports **English AND Hindi separately** (`per_language` incl `hi`): bpb, length-extrapolation
(degradation), LL-MCQ (Belebele/XNLI), and probes (induction/copy). Hardware presets: `rtx_6000/run.sh`
(batch 48, FA4) and `t4_x2/run.sh` (micro 12 × accum 4 = batch 48, SDPA).
```
