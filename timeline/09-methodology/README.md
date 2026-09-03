# 09 - Methodology

The lessons that cost the most time. Every one was learned by getting it wrong first.

## Measurement

**Noise floors are PER-STACK and never transfer.** The `bibo-baseline-2k` floors: val 0.0217,
ctx1024 0.0158, ctx4095 0.0131, delta_ctx4095 0.0027, winogrande 0.0101. A code change at fixed
seed moved val 0.0127 and winogrande 0.1388 - so winogrande is a **code-change detector**, not a
seed-noise metric. A new project starts with no floor, and its first arm IS the baseline.

**A learned parameter value does not tell you whether it matters.** Layer 0's carry settled at
0.009 in one run and 0.781 in another, at the same loss; removing it costs 0.43 in one model and
2.74 in the other. Only intervention separates a flat direction from a load-bearing one. And an
ablation measures **this model's dependence**, not the mechanism's necessity - a model trained
without the term adapts around it over 2000 steps.

**Rank on the absolute metric, not the delta.** A uniformly worse model degrades less and therefore
wins on deltas.

**Measure the noise floor of the measurement itself.** The fused MoE kernel is nondeterministic:
six identical forwards spread 1.1e-4. Without establishing that first, every small delta becomes a
story.

**Throughput must be paired and interleaved**, or not quoted at all. See `07-kernels-and-speed`.

**A monotone dose across three arms beats a single pairwise gap.** Three arms ordering the same way
across four metrics is evidence; one gap of the same size is a coin flip.

## Building

**Ablate downward.** Ask "can this be deleted?" before "how do we improve it?" The biggest recent
wins were deletions: the rope knobs, the whole eval package, the NS coefficient axis.

**A deletion is not done when grep is clean.** Grep finds the definitions you removed; only the
test suite finds the callers. One deletion passed a grep and broke model construction for a day.

**Run pytest locally.** `BiBo/.venv` has torch and the suite is about 15 seconds. Do not route a
correctness check through a GPU box.

**A silently inert flag is worse than a crash.** Rounds have been spent on arms that never applied:
`ACT_CYCLE` not reaching the eager path, XSA alpha dead behind a passing parity test, `report_ckpt`
rebuilding the default architecture for a checkpoint that was not default. `strict=True` on
`load_state_dict` is what catches that last class.

**Verify the artifact, not the plan.** Read the launch script the sweep actually wrote, not the
config you believe you set. The argparse default is frequently dead code.

**When a parity test fails, suspect the test.** The Muon FAIL at 6.7e-02 was the bench's own
reference; corrected, the kernel matched at 9.5e-07.

## Operating the box

**Drive molab cell-wise with `execute-code.sh`, never by taking over the session.** Takeover flips
every other client to read-only and locks the user out of their own notebook. Full playbook:
`ablate/marimo.md`.

**A new box is not a blank box.** Its notebook already carries the wandb login, HF token, clone and
dataset cells. Run them; do not hand-roll a bootstrap. `wandb login` writes `~/.netrc`, not
`WANDB_API_KEY`, so an environment probe reports a false negative on a correctly-authed box.

**The kernel replays cached output on connect, including the previous box's.** A fresh box will
announce `dataset ready` and `LAUNCHED: ...` before anything has run. Confirm state from the
filesystem or a freshly printed value, never from console text.

**Query W&B from the local terminal.** Runs survive box death; boxes die of 2h inactivity.

**Durability is not set up, it is verified.** Twice a queued job or mirroring loop was supposed to
preserve results and did not - one died with its box, one wrote a zero-byte file. Check that the
artifact exists and is non-empty before believing the data is safe.

**Never chain jobs on PIDs.** PID 1 on molab is marimo and never reaps orphans, so a dead job
satisfies `kill -0` forever. Use a sequential script and `pgrep -f` on a bracketed pattern.

**Reactivity fires expensive cells you did not ask for.** Editing a cell re-runs every stale
descendant, and on a fresh kernel that is everything downstream, launcher included. Check the
launch gate before touching any cell.

## Sources

Memory: `bibo-noise-floor`, `simplicity-philosophy`, `parity-vs-plumbing`,
`kernel-contract-bit-identity`, `local-venv-testing`, `sweep-methodology`, `marimo-cell-wise`,
`box-run-hygiene`, `wandb-local`, `poll-with-cron`, `box-tps-drift`, `carry-is-flat`,
`muon-parity-bench-trap`, `eval-purged`.
