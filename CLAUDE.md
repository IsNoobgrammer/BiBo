@AGENTS.md
@ablate/marimo.md

---

## Behavior Rules (Always-On — Follow Every Time)

0. **The molab box is driven with `execute-code.sh`, never by taking over the session.** Full rules in @ablate/marimo.md — read it before touching the box. Taking over flips every other client to read-only and locks the user out of their own notebook while you work.


1. **TODO first, always.** Before touching anything non-trivial, write a concrete TODO list. Each item must be self-verifiable — "works" is not a goal, "runs without error and returns X" is. Freeze the list once work starts. Never quietly drop or rewrite goals mid-task. If a goal turns out impossible, say so explicitly.

2. **Don't promise what you can't deliver.** Before committing to anything, ask: can I actually do this given the tools, access, and information I have right now? If no → say so upfront. A scoped honest answer beats a confident promise that falls apart two steps in.

3. **Be brutally honest.** No false optimism. If the approach has a known flaw, name it. If the result is partial or fragile, say so. If something will likely break in prod, warn about it.

4. **EDIT. DON'T REWRITE. NON-NEGOTIABLE.** Read file → locate exact lines → emit only those lines changed. Change 3 lines → emit 3 lines. Never reformat, re-indent, or reorder anything not broken. Only legitimate full-file output: file is <20 lines OR user explicitly said "rewrite from scratch."

5. **Clarify before acting on ambiguous tasks.** If a task is vague or missing critical details — stop. Ask the minimum set of questions needed to unblock. Only proceed once unknowns that would force a rewrite are resolved.

6. **Options before questions when user is unfamiliar.** If the user seems unfamiliar with the problem space, present 3–4 concrete options (name / upside / downside / fits-when) instead of open-ended questions. Then ask one focused question based on their reaction.

7. **Never self-assign a choice on an ambiguous task.** Surface the tradeoff, let the user decide, then implement. Exception: trivial tasks where all options converge — pick sensible default and state it in one line.

8. **Clean up after yourself.** Delete every temp file, directory, or intermediate artifact once the task is done. Leave the workspace exactly as found.

9. **No fluff.** No greetings, sign-offs, or filler phrases ("Great question!", "Certainly!"). Get straight to the point.

10. **Read before you act.** Never execute, edit, or create anything without first reading the relevant file(s).

11. **Confirm before destruction.** Before anything irreversible (`rm -rf`, `DROP TABLE`, force push, overwriting without backup) — state what you're about to do and why. One-line heads-up, mandatory.

12. **Keep AGENTS.md in sync on major architectural changes.** Whenever a major architectural decision changes — a core approach swap, storage backend, schema/data-model redesign, framework/language change, data strategy, scoring/algorithm, or a major feature add/remove — update `AGENTS.md` in the **same session**, as part of the change. Rule of thumb: if it's documented in AGENTS.md and it's now changing, AGENTS.md changes with it. Minor bug fixes, refactors, and cosmetic edits do NOT require an update.

13. **Never implement a feature without explicit approval — NON-NEGOTIABLE.** Do not add any feature, capability, or design element the user has not explicitly said "yes, build it" to — even one you're certain is necessary (e.g. adjective polarity to separate low-fever from high-fever). When you spot something worth adding: STOP, present it as a proposal — what it is, pros/cons, alternatives considered, and what it affects and how — then wait for an explicit go-ahead. Bundling an unrequested feature into other approved work is a violation. "I noticed we need X" → propose and ask; never build unprompted. This applies even mid-implementation of an approved task: new sub-features still need their own yes.

14. **Discussion ≠ writing. Do NOT write or edit ANY file during discussion — NON-NEGOTIABLE.** While we are talking, planning, or designing — including code, documentation, and markdown/`.md` docs — produce NOTHING on disk. Writing/editing files happens ONLY when (a) the user explicitly says "write this" / "implement this" / "go ahead", or (b) we are clearly in the coding/implementation phase. A task phrased as "let's discuss X and put it in doc.md" is still a discussion until the user gives the explicit go — discuss first, write only on the green light. When in doubt, talk and ask "want me to write this up now?" — never assume. (CLAUDE.md/AGENTS.md self-updates from rule #12 and edits the user directly requests are exempt.)

15. **ALWAYS use the project venv — NON-NEGOTIABLE.** Every Python execution and every `pip install` runs in the repo venv at `.venv` (`.venv/Scripts/python.exe` on Windows), NEVER the global interpreter. Never `pip install` globally; never invoke bare `python`/`pip` for project work. If `.venv` is missing, create it (`python -m venv .venv`) and install into it. The ONLY exception is when the user explicitly says to run in the global shell. This keeps runs reproducible and the global environment clean.

---

## Training Data Facts (measured, not assumed)

**The training corpus is PADDED, and the loader does not mask it. Pass `--pad_id 0` on every run.**

Measured on `tinycompany/Better-Instruct-packed-2` (`bip2`), the default corpus, over 2000 rows:

| | |
|---|---|
| trailing `<\|endoftext\|>` padding | **15.93%** of all tokens |
| id 0 anywhere except a trailing run | **0.000%** |
| row shape | 1719 content + 329 pad, `attention_mask` sums to 1719 |

Despite the name, rows are padded to 2048, not packed. `ablate/common/data.py` reads only
`input_ids` — it flattens rows into a buffer and re-cuts at `batch × (seq_len+1)`, so
`attention_mask` is never consulted and the padding enters training as real targets.

`_ce` in `ablate/common/train.py` also passed no `ignore_index`, so the kernel default of `-100`
applied and pad ids (non-negative) never matched it. **Every run before Aug 7 2026 trained on that
~16%.** Fixed by `--pad_id 0`, which is now the correct default for this corpus.

Why id 0 is safe to mask here: QTK-81K has `pad_token_id == eos_token_id == 0` (`<|endoftext|>`),
so masking it also masks end-of-text. That would normally be dangerous — a model that never sees an
EOS target never learns to stop. It is safe in this corpus **only because documents are delimited by
`<|im_end|>` (81914) and 0.000% of id 0 appears outside a trailing pad run**. Training prints a
warning restating this; if the corpus ever changes, re-measure before trusting `--pad_id 0`.

Do NOT add pad masking to bpb. `eval/bpb.py::_text_nll_bytes` encodes one raw text at a time with
`add_special_tokens=False` and windows the real sequence, so no padding exists there. Masking would
shrink the NLL numerator while the byte denominator stayed, faking an improvement and breaking
comparability with the whole board.

**When building a new corpus**: pack it (concatenate with an `<|im_end|>` separator, cut at the row
length, no padding). Then `--pad_id` is unnecessary and no batch compute is wasted.

## Tokenization: use gigatoken, not HF tokenizers (Aug 7 2026)

`gigatoken` (PyPI, Rust BPE with Python bindings) is a drop-in replacement for the HF fast
tokenizer and is **10.1x faster** on our corpus. Adopt it for any bulk tokenization.

```python
import gigatoken as gt
tk  = gt.Tokenizer("/path/to/qtk_patched.json")   # see the byte_fallback note below
ids = tk.encode_batch_list(list_of_strings)       # native API, fastest
hfc = tk.as_hf()                                  # HF-compatible __call__ if you need the shim
```

Measured on 4000 English fineweb-edu docs, single process:

| | throughput | speedup |
|---|---|---|
| HF fast tokenizer | 13.0 MB/s | 1x |
| gigatoken `as_hf()` | 89.6 MB/s | 6.9x |
| gigatoken native `encode_batch_list` | 131.5 MB/s | **10.1x** |

The advertised 989x is their native file-reading path on a different tokenizer; ours is 10x. At
131 MB/s a ~33 GB corpus tokenizes in about 4 minutes single-process, so **tokenization is no longer
the bottleneck and `datasets.map(num_proc=...)` is not needed** — a plain `batched=True` map, or
just reading parquet directly, is enough.

**QTK-81K needs one patch.** `fhai50032/QTK-81K` declares `model.byte_fallback: true` but ships no
`<0xHH>` byte tokens, so gigatoken refuses to load it:

    RuntimeError: byte_fallback is set but the vocab has no <0xHH> byte tokens

Clearing the flag is safe **because the vocab has no byte tokens for it to ever fall back to**:

```python
d = json.loads(hf.backend_tokenizer.to_str())
d["model"]["byte_fallback"] = False
```

That argument is not what licenses the change — the parity check is. Gate any tokenizer swap on
**exact token-id equality against the HF tokenizer**, on real corpus text, sampled from several
positions. Ours: **0 mismatches / 3200 docs** (Hindi and English, at start/quarter/middle/late of
the corpus). Same rule as kernels: bit identity or it does not ship.
