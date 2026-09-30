# benchmark

Downstream benchmarks for big runs. A run that passes `--hf_repo <repo> --ckpt_every N` to
`ablate.common.train` pushes `step<N>/` folders (weights + `train_args.json`) and a final checkpoint at the
repo root; everything here loads from those, so any checkpoint can be benchmarked long after the box died.

    # on the box (HF_TOKEN in the env for a private repo); ~10 min for the full set at 150M active
    python -m benchmark.eval --repo fhai50032/bibo-base-1b-6k-s23 --sub step4000
    python -m benchmark.eval --repo ... --sub step3000 --samples_only      # en + hi text samples only
    python -m benchmark.eval --repo ... --tasks belebele_eng,belebele_hin    # add tasks to the same file

    # anywhere
    python -m benchmark.board

- `eval.py` -- lm-eval-harness-style scoring, 0-shot and 5-shot by default (`--shots`). English: HellaSwag,
  ARC-Easy, PIQA, WinoGrande, LAMBADA, Belebele-eng. Hindi: XStoryCloze-hi, Belebele-hin. The experts run
  the way the run trained (fp8 for a `--moe_fp8` run, via `report_ckpt.load_from_hub`).
- `results/<repo>__<sub>.json` -- one file per checkpoint, merged across runs; commit them.
- `references.json` -- published numbers for comparable models, marked approximate until rerun.
- `board.py` -- all results and references in one markdown table per shot count.

Few-shot demonstrations are drawn from the scored split itself (never the scored item), a fixed sample
per item. Numbers are comparable across our checkpoints; against published few-shot tables they are close
but not identical (those draw demonstrations from the train split).
