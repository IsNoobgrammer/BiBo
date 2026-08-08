"""End-of-run report: generation samples, degeneration metrics, and length extrapolation.

Runs ONCE after the last training step, on the final weights. Three things, logged separately:

  samples/*      4 fixed prompts (2 en, 2 hi) decoded greedily and with sampling, as a W&B table.
                 Fixed prompts on purpose -- the whole point is comparing run N against run N-1, and
                 a prompt that changes between runs makes that comparison meaningless.

  degen/*        Degeneration diagnostics on the generated text: consecutive-token repeat rate,
                 repeated 4-grams, distinct-n. A model that has collapsed into a loop still reports
                 a healthy training loss, so loss alone never catches this.

  extrap/*       Val CE at 1024 / 2048 / 4096 tokens of context. Training runs at seq_len 1024, so
                 2048 and 4096 are strictly beyond what RoPE ever saw. Logged as SEPARATE keys, one
                 panel each -- averaging them into a single "val" number would hide the very thing
                 being measured, which is how the curve BENDS with length.

The extrapolation slices come from the held-out shard (data.split_shards keeps training away from
it), so this is measured on text the model has never seen at any length.

Self-check (no GPU, no model): python -m ablate.common.final_report
"""
from . import _paths  # noqa: F401
import torch

# 2 en + 2 hi. Completion-style rather than instruction-style: this is a BASE model trained on
# FineWeb, so "The meaning of life is" reads in-distribution while "Answer the following:" does not,
# and an out-of-distribution prompt measures the prompt rather than the model.
PROMPTS = [
    ("en", "The meaning of life is"),
    ("en", "Artificial intelligence is changing the world because"),
    ("hi", "जीवन का अर्थ है"),
    ("hi", "भारत एक ऐसा देश है जो"),
]


@torch.no_grad()
def generate(model, tok, prompt, max_new=96, temperature=0.0, top_k=50, device="cuda", amp=None):
    """Greedy when temperature == 0, else top-k sampling. Returns (text, generated_ids).

    No KV cache: this re-runs the full forward per token, which is O(n^2) but is ~100 forwards of a
    <200 token sequence and runs once per training run. A cache here would be a second inference
    path to keep correct against SWA + XSA for no measurable benefit.
    """
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    ctx = amp if amp is not None else torch.autocast("cuda", torch.bfloat16)
    for _ in range(max_new):
        with ctx:
            h = model.model(input_ids=x, use_cache=False)
            h = h.last_hidden_state if hasattr(h, "last_hidden_state") else h[0]
            logits = model.lm_head(h[:, -1]).float()
        if temperature <= 0:
            nxt = int(logits.argmax(-1))
        else:
            v, i = torch.topk(logits / temperature, min(top_k, logits.shape[-1]), dim=-1)
            nxt = int(i[0, torch.multinomial(torch.softmax(v, -1)[0], 1)])
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out, skip_special_tokens=True), out


def degen_metrics(ids):
    """Degeneration diagnostics. All in [0,1]; HIGHER rep_* = more degenerate.

    rep@1     fraction of positions whose token equals the previous token
    rep@4     fraction of 4-grams that already occurred earlier in the sample
    distinct1 unique unigrams / total  (LOWER = more repetitive)
    distinct3 unique trigrams / total
    """
    n = len(ids)
    if n < 5:
        return {"rep@1": 0.0, "rep@4": 0.0, "distinct1": 1.0, "distinct3": 1.0}
    rep1 = sum(1 for a, b in zip(ids, ids[1:]) if a == b) / (n - 1)
    grams4, seen4, rep4 = [tuple(ids[i:i + 4]) for i in range(n - 3)], set(), 0
    for g in grams4:
        if g in seen4:
            rep4 += 1
        seen4.add(g)
    tri = [tuple(ids[i:i + 3]) for i in range(n - 2)]
    return {"rep@1": rep1, "rep@4": rep4 / len(grams4),
            "distinct1": len(set(ids)) / n, "distinct3": len(set(tri)) / len(tri)}


def _holdout_rows(dataset, n_tokens, n_seqs, device, field="input_ids"):
    """Rows of >= n_tokens from the HELD-OUT shard. Returns None when the corpus is Hub-streamed
    (no disjointness guarantee) or the rows are too short for the requested length."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .data import split_shards
    _train, held = split_shards(dataset)
    if not held:
        return None
    rows = []
    for f in held:
        if f.endswith(".parquet"):
            t = pq.ParquetFile(f).read_row_group(0)
        else:
            with pa.memory_map(f, "rb") as h:
                t = pa.ipc.open_stream(h).read_all()
        col = t.column(field) if field in t.column_names else t.column("label")
        for r in col.slice(0, max(n_seqs * 4, 64)).to_pylist():
            if len(r) >= n_tokens:
                rows.append(r[:n_tokens])
            if len(rows) >= n_seqs:
                break
        break
    if len(rows) < n_seqs:
        return None
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.no_grad()
def extrapolation(model, dataset, ce_fn, amp, device, lens=(1024, 2048, 4096), n_seqs=4, pad_id=0):
    """Val CE at each context length. Returns {length: loss}, skipping lengths the corpus cannot fill.

    A 4096-token row yields 4095 predicted positions at length 4096 (the last token has no target),
    so `lens` is capped by the row length rather than silently wrapping into the next document --
    concatenating rows would splice two unrelated documents and inflate the loss for a reason that
    has nothing to do with extrapolation.
    """
    was_training = model.training
    model.eval()
    out = {}
    try:
        for L in lens:
            batch = _holdout_rows(dataset, min(L + 1, 4096), n_seqs, device)
            if batch is None:
                continue
            inp, tgt = batch[:, :-1], batch[:, 1:].reshape(-1)
            with amp:
                h = model.model(input_ids=inp, use_cache=False)
                h = h.last_hidden_state if hasattr(h, "last_hidden_state") else h[0]
                out[L] = float(ce_fn(h.reshape(-1, h.shape[-1]), model.lm_head.weight, tgt,
                                     ignore_index=int(pad_id)))
    finally:
        if was_training:
            model.train()
    return out


def run(model, tok, dataset, ce_fn, amp, device="cuda", wb=None, max_new=96, n_seqs=4,
        lens=(1024, 2048, 4096)):
    """Everything above, printed and (if wb) logged. Never raises into the training script: a
    reporting bug must not destroy a finished run's checkpoint and result.json."""
    flat, rows = {}, []
    try:
        for lang, p in PROMPTS:
            for mode, temp in (("greedy", 0.0), ("sampled", 0.8)):
                txt, ids = generate(model, tok, p, max_new=max_new, temperature=temp,
                                    device=device, amp=amp)
                m = degen_metrics(ids)
                rows.append([lang, mode, p, txt, m["rep@1"], m["rep@4"], m["distinct1"], m["distinct3"]])
                if mode == "greedy":       # degen headline from greedy only: sampling hides loops
                    for k, v in m.items():
                        flat[f"degen/{lang}/{k}"] = v
                print(f"\n--- [{lang}/{mode}] {p}\n{txt}\n    rep@1={m['rep@1']:.3f} "
                      f"rep@4={m['rep@4']:.3f} distinct1={m['distinct1']:.3f} "
                      f"distinct3={m['distinct3']:.3f}", flush=True)
        for k in ("rep@1", "rep@4", "distinct1", "distinct3"):
            vals = [v for kk, v in flat.items() if kk.endswith("/" + k)]
            if vals:
                flat[f"degen/{k}"] = sum(vals) / len(vals)
    except Exception as e:
        print(f"[final_report] sampling FAILED: {type(e).__name__}: {e}", flush=True)

    try:
        ex = extrapolation(model, dataset, ce_fn, amp, device, lens=lens, n_seqs=n_seqs)
        for L, v in ex.items():
            flat[f"extrap/seq{L}"] = v                       # one W&B panel per length, never averaged
        base = ex.get(1024)
        if base:
            for L, v in ex.items():
                if L != 1024:
                    flat[f"extrap/delta_seq{L}"] = v - base  # + = worse than at trained length
        print("\n[extrapolation] " + "  ".join(f"seq{L}={v:.4f}" for L, v in sorted(ex.items())),
              flush=True)
        if base and len(ex) > 1:
            print("[extrapolation] delta vs trained 1024: " + "  ".join(
                f"seq{L}={v - base:+.4f}" for L, v in sorted(ex.items()) if L != 1024), flush=True)
    except Exception as e:
        print(f"[final_report] extrapolation FAILED: {type(e).__name__}: {e}", flush=True)

    if wb is not None and (flat or rows):
        try:
            import wandb
            payload = dict(flat)
            if rows:
                payload["samples"] = wandb.Table(
                    columns=["lang", "mode", "prompt", "completion",
                             "rep@1", "rep@4", "distinct1", "distinct3"], data=rows)
            wb.log(payload)
        except Exception as e:
            print(f"[final_report] W&B log FAILED: {type(e).__name__}: {e}", flush=True)
    return flat


if __name__ == "__main__":
    assert len(PROMPTS) == 4 and [l for l, _ in PROMPTS].count("en") == 2 \
        and [l for l, _ in PROMPTS].count("hi") == 2, "need exactly 2 en + 2 hi prompts"
    # a pure loop must score rep@1 = 1 and minimal distinct-n; varied text must not
    loop = degen_metrics([7] * 40)
    assert loop["rep@1"] == 1.0 and loop["distinct1"] < 0.05, loop
    assert loop["rep@4"] > 0.9, loop
    varied = degen_metrics(list(range(40)))
    assert varied["rep@1"] == 0.0 and varied["distinct1"] == 1.0 and varied["rep@4"] == 0.0, varied
    # a half-repeating sample must land strictly between the two extremes
    mixed = degen_metrics(list(range(20)) + [99] * 20)
    assert 0.0 < mixed["rep@1"] < 1.0 and 0.0 < mixed["rep@4"] < 1.0, mixed
    print(f"final_report self-check OK -- {len(PROMPTS)} prompts (2 en, 2 hi); "
          f"degen metrics separate a loop (rep@1=1.000) from varied text (rep@1=0.000)")
