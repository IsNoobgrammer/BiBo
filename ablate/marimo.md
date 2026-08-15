# Driving the molab box (marimo) — the only sanctioned way

Our GPU runs live on a **molab sandbox**: a marimo notebook on a GPU box at a random
`https://sb-<hex>.sb.molab.run` URL. This file is the whole contract for working with it.

> **THE RULE: use `execute-code.sh`. Never take over the session.**
>
> `execute-code.sh` runs code in the kernel's **scratchpad**. It needs no token, holds no
> lock, and — the part that matters — **the user keeps editing and running the notebook at
> the same time you do.** Anything that calls `POST /api/kernel/takeover` (a second
> websocket connection, a hand-rolled CLI, the browser's "take over" button) flips every
> other client to **read-only**, so the user cannot check a thing while you work. That is
> never an acceptable trade. A `box.py` CLI that did exactly this was written and deleted
> on Aug 15 2026 for this reason; do not rebuild it.

---

## 1. Connect

```bash
E=~/.claude/skills/marimo-pair/scripts/execute-code.sh
U=https://sb-<hex>.sb.molab.run          # ask the user for the current box URL

curl -s "$U/api/sessions"                # {"s_uzaedg":{"filename":"/marimo/notebook.py",...}}
bash "$E" --url "$U" -c 'print("alive")'
```

**The URL is the credential.** molab serves the API with no token — the only thing between
the internet and a root shell on that box is the randomness of the hostname. Never paste it
anywhere public. (If a box ever *does* want a token, pass it as `MARIMO_TOKEN=...` in the
env, never `--token`, which is visible in `ps`.)

The "connecting to non-local server" warning is expected and benign for a known molab URL.

Three input forms, all equivalent:

```bash
bash "$E" --url "$U" -c 'print(1)'       # inline
bash "$E" --url "$U" script.py           # a local file
bash "$E" --url "$U" <<'EOF'             # heredoc — use this for anything multi-line
print(1)
EOF
```

There is **no `--timeout` flag**; wrap with the Bash tool's own `timeout`.

## 2. First contact with a NEW box

A new box is **not a blank box.** The notebook arrives with its setup cells already written
— they carry the wandb login, the HF token, the clone, the dataset prefetch. Hand-rolling a
bootstrap instead of running them wastes an hour and gets the credentials wrong.

```bash
bash "$E" --url "$U" <<'EOF'
import subprocess
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
                     "ls /home/marimo/work", shell=True, capture_output=True, text=True).stdout)
EOF
```

Then, in dependency order, run the notebook's own cells (see §4): `imports` → the wandb
login cell → the HF-token cell → `clone_install` → `env` → `config` → `dataset_prefetch`.

Two checks that have burned us:

- **`wandb login` writes `~/.netrc`, NOT `WANDB_API_KEY`.** An `os.environ` probe reports
  "no key" on a correctly-authed box. Check `os.path.exists("/home/marimo/.netrc")`.
- **Verify the GPU from inside the kernel**, not just via `nvidia-smi` in a subprocess:
  `torch.cuda.is_available()`, `get_device_name(0)`, `get_device_capability(0)`. A kernel
  started in a CUDA-less env still shells out to a working `nvidia-smi`.

## 3. The trap that makes a fresh box lie to you

**The kernel replays every cached cell output on connect — including the PREVIOUS box's.**
molab persists the notebook *with its outputs*, so a brand-new box greets you with
`>>> dataset ready`, `downloaded 22 shards`, and `LAUNCHED: arm-a, arm-b, arm-c` before a
single thing has run. On Aug 15 2026 this cost three rounds of "killing" sweeps that never
existed — `pkill` found nothing and the GPU sat at 3 MiB the whole time.

**Confirm state from the filesystem or a freshly printed value. Never from console text.**

```bash
bash "$E" --url "$U" <<'EOF'
import subprocess
print(subprocess.run("ls /home/marimo/work; nvidia-smi --query-gpu=memory.used --format=csv,noheader; "
                     "ps -eo pid,etime,cmd | grep -E '[a]blate' | head", 
                     shell=True, capture_output=True, text=True).stdout)
print("gate:", LAUNCH_TRAINING)      # a printed global is live; a replayed line is not
EOF
```

## 4. Cells: list, read, create, edit, run, delete

`execute-code.sh` runs code in a scratchpad — **top-level names you bind there are
discarded**. To make anything persist, or to make a cell the user can see, go through
`marimo._code_mode` (`cm`) *inside* an execute-code call:

```bash
bash "$E" --url "$U" <<'EOF'
import marimo._code_mode as cm
with cm.get_context() as ctx:
    for cid, cell in ctx.cells.items():
        head = next((l for l in cell.code.splitlines() if l.strip()), "")
        print(f"{cid:9} {cell.name:18} {head[:60]}")
EOF
```

**Address cells by ID, never by name.** Names are not unique — our notebook has four cells
called `_`, one of which is `pkill` and another the sweep launcher.

| task | call |
|---|---|
| read one | `ctx.cells["lEQa"].code` |
| edit | `ctx.edit_cell("lEQa", NEW_CODE)` |
| create | `ctx.create_cell(name="foo", code=..., hide_code=False)` |
| run | `ctx.run_cell("lEQa")` (queued; executes after the context exits) |
| delete | `ctx.delete_cell(cid)` |

`create_cell` **hides code by default** — pass `hide_code=False` for anything the user
should watch.

**Never read-modify-write a cell across two calls.** Build the entire new cell body in one
message and `edit_cell` it once. A stale read once put the previous round's `SWEEP` back
into a cell alongside the new one; the old definition won and the wrong sweep launched.
If you intend to overwrite wholesale, `cm.get_context(skip_staleness_check=True)`.

For long cell bodies, **base64 the source** rather than fighting heredoc + triple-quote
escaping:

```bash
python -c "import base64;print(base64.b64encode(open('cfg.py','rb').read()).decode())" > /tmp/b64
bash "$E" --url "$U" <<EOF
import base64, marimo._code_mode as cm
code = base64.b64decode("$(cat /tmp/b64)").decode()
with cm.get_context() as ctx:
    ctx.edit_cell("lEQa", code)
EOF
```

## 5. Reactivity — the expensive one

marimo is a **reactive DAG**. Running or editing a cell re-runs every *stale descendant*,
and on a freshly started kernel **every** downstream cell is stale — launcher included.

**Before touching any cell on a fresh box, check the launch gate:**

```bash
bash "$E" --url "$U" -c 'print("gate:", LAUNCH_TRAINING)'
```

Our notebook guards this with `LAUNCH_TRAINING`, and the launcher cell is a no-op when it
is `False`. Keep it `False` in the file; flip it only in the same edit that launches.

Related rules:

- **Multiply-defined names.** Every public top-level name has exactly ONE owning cell.
  A cell that does `import subprocess` when the `imports` cell already owns it is rejected
  with `Multiply-defined names`. Just *use* the name — cells share one namespace.
- **Underscore-prefix cell-local intermediates** (`_tmp`) so they never enter the graph.
  But note marimo *mangles* top-level `_name` per cell, so anything referenced later from
  inside another function needs a public name.
- **Freeze expensive results as constants** (`BASE_TPS = 178_454  # measured`) so the cell
  holding them has no costly dependency to recompute.
- Keep heavy runs in **leaf cells** nothing depends on.

## 6. Subprocesses and long jobs

`subprocess` is already imported by the `imports` cell — use it, don't re-import.

```python
print(subprocess.run("cmd", shell=True, capture_output=True, text=True).stdout)
```

For a training sweep, write a **sequential bash script** and launch it detached:

```python
subprocess.run("nohup setsid bash /home/marimo/work/sweep.sh > /home/marimo/work/sweep.log 2>&1 &",
               shell=True)
```

**Never chain jobs on PIDs.** On molab, PID 1 is marimo and never reaps orphans, so an
exited job stays `<defunct>` and `kill -0 <pid>` succeeds forever. That silently left the
GPU idle for hours. A sequential script that runs `arm1; arm2; arm3` is the whole solution.

**`pkill -f 'ablate.common.train'` matches the invoking shell's own command line** — it
kills the shell that ran it (returncode -9) and can match your own probe. Use `'[a]blate'`.

## 7. Watching a run — do it from W&B, locally

Do **not** poll the box in a loop. Runs are logged to W&B, which survives the box dying,
and the query runs on your own machine:

```bash
cd BiBo && ./.venv/Scripts/python.exe -c "
import wandb; api = wandb.Api()
for r in api.runs('ablations-tinycompany-ai/<project>'):
    print(r.name, r.state, r.summary.get('_step'), r.summary.get('val/loss'))"
```

For a snapshot of the live log, one execute-code call is fine:
`tail -3 /home/marimo/work/train_<arm>_seed<N>.log`.

If you must wait on something, use an **hourly cron job**, never a polling loop — the loops
timed out and left the GPU idle twice.

Boxes die of ~2 h inactivity. W&B keeps the results; the box does not.

## 8. Escape hatches

Only `POST /api/kernel/execute` (what `execute-code.sh` wraps) and these are needed:

```bash
SID=$(curl -s "$U/api/sessions" | jq -r 'keys[0]')
curl -sX POST "$U/api/kernel/interrupt" -H "Marimo-Session-Id: $SID"   # stop a hung cell
curl -sX POST "$U/api/kernel/shutdown" -H "Marimo-Session-Id: $SID"    # teardown
```

`interrupt` is the one that saves you when a cell will not finish.

**Do not call `/api/kernel/takeover`.** See the rule at the top.

The molab **VM** is the user's to stop from the molab dashboard — the marimo API can stop
the kernel but not the billing.

## 9. Windows notes

We drive the Linux box from Windows via Git Bash:

- Bash's `/tmp/x` and Windows Python's `/tmp/x` are **different paths**. A heredoc that
  writes `/tmp/f.py` from Git Bash is invisible to a Windows `python` reading `/tmp/f.py`.
  Use full `C:/...` paths in Python, or keep the file on the box.
- A file written *locally* is not on the box. `subprocess` on the box reading `/tmp/x.py`
  gets the box's copy — which may be a **stale** one from an earlier call. Prefer sending
  the code inline over writing a temp file and referencing it later.
- `tar` to a Windows path fails; use `tar --force-local`.
- `LF will be replaced by CRLF` git warnings are harmless.
