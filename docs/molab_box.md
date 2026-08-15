# Driving a molab box from the terminal — `ablate/tools/box.py`

Our runs live on [molab](https://molab.marimo.io) sandboxes: a marimo notebook on a GPU box,
reachable at a random `https://sb-<hex>.sb.molab.run` URL. `box.py` talks to that notebook's
kernel over the same websocket + REST API the browser editor uses, so everything it does is
cell-wise and visible in the notebook the user has open.

```bash
export MOLAB_URL=https://sb-9f6b222867d3d680.sb.molab.run     # or pass --url
python -m ablate.tools.box ls
python -m ablate.tools.box 'nvidia-smi'
```

**The URL is the credential.** The box serves its API with no token — the only thing standing
between the internet and a root shell on it is the randomness of that hostname. Never paste it
into a public place, and treat a leaked URL as a compromised box.

Needs `websockets` locally; nothing is installed on the box.

## Verbs

| | |
|---|---|
| `<shell command>` | run it on the box (the default verb — no keyword) |
| `py '<code>'` | run python **in the kernel's namespace** — sees `SWEEP`, `LAUNCH_TRAINING`, `subprocess` |
| `-f local.py` | run a local python file up there |
| `ls` | list cells: id, name, first line |
| `cat ID` | print one cell's source |
| `run ID [ID...]` | run cells — **reactive**, see below |
| `edit ID file.py [--run]` | replace a cell's source and save it to `notebook.py` |
| `new NAME file.py [--after ID]` | create a cell |
| `rm ID` | delete a cell |

`ls` first, always. Cell ids are what you pass everywhere else, and **names are not unique** —
a typical notebook has four cells named `_`, one of which is `pkill` and another the sweep
launcher. Selecting by name is how you delete the wrong thing.

```
Hbol      imports            # shared imports (single owner for os/sys/subprocess/torch/Path/mo)
lEQa      config             # EDIT ME - ROPE/NOPE ROUND, 4 arms x 2000 steps.
SFPL      start_training     # RUN THIS to launch the whole SWEEP in the background
Hstk      _                  # LAUNCH the current SWEEP sequentially on the single GPU
```

## Two execution paths, and when to use which

**Scratchpad** (`<shell>`, `py`, `-f`) shares the kernel's globals but sits *outside* the reactive
graph. A probe there cannot trigger anything else. This is the default for status checks,
monitoring, and launching.

**Real cells** (`run`, `edit`, `new`) are where round configuration belongs — the notebook is the
reproducible record of what actually ran, and a config that lives only in a terminal history is a
result nobody can reproduce. But they **cascade**: marimo re-runs every stale descendant of what
you touch, and on a freshly started kernel *everything* downstream is stale, launcher included.

> On a fresh box, check the launch gate before you touch any cell.
> `python -m ablate.tools.box py 'print(LAUNCH_TRAINING)'`

## Three traps this tool encodes

Each one cost a debugging round; the code carries them as comments so they stay fixed.

**1. `run` does not persist.** Overriding `codes` in `/api/kernel/run` changes the live kernel and
nothing else — `notebook.py` on disk keeps its old text, and a kernel or box restart silently
reverts you. We shut a launch gate this way, confirmed `LAUNCH_TRAINING = False` straight from the
kernel, and the file still said `True`. Only `/api/kernel/save` writes the file, which is why
`edit` exists. `save` takes the *whole* cell list, so `new` and `rm` go through it too —
`/api/kernel/delete` drops a cell from the kernel only and it reappears on reconnect.

**2. The kernel replays cached output on connect, including the previous box's.** molab persists
the notebook *with its outputs*, so a brand-new box greets you with `>>> dataset ready`,
`downloaded 22 shards`, and `LAUNCHED: rope-g334-l334, ...` before anything has run. We read that
as live and spent three rounds killing sweeps that did not exist. `box.py` drains 3 s of replay
before issuing any request and ignores every cell event until then — but the discipline still
applies to you: **confirm state from the filesystem (`ls`, `nvidia-smi`) or a freshly printed
value, never from console text.**

**3. A second connection is read-only.** Until it calls `/api/kernel/takeover`, any write returns
`This connection is read-only for this action.` `box.py` takes over on every invocation, which
will bump a browser tab you have open on the same notebook into read-only.

## Wire protocol, if you need to extend it

- `GET /` with a browser `User-Agent` (python-urllib's default gets a 403) → sets the proxy session
  cookie and carries `<marimo-server-token data-token=...>`, needed as `Marimo-Server-Token`.
- `GET /api/status` → notebook filename, GPU/session state. Open, no auth.
- `wss://HOST/ws?session_id=<any new id>` → `kernel-ready` carries `cell_ids`, `codes`, `names`,
  `configs`; then a stream of `cell-op` messages.
- `POST /api/kernel/{run,save,delete,takeover,scratchpad/run}` with `Marimo-Session-Id` +
  `Marimo-Server-Token`. **camelCase** — `cellIds`, not `cell_ids`, which 400s.

## Gotchas when writing cell code

- Don't `import subprocess` in a cell that another cell already imports — marimo raises
  `multiple-defs`. Just use it; cells share one namespace.
- `pkill -f 'ablate.common.train'` matches the *invoking shell's own* command line. It will kill
  the shell that ran it (returncode -9) and can match your own probe. Use `'[a]blate...'`.
- Don't chain jobs on PIDs: on molab, PID 1 is marimo and never reaps orphans, so an exited job
  stays `<defunct>` and `kill -0 <pid>` succeeds forever. Write a sequential bash script instead.
