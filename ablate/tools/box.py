"""Drive a molab (marimo) box from the terminal: run shell/python on it, and list, read,
run, edit, create or delete its notebook cells.

    export MOLAB_URL=https://sb-xxxxxxxxxxxx.sb.molab.run      # or pass --url
    python -m ablate.tools.box ls
    python -m ablate.tools.box 'nvidia-smi'

The random `sb-*.molab.run` URL IS the credential -- the box serves its API with no token, so
anyone holding the URL is root on it. Do not paste it into anything public.

VERBS
    <shell>                     run a shell command (the default verb -- no keyword needed)
    py '<code>'                 run python in the KERNEL's namespace (sees SWEEP, LAUNCH_TRAINING)
    -f local.py                 run a local python file up there
    ls                          list cells: id, name, first line
    cat ID                      print one cell's source
    run ID [ID...]              run cells  (REACTIVE -- see below)
    edit ID file.py [--run]     replace a cell's source and save it to notebook.py
    new NAME file.py [--after ID]   create a cell (appended unless --after)
    rm ID                       delete a cell

TWO EXECUTION PATHS, AND THE DIFFERENCE MATTERS
    Shell/py/-f go to marimo's SCRATCHPAD: it shares the kernel's globals but sits OUTSIDE the
    reactive graph, so a probe cannot cascade into anything. Use it for probes, monitoring and
    launching.
    run/edit/new touch REAL CELLS, which DO cascade -- marimo re-runs every stale descendant, and
    on a freshly started kernel that is *every* downstream cell including the sweep launcher.
    Check the launch gate before touching a cell on a fresh box. Round configuration still belongs
    in a real cell, so the notebook stays a reproducible record of what ran.

THREE TRAPS THIS TOOL ENCODES (each one cost a debugging round)
    1. `run` DOES NOT PERSIST. Overriding `codes` in /api/kernel/run changes the live kernel and
       nothing else; notebook.py on disk keeps the old text and a restart silently reverts you.
       Only /api/kernel/save writes the file -- which is why `edit` exists and why `new`/`rm` go
       through it too (/api/kernel/delete drops the cell from the kernel only; it returns on
       reconnect).
    2. THE KERNEL REPLAYS CACHED OUTPUT ON CONNECT, INCLUDING THE PREVIOUS BOX'S. molab persists
       the notebook with its outputs, so a brand-new box greets you with `>>> dataset ready` and
       `LAUNCHED: ...` before anything has run. This tool drains 3s of replay before issuing any
       request and ignores every cell-op until then. Confirm state from the filesystem or a
       freshly printed value -- never from console text.
    3. A SECOND CONNECTION IS READ-ONLY until it calls /api/kernel/takeover ("This connection is
       read-only for this action.").

Requires `websockets` locally. No dependency on the box side.
"""
import argparse, asyncio, http.cookiejar, json, os, re, sys, urllib.request, uuid

VERBS = ("ls", "cat", "run", "edit", "new", "rm", "py", "-f")
UA = "Mozilla/5.0"          # the proxy 403s python-urllib's default User-Agent
SID = "cli-" + uuid.uuid4().hex[:12]

JAR = http.cookiejar.CookieJar()
urllib.request.install_opener(urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(JAR)))


def get(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read().decode()


def post(base, token, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Marimo-Session-Id": SID,
                 "Marimo-Server-Token": token, "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


def text(o):
    """One marimo output/console payload -> printable text."""
    d = (o or {}).get("data")
    return d if isinstance(d, str) else ("" if d is None else json.dumps(d)[:4000])


def shell_wrap(cmd):
    """Shell verb = python that shells out, since the scratchpad only accepts python."""
    return ("import subprocess as _sp\n"
            "_r = _sp.run(%r, shell=True, capture_output=True, text=True)\n"
            "print(_r.stdout, end=''); print(_r.stderr, end='')\n"
            "print('[exit %%d]' %% _r.returncode) if _r.returncode else None" % cmd)


async def main():
    import websockets
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--url", default=os.environ.get("MOLAB_URL", ""))
    ap.add_argument("-h", "--help", action="store_true")
    opts, a = ap.parse_known_args()
    if opts.help or not a:
        print(__doc__)
        return
    if not opts.url:
        sys.exit("no box URL: set MOLAB_URL or pass --url https://sb-....sb.molab.run")

    base = opts.url.rstrip("/")
    token = re.search(r'server-token data-token="([^"]+)"', get(base)).group(1)
    # The notebook path differs per box; ask instead of hardcoding it.
    fname = json.loads(get(base + "/api/status"))["filenames"][0]

    verb = a[0] if a[0] in VERBS else "sh"
    rest = a[1:] if verb != "sh" else a

    cookie = "; ".join(f"{c.name}={c.value}" for c in JAR)
    async with websockets.connect(base.replace("https", "wss") + f"/ws?session_id={SID}",
                                  additional_headers={"Cookie": cookie},
                                  max_size=None, ping_interval=20, open_timeout=60) as ws:
        armed, targets, done = False, [], set()
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=3600))
            op, data = msg.get("op"), msg.get("data", {})

            if op == "kernel-ready":
                ids = list(data.get("cell_ids") or [])
                codes = list(data.get("codes") or [])
                names = list(data.get("names") or [])
                cfgs = list(data.get("configs") or [{}] * len(ids))

                if verb == "ls":
                    for i, c in enumerate(ids):
                        head = next((l for l in codes[i].splitlines() if l.strip()), "")
                        print(f"{c:9} {names[i]:18} {head[:66]}")
                    return
                if verb == "cat":
                    print(codes[ids.index(rest[0])])
                    return

                def save():
                    """Write the WHOLE cell list back to the notebook -- marimo exposes no
                    per-cell persistence, so edit/new/rm all land here."""
                    print("save:", post(base, token, "/api/kernel/save", {
                        "cellIds": ids, "codes": codes, "names": names,
                        "configs": cfgs, "filename": fname})[0], flush=True)

                async def go():
                    nonlocal armed, targets
                    await asyncio.sleep(3)             # drain the cached-output replay (trap 2)
                    post(base, token, "/api/kernel/takeover", {})            # trap 3

                    if verb in ("sh", "py", "-f"):
                        code = (shell_wrap(rest[0]) if verb == "sh" else
                                rest[0] if verb == "py" else
                                open(rest[0], encoding="utf-8").read())
                        s, b = post(base, token, "/api/kernel/scratchpad/run", {"code": code})
                    elif verb == "run":
                        targets = [c for c in ids if c in rest]
                        s, b = post(base, token, "/api/kernel/run",
                                    {"cellIds": targets,
                                     "codes": [codes[ids.index(c)] for c in targets]})
                    elif verb == "edit":
                        cid = rest[0]
                        codes[ids.index(cid)] = open(rest[1], encoding="utf-8").read()
                        save()                          # persist FIRST (trap 1)
                        if "--run" not in rest:
                            os._exit(0)
                        targets = [cid]
                        s, b = post(base, token, "/api/kernel/run",
                                    {"cellIds": [cid], "codes": [codes[ids.index(cid)]]})
                    elif verb == "new":
                        at = (ids.index(rest[rest.index("--after") + 1]) + 1
                              if "--after" in rest else len(ids))
                        cid = "cli" + uuid.uuid4().hex[:5]
                        ids.insert(at, cid)
                        names.insert(at, rest[0])
                        codes.insert(at, open(rest[1], encoding="utf-8").read())
                        cfgs.insert(at, {"disabled": False, "hide_code": False})
                        save()
                        print("new cell:", cid)
                        os._exit(0)
                    elif verb == "rm":
                        i = ids.index(rest[0])
                        post(base, token, "/api/kernel/delete", {"cellId": rest[0]})
                        for lst in (ids, codes, names, cfgs):
                            lst.pop(i)
                        save()                          # kernel-only delete reverts on reconnect
                        os._exit(0)

                    if s != 200:
                        print("FAILED", s, b, file=sys.stderr)
                        os._exit(1)
                    armed = True
                asyncio.create_task(go())

            elif op == "cell-op" and armed:
                con = data.get("console")
                for c in (con if isinstance(con, list) else [con] if con else []):
                    print(text(c), end="", flush=True)
                o = data.get("output")
                if o and o.get("channel") == "marimo-error":
                    print("ERROR:", text(o), flush=True)
                elif o and o.get("data"):
                    print(text(o)[:4000], flush=True)
                # The scratchpad is one cell, so its idle is the end. `run` may span several
                # cells plus whatever marimo cascades into, so wait for every requested one.
                if data.get("status") == "idle":
                    done.add(data.get("cell_id"))
                    if set(targets) <= done:
                        return
            elif op in ("alert", "banner"):
                print("ALERT:", json.dumps(data)[:300], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
