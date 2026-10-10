"""Live progress dashboard for the voice data pipeline on the molab box (localhost, stdlib only).

    python voice/asr/dash.py --box https://sb-<id>.sb.molab.run [--port 8765]   -> open http://localhost:8765

A poller thread asks the box every --every seconds (one short, read-only Python call through the notebook's execute API:
reads the lane logs / status files / manifests, nvidia-smi, df) and caches the JSON; the page refreshes itself.
Read-only: it never starts, stops or edits anything on the box. The box URL is the credential -- keep it local.
"""
import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROBE = r'''
import json, os, re, glob, subprocess, sys
A = "/home/marimo/work/asr"; L = A + "/par_en"; M = A + "/mix_en"
sys.path.insert(0, "/home/marimo/work/BiBo/voice/asr")
try:   # budgets straight from the source text (importing build_mix needs torchaudio, absent in the kernel env)
    src = open("/home/marimo/work/BiBo/voice/asr/build_mix.py", encoding="utf-8").read()
    budget = {n: (None if h == "None" else float(h))
              for n, h in re.findall(r'dict\(name="(\w+)".*?hours=(None|[\d.]+)', src, re.S)}
except Exception:
    budget = {}
def lane_srcs(name):          # the lane lists straight from par_en.sh
    try:
        t = open("/home/marimo/work/BiBo/voice/asr/par_en.sh").read()
        return re.search(f'LANE_{name}="([^"]+)"', t).group(1).split()
    except Exception:
        return []
LANES = {"A": lane_srcs("A"), "B": lane_srcs("B")}
if os.path.exists(f"{L}/C.srcs"):              # extra lanes started by hand (e.g. the AMI <1 s rebuild)
    LANES["C"] = open(f"{L}/C.srcs").read().split()
pat = re.compile(r"^(\w+):\s+([\d.]+) / (\S+) h\s+(\d+) utts\s+(\d+) skipped(?:\s+(\d+) dropped)?(.*)$")
def tail(p, n=200000):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); s = f.tell(); f.seek(max(0, s - n)); return f.read().decode("utf-8", "replace")
    except Exception:
        return ""
def status(name):
    try: return open(f"{L}/{name}.status").read().strip()
    except Exception: return None
out = {"time": subprocess.run(["date", "+%H:%M:%S"], capture_output=True, text=True).stdout.strip(), "lanes": {}}
for lane, srcs in LANES.items():
    log = tail(f"{L}/{lane}.log").replace("\r", "\n")
    last = {}
    for line in log.splitlines():
        m = pat.match(line.strip())
        if m:
            last[m.group(1)] = dict(h=float(m.group(2)), utts=int(m.group(4)), skipped=int(m.group(5)),
                                    dropped=int(m.group(6) or 0), extra=m.group(7).strip())
    err = [l[:200] for l in log.splitlines() if "SOURCE FAILED" in l or "Traceback" in l][-2:]
    rows = []
    for s in srcs:
        d = last.get(s, {})
        if s == "restore_old":
            r = re.findall(r"fetch_mix: .*", log)
            d = {"h": 0, "utts": 0, "skipped": 0, "dropped": 0, "extra": (r[-1][:120] if r else "restoring fhai50032/asr-english")}
        rows.append(dict(source=s, budget=budget.get(s), **({"h": 0, "utts": 0, "skipped": 0, "dropped": 0, "extra": ""} | d),
                         finished=os.path.exists(f"{L}/{s}.built"),
                         push=("pushed" if os.path.exists(f"{L}/{s}.pushed") else "push failed" if os.path.exists(f"{L}/{s}.pushfail")
                               else "pushing" if os.path.exists(f"{L}/{s}.built") else "")))
    out["lanes"][lane] = dict(status=status(lane), sources=rows, errors=err)
pl = tail(f"{L}/push.log", 40000).replace(chr(13), chr(10))
lines = [l.strip() for l in pl.splitlines() if l.strip()]
cur = [l for l in lines if l.startswith("PUSH_START")]
pct = re.findall(r"(\d+)%\|", pl)
out["push"] = dict(status=status("push"), current=(cur[-1][11:] if cur else None),
                   pushed=[l for l in lines if l.startswith("PUSHED")][-20:], failed=[l for l in lines if "FAILED" in l][-5:],
                   last=(lines[-1][:160] if lines else ""), pct=(int(pct[-1]) if pct else None))
try:
    g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().split(", ")
    out["gpu"] = dict(util=int(g[0]), mem=int(g[1]), total=int(g[2]))
except Exception: out["gpu"] = None
try:
    du = subprocess.run(["du", "-sh", M], capture_output=True, text=True, timeout=20).stdout.split()[0]
except Exception: du = "?"
out["disk"] = du
print("DASHJSON" + json.dumps(out))
'''


class Box:
    def __init__(self, url, every):
        self.url, self.every = url.rstrip("/"), every
        self.data, self.err, self.at = None, None, 0.0

    def call(self):
        ua = {"User-Agent": "curl/8.5.0"}                         # the box proxy 403s Python-urllib's default agent
        sid = list(json.load(urllib.request.urlopen(urllib.request.Request(f"{self.url}/api/sessions", headers=ua),
                                                    timeout=20)))[0]
        req = urllib.request.Request(f"{self.url}/api/kernel/execute", data=json.dumps({"code": PROBE}).encode(),
                                     headers={"Content-Type": "application/json", "Marimo-Session-Id": sid, **ua})
        text = ""
        with urllib.request.urlopen(req, timeout=90) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    try:
                        text += json.loads(line[5:]).get("data", "") or ""
                    except Exception:
                        pass
        i = text.find("DASHJSON")
        if i < 0:
            raise RuntimeError("no data from box: " + text[-200:])
        return json.loads(text[i + 8:].splitlines()[0])

    def loop(self):
        while True:
            try:
                self.data, self.err, self.at = self.call(), None, time.time()
            except Exception as e:
                self.err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(self.every)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>English Data Build</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--fg:#1d2330;--muted:#677084;--rule:#e1e4ea;--bar:#e7eaf0;--fill:#2f6fde;--done:#1f9d55;--warn:#c47a10;--bad:#c4372f}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1a1f27;--fg:#e8ebf1;--muted:#9aa3b5;--rule:#2b323d;--bar:#262d38;--fill:#5b93f0;--done:#3cc078;--warn:#e0a03a;--bad:#ef6a62}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 18px 40px}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:baseline;justify-content:space-between}
h1{font-size:22px;margin:0}.meta{color:var(--muted);font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;min-width:0}
.card h2{font-size:15px;margin:0 0 10px;display:flex;justify-content:space-between;align-items:center}
.pill{font-size:12px;padding:2px 9px;border-radius:99px;border:1px solid currentColor;font-weight:600}
.run{color:var(--fill)}.ok{color:var(--done)}.bad{color:var(--bad)}.wait{color:var(--muted)}
.row{display:grid;grid-template-columns:120px 1fr;gap:4px 10px;align-items:center;margin:7px 0}
.name{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{height:9px;background:var(--bar);border-radius:5px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--fill)}
.bar>i.full{background:var(--done)}
.sub{grid-column:2;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.err{color:var(--bad);font-size:12px;margin-top:6px;word-break:break-word}
.alert{margin-top:14px;padding:10px 14px;border-radius:8px;border:1px solid var(--bad);color:var(--bad);display:none}
.kv{display:flex;flex-wrap:wrap;gap:6px 18px;font-variant-numeric:tabular-nums}.kv b{font-weight:600}
code{font-size:12px;color:var(--muted)}
</style></head><body><div class="wrap">
<header><h1>English data build &middot; ~2,000 h</h1><div class="meta" id="meta">connecting...</div></header>
<div class="alert" id="alert"></div>
<div class="grid" id="top"></div>
<div class="grid" id="lanes"></div>
</div>
<script>
const LANE_NAMES={A:"Lane A (restore old English, AMI-sdm, LibriSpeech, Earnings-22, NOTSOFAR, VITW x4)",B:"Lane B (SPGISpeech 2.0, TED-LIUM, VITW x4)",C:"Lane C (AMI rebuild, clips down to 0.2 s)"};
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function pill(st){if(st===null||st===undefined)return '<span class="pill run">running</span>';return st==="0"?'<span class="pill ok">done</span>':'<span class="pill bad">failed ('+esc(st)+')</span>';}
const HIST={};let LAST=null,AT=0;
function rate(s){if(!s.h&&!s.finished)return ' &middot; queued / preparing (download, unpack)';const h=HIST[s.source]||[];const [t0,h0]=h[0]||[],[t1,h1]=h[h.length-1]||[];if(!(t1-t0>=60000))return ' &middot; rate in ~1 min';const r=(h1-h0)/((t1-t0)/60000);
 if(!(r>0))return s.finished?'':' &middot; <b>stalled</b>';const eta=s.budget&&!s.finished?(s.budget-s.h)/r:null;
 return ' &middot; <b>'+(r*60).toFixed(1)+' h/hr</b>'+(eta!==null?' &middot; ETA '+(eta>=60?(eta/60).toFixed(1)+' h':Math.round(eta)+' min'):'');}
function srcRow(s){const b=s.budget;const pct=b?Math.min(100,100*s.h/b):(s.finished?100:null);
 const bar=pct===null?'<div class="bar"><i style="width:100%;opacity:.35"></i></div>':'<div class="bar"><i class="'+(pct>=99.5||s.finished?'full':'')+'" style="width:'+pct.toFixed(1)+'%"></i></div>';
 return '<div class="row"><div class="name" title="'+esc(s.source)+'">'+esc(s.source)+'</div>'+bar+
 '<div class="sub">'+s.h.toFixed(1)+' / '+(b?b+' h':'all')+' &middot; '+s.utts.toLocaleString()+' rows &middot; '+s.dropped.toLocaleString()+' dropped &middot; '+s.skipped.toLocaleString()+' skipped '+(s.extra?'&middot; '+esc(s.extra):'')+(s.push?' &middot; <b>'+esc(s.push)+'</b>':'')+rate(s)+'</div></div>';}
async function tick(){let d;try{d=await (await fetch('/api')).json();}catch(e){document.getElementById('meta').textContent='dashboard server not reachable';return;}
 const al=document.getElementById('alert');
 if(d.err){al.style.display='block';al.textContent='Box not answering: '+d.err+(d.age?' (last good data '+Math.round(d.age)+' s ago)':'');}else al.style.display='none';
 const b=d.data;if(!b){document.getElementById('meta').textContent='waiting for first data...';return;}
 AT=Date.now()-(d.age||0)*1000;if(b.time!==LAST){LAST=b.time;for(const l of Object.values(b.lanes))for(const s of l.sources){const h=HIST[s.source]=HIST[s.source]||[];h.push([AT,s.h]);while(h.length>1&&AT-h[0][0]>600000)h.shift();}}
 document.getElementById('meta').innerHTML='box time '+b.time+' · updated <span id=age></span> s ago · output '+b.disk+(b.gpu?' · GPU '+b.gpu.util+'% / '+(b.gpu.mem/1024).toFixed(1)+' GB':'');
 const p=b.push;const pst=p.status==="0"?'<span class="pill ok">all pushed</span>':(p.current?'<span class="pill run">uploading</span>':'<span class="pill wait">waiting for a built source</span>');
 let top='<div class="card"><h2>Push to HF (each source as soon as it is built)'+pst+'</h2>'+
  (p.current?'<div class="sub">now: <b>'+esc(p.current)+'</b>'+(p.pct!==null?' &middot; '+p.pct+'%':'')+'</div>':'')+
  (p.pushed.length?'<div class="sub" style="margin-top:6px">'+p.pushed.map(esc).join('<br>')+'</div>':'')+
  (p.failed.length?'<div class="err">'+p.failed.map(esc).join('<br>')+'</div>':'')+
  (p.last?'<div class="sub" style="margin-top:6px"><code>'+esc(p.last)+'</code></div>':'')+'</div>';
 document.getElementById('top').innerHTML=top;
 let html='';for(const [lane,l] of Object.entries(b.lanes)){const tot=l.sources.reduce((a,s)=>a+s.h,0);
  html+='<div class="card"><h2>'+esc(LANE_NAMES[lane]||lane)+' <span>'+'<span class="meta" style="margin-right:8px">'+tot.toFixed(1)+' h</span>'+pill(l.status)+'</span></h2>'+l.sources.map(srcRow).join('')+(l.errors.length?'<div class="err">'+l.errors.map(esc).join('<br>')+'</div>':'')+'</div>';}
 document.getElementById('lanes').innerHTML=html;}
tick();setInterval(tick,5000);setInterval(()=>{const e=document.getElementById('age');if(e&&AT)e.textContent=Math.round((Date.now()-AT)/1000);},1000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    box = Box(a.box, a.every)
    threading.Thread(target=box.loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_GET(self):
            if self.path.startswith("/api"):
                body = json.dumps({"data": box.data, "err": box.err, "age": time.time() - box.at if box.at else None}).encode()
                ctype = "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    print(f"dashboard on http://localhost:{a.port}  (polling {a.box} every {a.every}s)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
