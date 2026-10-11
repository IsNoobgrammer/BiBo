#!/bin/bash
# Resume reproducibility check in the REAL train_asr loop (user spec, Oct 11 2026).
#   Schedule of a 1000-step run (warm-up 5% = 50, cosine to step 1000); --deterministic so it is bitwise repeatable.
#   A: 0 -> 500, then stops (as if the box died), checkpoints at 300 and 500 saved locally, per-step trace.
#   B: resume from A's step-300 checkpoint -> 500, per-step trace, checkpoint at 500.
#   Pass = B's steps 301..500 equal A's exactly (loss, grad norm, lr, look-ahead) AND the step-500 checkpoints are
#   bitwise equal (weights + optimizer state + RNG states).
#   bash voice/asr/e2e_resume.sh      -> last line REPRO_EXACT or REPRO_MISMATCH
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; O=$A/exp_repro_$(date +%Y%m%d_%H%M%S)   # fresh dir: a leftover last.ckpt would be RESUMED
STOP=${STOP:-500}; MID=${MID:-300}   # STOP=310 MID=300: a quick diagnosis run
mkdir -p $O && cd /home/marimo/work/BiBo
for f in $O/traceA.jsonl $O/traceA2.jsonl $O/traceB.jsonl; do : > $f; done      # empty the traces (they append)
COMMON="--project bibo-asr-test --no_hf --deterministic --seed 23 --new_vocab --train $E3/train.jsonl
  --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $E3/val_phone.jsonl --lr 5e-4 --total_hours 333.33334 --eval_hours 100000
  --ctc_head glu_glu --ctc_weight 0.6 --fastemit 0.01 --batch_sec 1200
  --lookaheads 13 6 3 1 0 -1 --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20
  --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $O"
T="env TKF=/home/marimo/work/tkf_ctc TRACE_MODULES=$((MID + 1)) $P voice/asr/train_asr.py"
$T $COMMON --run reproA --stop_step $STOP --save_steps $MID $STOP --trace $O/traceA.jsonl > $A/reproA.out 2>&1; echo "A rc $?"
grep -h "max_steps\|warm-up" $A/reproA.out | cut -c1-120
# control: a second fresh run must repeat A bit for bit (else the deterministic mode itself is not deterministic)
$T $COMMON --run reproA2 --stop_step $((MID + 20 < STOP ? MID + 20 : STOP)) --trace $O/traceA2.jsonl > $A/reproA2.out 2>&1; echo "A2 rc $?"
$T $COMMON --run reproB --stop_step $STOP --save_steps $STOP --ckpt $O/reproA/step$MID.ckpt --trace $O/traceB.jsonl \
  > $A/reproB.out 2>&1; echo "B rc $?"
grep -h "RNG states restored\|legacy\|resume:" $A/reproB.out
$P - <<EOF
import json, torch
ta = {r["step"]: r for r in map(json.loads, open("$O/traceA.jsonl"))}
tb = {r["step"]: r for r in map(json.loads, open("$O/traceB.jsonl"))}
t2 = {r["step"]: r for r in map(json.loads, open("$O/traceA2.jsonl"))}
cbad = [s for s in sorted(t2) if ta.get(s) != t2[s]]
print(f"control (fresh A2 vs A, steps 1..{max(t2)}): differing {len(cbad)}" + (f", first at step {cbad[0]}: A {ta.get(cbad[0])} | A2 {t2[cbad[0]]}" if cbad else ""), flush=True)
steps = sorted(s for s in tb if $MID < s <= $STOP)
bad = [s for s in steps if ta.get(s) != tb[s]]
print(f"trace: B has {len(steps)} steps after $MID (A has {sum(1 for s in ta if $MID < s <= $STOP)}); differing: {len(bad)}", flush=True)
for s in steps[:3]:
    fa, fb = (ta.get(s) or {}).get("fp") or {}, tb[s].get("fp") or {}
    d = {k: "same" for k in set(fa) | set(fb) if fa.get(k) == fb.get(k)}
    for k in set(fa) | set(fb):
        if fa.get(k) != fb.get(k):
            if isinstance(fa.get(k), dict):
                ks = set(fa.get(k) or {}) | set(fb.get(k) or {})
                d[k] = "differ: " + str([x for x in ks if (fa.get(k) or {}).get(x) != (fb.get(k) or {}).get(x)][:8])
            else:
                d[k] = f"A {fa.get(k)} B {fb.get(k)}"
    print("  fingerprint step", s, d, flush=True)
    ma, mb = (ta.get(s) or {}).get("mods") or [], tb[s].get("mods") or []
    if ma or mb:
        first = next(((x, y) for x, y in zip(ma, mb) if x[1:] != y[1:]), None)
        print(f"  modules step {s}: {len(ma)} vs {len(mb)} calls; first differing call:", first, flush=True)
        for x, y in list(zip(ma, mb))[:40]:
            print("     ", x[0], "out", "same" if x[1] == y[1] else "DIFF", "| rng after", "same" if x[2] == y[2] else "DIFF", flush=True)
for s in bad[:5]:
    print("  step", s, "A", ta.get(s), "| B", tb[s], flush=True)
def diff(a, b, path=""):
    if torch.is_tensor(a):
        return [] if (torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)) else [path]
    if isinstance(a, dict):
        return [x for k in set(a) | set(b) for x in diff(a.get(k), b.get(k), f"{path}.{k}")] if isinstance(b, dict) else [path]
    if isinstance(a, (list, tuple)):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")] if len(a) == len(b) else [path]
    return [] if a == b else [path]
ca = torch.load("$O/reproA/step$STOP.ckpt", map_location="cpu", weights_only=False)
cb = torch.load("$O/reproB/step$STOP.ckpt", map_location="cpu", weights_only=False)
w = diff(ca["state_dict"], cb["state_dict"], "w")
o = diff(ca["optimizer_states"], cb["optimizer_states"], "opt")
print(f"step-$STOP checkpoint: weights differing {len(w)} {w[:4]} | optimizer state differing {len(o)} {o[:4]}", flush=True)
ok = len(steps) == $STOP - $MID and not bad and not w and not o and not cbad
print("REPRO_EXACT" if ok else "REPRO_MISMATCH", flush=True)
EOF
