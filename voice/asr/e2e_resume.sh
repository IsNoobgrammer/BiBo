#!/bin/bash
# Resume reproducibility check in the REAL train_asr loop (user spec, Oct 11 2026).
#   Schedule of a 1000-step run (warm-up 5% = 50, cosine to step 1000); --deterministic so it is bitwise repeatable.
#   A: 0 -> 500, then stops (as if the box died), checkpoints at 300 and 500 saved locally, per-step trace.
#   B: resume from A's step-300 checkpoint -> 500, per-step trace, checkpoint at 500.
#   Pass = B's steps 301..500 equal A's exactly (loss, grad norm, lr, look-ahead) AND the step-500 checkpoints are
#   bitwise equal (weights + optimizer state + RNG states).
#   bash voice/asr/e2e_resume.sh      -> last line REPRO_EXACT or REPRO_MISMATCH
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; O=$A/exp_repro
mkdir -p $O && cd /home/marimo/work/BiBo
for f in $O/traceA.jsonl $O/traceB.jsonl; do : > $f; done      # empty the traces (they append)
COMMON="--project bibo-asr-test --no_hf --deterministic --seed 23 --new_vocab --train $E3/train.jsonl
  --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $E3/val_phone.jsonl --lr 5e-4 --total_hours 333.33334 --eval_hours 100000
  --ctc_head glu_glu --ctc_weight 0.6 --fastemit 0.01 --batch_sec 1200
  --lookaheads 13 6 3 1 0 -1 --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20
  --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $O"
T="env TKF=/home/marimo/work/tkf_ctc $P voice/asr/train_asr.py"
$T $COMMON --run reproA --stop_step 500 --save_steps 300 500 --trace $O/traceA.jsonl > $A/reproA.out 2>&1; echo "A rc $?"
grep -h "max_steps\|warm-up" $A/reproA.out | cut -c1-120
$T $COMMON --run reproB --stop_step 500 --save_steps 500 --ckpt $O/reproA/step300.ckpt --trace $O/traceB.jsonl \
  > $A/reproB.out 2>&1; echo "B rc $?"
grep -h "RNG states restored\|legacy\|resume:" $A/reproB.out
$P - <<EOF
import json, torch
ta = {r["step"]: r for r in map(json.loads, open("$O/traceA.jsonl"))}
tb = {r["step"]: r for r in map(json.loads, open("$O/traceB.jsonl"))}
steps = sorted(s for s in tb if 300 < s <= 500)
bad = [s for s in steps if ta.get(s) != tb[s]]
print(f"trace: B has {len(steps)} steps in 301..500 (A has {sum(1 for s in ta if 300 < s <= 500)}); differing: {len(bad)}", flush=True)
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
ca = torch.load("$O/reproA/step500.ckpt", map_location="cpu", weights_only=False)
cb = torch.load("$O/reproB/step500.ckpt", map_location="cpu", weights_only=False)
w = diff(ca["state_dict"], cb["state_dict"], "w")
o = diff(ca["optimizer_states"], cb["optimizer_states"], "opt")
print(f"step-500 checkpoint: weights differing {len(w)} {w[:4]} | optimizer state differing {len(o)} {o[:4]}", flush=True)
ok = len(steps) == 200 and not bad and not w and not o
print("REPRO_EXACT" if ok else "REPRO_MISMATCH", flush=True)
EOF
