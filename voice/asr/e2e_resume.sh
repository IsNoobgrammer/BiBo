#!/bin/bash
# End-to-end resume check (the REAL train_asr loop, not just the optimizer): --deterministic so same-seed runs are
# bitwise repeatable. A: 0 -> 200 straight. B: 0 -> 100, stop, resume from its step-100 checkpoint -> 200.
# Pass = A's and B's step-200 checkpoints are bitwise equal (weights AND optimizer state); A/B step 100 must match too.
#   bash voice/asr/e2e_resume.sh      -> last line E2E_RESUME_EXACT or E2E_RESUME_MISMATCH
set -uo pipefail
A=/home/marimo/work/asr; E3=$A/en3; P=/home/marimo/asrenv/bin/python; O=$A/exp_e2e
cd /home/marimo/work/BiBo
COMMON="--project bibo-asr-test --no_hf --deterministic --seed 23 --new_vocab --train $E3/train.jsonl
  --tok $E3/tok/tokenizer_spe_bpe_v2047 --val $E3/val_phone.jsonl --lr 5e-4 --total_hours 3750 --eval_hours 100000
  --ctc_head glu_glu --ctc_weight 0.6 --fastemit 0.01 --batch_sec 1200
  --lookaheads 13 6 3 1 0 -1 --lookahead_probs 0:0.05 1:0.15 3:0.35 6:0.10 13:0.15 full:0.20
  --fused_joint --fused_layer --fused_attn --fused_conv --fused_ctc --out $O"
T="env TKF=/home/marimo/work/tkf_ctc $P voice/asr/train_asr.py"
$T $COMMON --run e2eA --stop_step 200 --save_steps 100 200 > $A/e2eA.out 2>&1; echo "A rc $?"
$T $COMMON --run e2eB --stop_step 100 --save_steps 100 > $A/e2eB1.out 2>&1; echo "B1 rc $?"
$T $COMMON --run e2eB --stop_step 200 --save_steps 200 --ckpt $O/e2eB/step100.ckpt > $A/e2eB2.out 2>&1; echo "B2 rc $?"
grep -h "RNG states restored\|legacy\|resume:" $A/e2eB2.out
$P - <<EOF
import torch
def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)
def diff(a, b, path=""):
    if torch.is_tensor(a):
        return [] if (torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)) else [path]
    if isinstance(a, dict):
        return [x for k in set(a) | set(b) for x in diff(a.get(k), b.get(k), f"{path}.{k}")] if isinstance(b, dict) else [path]
    if isinstance(a, (list, tuple)):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")] if len(a) == len(b) else [path]
    return [] if a == b else [path]
ok = True
for s in (100, 200):
    a, b = load(f"$O/e2eA/step{s}.ckpt"), load(f"$O/e2eB/step{s}.ckpt")
    w = diff(a["state_dict"], b["state_dict"], "w")
    o = diff(a["optimizer_states"], b["optimizer_states"], "opt")
    print(f"step {s}: weights differing {len(w)} {w[:4]} | optimizer state differing {len(o)} {o[:4]}", flush=True)
    ok &= not w and not o
print("E2E_RESUME_EXACT" if ok else "E2E_RESUME_MISMATCH", flush=True)
EOF
