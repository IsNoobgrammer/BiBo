#!/usr/bin/env bash
# Manas 3-way, 400 steps, eval OFF: does annealing the probe dose with the LR matter?
#
#   mu_anchor   plain Muon -- the in-session anchor. NEVER compare across sessions.
#   mn_fixed    manas, probe_gamma pinned at the law value for the whole run
#   mn_gs       manas, same gamma but tracking the LR schedule (held at peak through warmup,
#               then law(lr_t) down the cosine)
#
# THE QUESTION: the probe holds a standing displacement ~gamma/(1-rho_step). Fixed gamma keeps it
# at full size while the cosine anneals lr -> 0, so the endgame is optimised around a point the run
# never lands on. That cost 0.02-0.03 test acc at saturation on the MNIST demo. 400 steps with
# warmup_frac 0.1 = 40 warmup + 360 of anneal, so the arms separate over the back two thirds.
#
# CAVEAT ON READING THIS: manas train loss is logged AT THE PROBED THETA, which is downhill by
# construction, and mn_fixed keeps a bigger standing probe at the end than mn_gs does. So mn_fixed
# is FLATTERED by this metric exactly where the two arms differ. A train-loss tie means mn_gs is
# ahead; only the bpb run settles it. Do not read a small mn_fixed lead as a win.
#
# Everything else is the base-s board config verbatim. --cautious_decay false is explicit: the
# flag defaults TRUE and every baseline on the board is non-cautious.
#
#   bash ablate/sweeps/manas_3way.sh          # ~36 min, 3 arms x 400 steps
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=/tmp/sweeps/manas_3way
mkdir -p "$OUT"

BASE=(--arm bibo_min --seed 42069
      --data real --dataset /home/marimo/work/data/bip2
      --act silu --polyglu_mult 32 --top_k 8
      --batch 64 --grad_accum 4 --seq_len 1024 --precision bf16
      --muon_lr 0.01 --adam_lr 5e-4 --wd 0.1 --cautious_decay false
      --load_balance bias --bias_update_mode prop --bias_update_factor 0.4
      --bias_update_threshold 2621440 --aux_coef 0.001
      --router_type mlp --router_gate sigmoid --router_norm sum --router_optim muon
      --scheduler cosine --warmup_frac 0.1 --decay_frac 0.2 --grad_clip 1
      --patches liger_norm,liger_rope,ce,moe --attn sdpa
      --peak_tflops 480 --log_every 25
      --max_steps 400 --eval_every -1          # -1 = OFF. 0 means FINAL-ONLY, not off.
      --wandb --wandb_project bibo-manas-3way)

run () {           # name, extra args...
  local name=$1; shift
  local log="$OUT/$name.log"
  if grep -q "^\[done\]" "$log" 2>/dev/null; then echo "skip $name (done)"; return; fi
  echo "=== $name  $*  $(date -u +%H:%M:%S) ==="
  python -m ablate.common.train "${BASE[@]}" "$@" >"$log" 2>&1
  echo "[done] rc=$?" >>"$log"
  grep -oE "step +[0-9]+/400 loss=[0-9.]+ run[0-9]+=[0-9.]+" "$log" | tail -1 | sed "s/^/  $name | /"
  grep -oE "tps=[0-9.]+k" "$log" | tail -1 | sed "s/^/  $name | /"
}

run mu_anchor --optim muon
run mn_fixed  --optim manas
run mn_gs     --optim manas --probe_gamma_schedule lr

echo; echo "=== 3-WAY (last 20-step window; compare WITHIN this sweep only) ==="
for n in mu_anchor mn_fixed mn_gs; do
  printf "%-10s %s  %s\n" "$n" \
    "$(grep -oE 'run[0-9]+=[0-9.]+' "$OUT/$n.log" 2>/dev/null | tail -1)" \
    "$(grep -oE 'tps=[0-9.]+k' "$OUT/$n.log" 2>/dev/null | tail -1)"
done
