#!/usr/bin/env bash
# Manas dose ladder — 300-step tail train loss, in-session muon anchor.
#
# !! NOT COMPARABLE TO THIS ROUND'S RECORDED NUMBERS (updated Aug 1 2026) !!
# These arms ran `--act silu` on 64 GLU experts. src deleted every activation except radial
# NormSiLU, so re-running this script trains a DIFFERENT model under the same run names. The
# recorded manas verdict (bpb 0.6812 muon vs 0.6817 manas, a tie at 0.36 sigma) stands as history;
# any new number from this script belongs to a new baseline, not to that comparison.
#
# WHY A LADDER AND NOT A SINGLE RUN: the auto-gamma law is
#   gamma = 0.08 * sqrt(lr/3e-4) * k / sqrt(m)
# and the board runs at muon_lr 1e-2, so it prescribes 0.2309 at k=4 / m=64. The sqrt-LR term
# was only ever calibrated at 3e-4 and 3e-3 -- 1e-2 is a 1.8x extrapolation past its highest
# validated point. Under-dose is benign (every dose error ever measured was on that side);
# k-extrapolated overdose is not. So bracket the law at half and double before spending an hour.
#
# EVERY ARM IS THE base-s BOARD CONFIG VERBATIM (pulled from the W&B run config) except --optim
# and --probe_gamma. --cautious_decay false is MANDATORY and explicit: the flag defaults to TRUE
# and every baseline on the board is non-cautious.
#
# Separate W&B project on purpose: the 300-step muon anchor would otherwise collide with the
# 2000-step base-s run name (same arm/seed/acts) and overwrite its ckpt + log names.
#
# Arms run SEQUENTIALLY -- contention breaks tps (not bpb), and tps is one of the answers here.
#
#   bash ablate/sweeps/manas_ladder.sh          # ~34 min, 4 arms x 300 steps
set -u
cd "$(dirname "$0")/../.." || exit 1

# Logs go where the notebook TRAINING MONITOR globs (/home/marimo/work/*.log, arm name =
# basename[6:-4], so the "sweep_" prefix is load-bearing).
OUT=/home/marimo/work
mkdir -p "$OUT"

BASE=(--arm bibo_min --seed 42069
      --data real --dataset /home/marimo/work/data/bip2
      --experts 64 --top_k 8
      --batch 64 --grad_accum 4 --seq_len 1024 --precision bf16
      --muon_lr 0.01 --adam_lr 5e-4 --wd 0.1 --cautious_decay false
      --bias_update_factor 0.4
      --bias_update_threshold 2621440 --aux_coef 0.001
      --router_optim muon
      --scheduler cosine --warmup_frac 0.1 --decay_frac 0.2 --grad_clip 1
      --patches liger_norm,liger_rope,ce,moe --attn sdpa
      --peak_tflops 480 --log_every 25
      --max_steps 300 --eval_every -1          # -1 = OFF. 0 means FINAL-ONLY, not off.
      --wandb --wandb_project bibo-manas-ladder)

# name            optim   gamma      note
ARMS=(
  "mu_anchor      muon    0          in-session anchor -- never compare across sessions"
  "mn_g0.115      manas   0.115      half the law"
  "mn_g0.231      manas   0.2309     the law at lr 1e-2, k=4, m=64"
  "mn_g0.462      manas   0.4618     double -- looking for the overdose turnover"
)

for a in "${ARMS[@]}"; do
  set -- $a; name=$1; optim=$2; gamma=$3
  log="$OUT/sweep_$name.log"
  if grep -q "^\[done\]" "$log" 2>/dev/null; then echo "skip $name (done)"; continue; fi
  echo "=== $name  optim=$optim gamma=$gamma  $(date -u +%H:%M:%S) ==="
  extra=(--optim "$optim")
  [ "$optim" = manas ] && extra+=(--probe_gamma "$gamma")
  python -m ablate.common.train "${BASE[@]}" "${extra[@]}" >"$log" 2>&1
  rc=$?
  echo "[done] rc=$rc" >>"$log"
  # dump the VALUES now, not just a count -- VMs die mid-sweep and a bare progress counter
  # tells you nothing about the arms that already finished.
  tail -3 "$log" | sed "s/^/  $name | /"
  grep -oE "tps [0-9.]+k" "$log" | tail -1 | sed "s/^/  $name | /"
done

echo; echo "=== LADDER TAIL (last logged window per arm) ==="
for a in "${ARMS[@]}"; do
  set -- $a; name=$1
  printf "%-12s %s\n" "$name" "$(grep -oE "loss [0-9.]+ \(run [0-9.]+\)" "$OUT/sweep_$name.log" 2>/dev/null | tail -1)"
done
echo "Read the run-mean (the 20-step window), not the point loss. Compare WITHIN this sweep only."
