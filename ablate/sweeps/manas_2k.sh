#!/usr/bin/env bash
# Manas 3-way at FULL SCALE: 2000 steps / 524M tokens, final bpb+ICL eval. ~3 hours.
#
# !! NOT COMPARABLE TO THIS ROUND'S RECORDED NUMBERS (updated Aug 1 2026) !!
# These arms ran `--act silu` on 64 GLU experts. src deleted every activation except radial
# NormSiLU, so re-running this script trains a DIFFERENT model under the same run names. The
# recorded manas verdict (bpb 0.6812 muon vs 0.6817 manas, a tie at 0.36 sigma) stands as history;
# any new number from this script belongs to a new baseline, not to that comparison.
#
#   mu_anchor   plain Muon -- in-session anchor. The board has 5 base-s replicates at
#               bpb_overall 0.68152-0.68189 (spread 0.00037), but those ran in a different
#               session and the round's standing rule is within-session anchors only.
#   mn_fixed    manas, probe_gamma pinned at the law value 0.23094
#   mn_gs       manas, same gamma tracking the LR schedule
#
# WHY THIS RUN EXISTS. The 400-step 3-way said manas beats muon at every window (-0.080 tail)
# but could NOT separate fixed from gs: the gap (+0.004..+0.010) sat at the same size as the
# 0-100 window where the two arms are near-identical by construction, and train loss is read AT
# THE PROBED THETA -- so mn_fixed, holding 2.4x the standing probe at step 400, is read from
# further downhill exactly where the arms differ. bpb is evaluated at RESTORED theta, which is
# the only measurement that is not biased toward whichever arm probes harder.
#
# Also under test: whether the manas edge SURVIVES. It decayed -0.286 / -0.142 / -0.090 / -0.080
# across the four 400-step windows -- front-loaded, consistent with the round's warmup finding.
# 2000 steps is 5x further down that curve.
#
# --eval_every 0 is FINAL-ONLY (not off; -1 is off) -- matches every baseline on the board.
# --cautious_decay false is explicit here; it is also the default now, and the board is non-cautious.
#
#   bash ablate/sweeps/manas_2k.sh
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
      --peak_tflops 480 --log_every 25 --router_log 1
      --max_steps 2000 --eval_every 0                    # 0 = FINAL-ONLY eval
      --eval_bpb_n 200 --eval_icl_n 50 --eval_mcq_n 200
      --final_mcq_n 500 --final_extrap 1024,2048,4096
      --wandb --wandb_project bibo-manas-2k)

run () {           # name, extra args...
  local name=$1; shift
  local log="$OUT/sweep_$name.log"
  if grep -q "^\[done\]" "$log" 2>/dev/null; then echo "skip $name (done)"; return; fi
  echo "=== $name  $*  $(date -u +%H:%M:%S) ==="
  python -u -m ablate.common.train "${BASE[@]}" "$@" >"$log" 2>&1
  echo "[done] rc=$?" >>"$log"
  # dump VALUES per arm as it lands -- VMs die mid-sweep and a progress counter tells you nothing
  # about the arms that already finished.
  grep -oE "eval/bpb_overall +[0-9.]+" "$log" | tail -1 | sed "s/^/  $name | /"
  grep -oE "step +[0-9]+/2000 .*run20=[0-9.]+" "$log" | tail -1 | sed "s/^/  $name | /"
}

run mu_anchor --optim muon
run mn_fixed  --optim manas
run mn_gs     --optim manas --probe_gamma_schedule lr

echo; echo "=== 2K 3-WAY: bpb_overall (board base-s = 0.68189, 5 replicates, spread 0.00037) ==="
for n in mu_anchor mn_fixed mn_gs; do
  printf "%-10s %s\n" "$n" "$(grep -oE 'eval/bpb_(overall|en|hi) +[0-9.]+' "$OUT/sweep_$n.log" 2>/dev/null | tr '\n' ' ')"
done
