#!/bin/bash
# Paired throughput probe: measure step time for N arm configs INTERLEAVED, twice each.
#
# Why interleaved (A B A B) and not A then B: on Aug 15 2026 a molab box drifted ~20% in
# throughput over one afternoon -- SW power capping for the first 21 minutes, then full speed,
# then slow again mid-run with clocks, memory clocks, temperature, ECC and host load all clean.
# Arms trained back-to-back therefore had tps measured in DIFFERENT machine states, and the
# apparent "cost of MoE" was partly the box. Interleaving turns drift into A1 != A2, which is
# visible, instead of A != B, which is a wrong conclusion.
#
#   bash ablate/tools/tps_probe.sh                  # runs immediately
#   WAIT_FOR_SWEEP=1 bash ablate/tools/tps_probe.sh # queue behind a running sweep
#
# val is OFF (--val_every defaults to 0) and --wandb is omitted: this measures step time only.
set -u
PY=${PY:-/tmp/uv-venv/bin/python}
WORK=${WORK:-/home/marimo/work}
STEPS=${STEPS:-150}
# "tag:flagvalue" pairs -- edit for the axis under test
ARMS=${ARMS:-"A:0,9 B:0 C:none"}

if [[ "${WAIT_FOR_SWEEP:-0}" == "1" ]]; then
  # pgrep on a PATTERN, never `kill -0 <pid>`: PID 1 on molab is marimo and never reaps orphans,
  # so an exited job stays <defunct> and kill -0 succeeds forever. A defunct process has an empty
  # cmdline, so pgrep -f cannot match it.
  while pgrep -f '[a]blate.common.train' >/dev/null; do sleep 30; done
  sleep 20
fi

BASE="$PY -u -m ablate.common.train --arm bibo_min --seed 42069 --data real \
--dataset $WORK/data/bibo_mix --experts 64 --top_k 6 --special_pairs 0 \
--act radial --radial_p sigmoid --act_scale_lr 0.01 \
--patches liger_norm,liger_rope,ce,moe,megakernel,xsa --use_xsa --xsa_alpha_init 0.0 \
--swa_pattern block3 --sliding_window 128 --batch 64 --grad_accum 4 --seq_len 1024 \
--precision bf16 --bf16_residual_stream true --muon_lr 0.01 --adam_lr 0.0005 --wd 0.1 \
--cautious_decay false --bias_update_factor 0.4 --bias_update_threshold 2621440 \
--scheduler cosine --max_steps $STEPS --warmup_frac 0.1 --decay_frac 0.2 --norm_topk_prob 1 \
--router_optim muon --vec_matrices_adamw true --attn_res 3 --attn_res_sites 1 \
--attn_res_carry true --attn_res_carry_per_dim true --attn_res_carry_scale raw \
--fused_res_add true --peak_tflops 480 --log_every 10 --dense_inter 4608 \
--out $WORK/probe_runs"

for pass in 1 2; do                      # two passes: drift shows up as pass1 != pass2
  for spec in $ARMS; do
    tag="${spec%%:*}${pass}"; dense="${spec#*:}"
    echo "=== $tag  dense=$dense  $(date +%H:%M:%S)  $(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader)"
    $BASE --mlp_only_layers "$dense" --run_tag "probe-$tag" > "$WORK/probe_$tag.log" 2>&1
  done
done

echo "=== ALL PROBES DONE $(date +%H:%M:%S)"
echo "--- median tps per probe (steps > 50, i.e. past compile + warmup)"
for f in "$WORK"/probe_*.log; do
  med=$(grep -oE 'step ([0-9]+)/.* tps=[0-9.]+k' "$f" \
        | awk '{for(i=1;i<=NF;i++) if($i ~ /^step/) s=$(i+1); if(s+0>50) print}' \
        | grep -oE 'tps=[0-9.]+k' | tr -d 'tps=k' | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int(NR/2)+1]}')
  echo "  $(basename "$f" .log): ${med:-n/a}k"
done
echo "--- power cap counter (must be UNCHANGED across the probe, or the numbers are void)"
nvidia-smi -q | grep 'SW Power Capping'
