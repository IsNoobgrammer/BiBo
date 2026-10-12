#!/bin/bash
# Overnight queue (Oct 12 2026): abl.sh drop02, then ep5-lr25, then ep5-w10 (started on the fork box; resumes from HF if
# that box died). drop05 stopped at step 4500: CTC-bp val 39.03 vs 24.95-25.16 for every other arm. Everything resumes from HF after a box death: re-run this script.
#   nohup bash voice/asr/night.sh > /home/marimo/work/asr/night.out 2>&1 &
A=/home/marimo/work/asr
until [ -f $A/.markers/ready ]; do sleep 30; done
cd /home/marimo/work/BiBo && git pull -q origin main; git -C /home/marimo/work/tkf_ctc pull -q origin master
echo "$(date +%H:%M) BiBo $(git log --oneline -1 | cut -c1-50)"
ARMS="drop02" bash voice/asr/abl.sh >> $A/abl_night.out 2>&1; echo "$(date +%H:%M) ABL rc $?"
RUN=ep5-lr25 WU=0.05 LR=2.5e-4 bash voice/asr/long5.sh >> $A/long5_lr25.out 2>&1; echo "$(date +%H:%M) EP5LR25 rc $?"
RUN=ep5-w10 WU=0.10 LR=5e-4 bash voice/asr/long5.sh >> $A/long5_w10.out 2>&1; echo "$(date +%H:%M) EP5W10 rc $?"
echo NIGHT_DONE
