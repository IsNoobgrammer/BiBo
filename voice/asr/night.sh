#!/bin/bash
# Overnight queue (Oct 12 2026), one box: abl.sh arms drop02 then drop05 (drop05 started on the fork box; a re-run
# resumes it from HF), then the two 5-epoch runs. Everything resumes from HF after a box death: re-run this script.
#   nohup bash voice/asr/night.sh > /home/marimo/work/asr/night.out 2>&1 &
A=/home/marimo/work/asr
until [ -f $A/.markers/ready ]; do sleep 30; done
cd /home/marimo/work/BiBo && git pull -q origin main; git -C /home/marimo/work/tkf_ctc pull -q origin master
echo "$(date +%H:%M) BiBo $(git log --oneline -1 | cut -c1-50)"
ARMS="drop02 drop05" bash voice/asr/abl.sh >> $A/abl_night.out 2>&1; echo "$(date +%H:%M) ABL rc $?"
RUN=ep5-w10 WU=0.10 LR=5e-4 bash voice/asr/long5.sh >> $A/long5_w10.out 2>&1; echo "$(date +%H:%M) EP5W10 rc $?"
RUN=ep5-lr25 WU=0.05 LR=2.5e-4 bash voice/asr/long5.sh >> $A/long5_lr25.out 2>&1; echo "$(date +%H:%M) EP5LR25 rc $?"
echo NIGHT_DONE
