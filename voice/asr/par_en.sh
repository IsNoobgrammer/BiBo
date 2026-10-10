#!/bin/bash
# English data build on a fresh box: two download lanes in parallel + a push lane that uploads each source to HF the
# moment it is built (a box death loses only the sources in flight). Logs / markers in $L (voice/asr/dash.py reads them).
#   bash voice/asr/par_en.sh        (re-runnable: built + pushed sources are skipped)
A=/home/marimo/work/asr; M=$A/mix_en; L=$A/par_en; P=/home/marimo/asrenv/bin/python; B=/home/marimo/work/BiBo/voice/asr
OPEN=fhai50032/asr-english-v2       # CC-BY / CC-BY-SA / Apache sources
NC=fhai50032/asr-english-nc         # SPGISpeech 2.0 (Kensho, non-commercial) + TED-LIUM (NC-ND): PRIVATE storage only
mkdir -p $M $L
export HF_XET_HIGH_PERFORMANCE=1
LANE_A="restore_old ami_sdm librispeech earnings22 notsofar vitw_distortion vitw_dropout vitw_echo vitw_far_field"
LANE_B="spgi2 tedlium vitw_noise vitw_obstructed vitw_recording vitw_far_field_noise"

lane() {
  local name=$1; shift
  rm -f $L/$name.status
  {
    for s in "$@"; do
      [ -e $L/$s.built ] && { echo "skip $s (built)"; continue; }
      if [ $s = restore_old ]; then
        $P $B/fetch_mix.py --repo fhai50032/asr-english --mix $M --merge && touch $L/$s.built || echo "SOURCE FAILED $s"
      else
        $P $B/build_mix.py --out $M --only $s && touch $L/$s.built || echo "SOURCE FAILED $s"
      fi
    done
    echo 0 > $L/$name.status
  } >> $L/$name.log 2>&1
}

pusher() {
  rm -f $L/push.status
  {
    while :; do
      for f in $L/*.built; do
        [ -e "$f" ] || continue
        s=$(basename $f .built)
        [ -e $L/$s.pushed ] || [ -e $L/$s.pushfail ] && continue
        [ $s = restore_old ] && { touch $L/$s.pushed; continue; }   # already on HF (fhai50032/asr-english)
        case $s in spgi2|tedlium) repo=$NC; priv=--private;; *) repo=$OPEN; priv=;; esac
        echo "PUSH_START $s -> $repo"
        $P $B/push_src.py --mix $M --source $s --repo $repo $priv && touch $L/$s.pushed || { touch $L/$s.pushfail; echo "PUSH FAILED $s"; }
      done
      left=0
      for f in $L/*.built; do s=$(basename $f .built); [ -e $L/$s.pushed ] || [ -e $L/$s.pushfail ] || left=1; done
      [ -e $L/A.status ] && [ -e $L/B.status ] && [ $left = 0 ] && break
      sleep 60
    done
    echo 0 > $L/push.status
  } >> $L/push.log 2>&1
}

cd /home/marimo/work/BiBo && git pull -q && git log --oneline -1
lane A $LANE_A &
lane B $LANE_B &
pusher &
wait
echo PAR_EN_DONE
