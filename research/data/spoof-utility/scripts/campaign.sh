#!/bin/bash
# Background SPD campaign (2 rayon threads). Pauses while the swarm bench
# lock is held, waits for >= 4 GB free+inactive before each run.
cd ~/qsim-spoof
OUT=${OUT:-~/qsim-wt-logs/spoof-campaign.jsonl}
BIN=./target/release/examples/spoof_utility
TH="0.0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.5707"
export RAYON_NUM_THREADS=2
memok() { vm_stat | awk '/Pages free|Pages inactive/{s+=$NF} END{exit !(s*16384/1e9 >= 4)}'; }
run() { # fig delta extra-args...
  local fig=$1 d=$2; shift 2
  until memok; do sleep 20; done
  nice -n 5 $BIN $fig --thetas ${THETAS:-$TH} --delta $d --mem-gb 2.2 "$@" >> $OUT 2>>~/qsim-wt-logs/spoof-campaign.err &
  local pid=$!
  local t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    if [ -d /tmp/qsim-mac-bench.lock ] && [ ! -f /tmp/spoof-holds-lock ]; then
      kill -STOP $pid; while [ -d /tmp/qsim-mac-bench.lock ]; do sleep 5; done; kill -CONT $pid
    fi
    sleep 5
  done
  echo $(( $(date +%s) - t0 ))
}
for spec in ${SPECS:-"3b 3c 4a 3a 4b"}; do
  for d in ${DELTAS:-1e-3 1e-4 1e-5 1e-6 3e-7 1e-7}; do
    el=$(run $spec $d)
    echo "$(date +%H:%M:%S) fig $spec delta $d took ${el}s" >> ~/qsim-wt-logs/spoof-campaign.err
    if tail -30 $OUT | grep "\"fig\":\"$spec\"" | grep -q '"aborted":true'; then echo "abort at $d" >> ~/qsim-wt-logs/spoof-campaign.err; break; fi
    [ $el -gt ${MAXT:-1800} ] && break
  done
done
echo DONE >> ~/qsim-wt-logs/spoof-campaign.err
