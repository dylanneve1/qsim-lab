#!/bin/bash
cd ~/qsim-spoof
until grep -q DONE ~/qsim-wt-logs/spoof-campaign8.err 2>/dev/null; do sleep 30; done
OUT=~/qsim-wt-logs/spoof-campaign9.jsonl
LOG=~/qsim-wt-logs/spoof-campaign9.err
export RAYON_NUM_THREADS=2
memok() { vm_stat | awk '/Pages free|Pages inactive/{s+=$NF} END{exit !(s*16384/1e9 >= 4)}'; }
run() { # out bin args...
  local out=$1; shift
  until memok; do sleep 20; done
  local t0=$(date +%s)
  nice -n 5 "$@" >> $out 2>>$LOG &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then kill -STOP $pid; while [ -d /tmp/qsim-mac-bench.lock ]; do sleep 5; done; kill -CONT $pid; fi
    sleep 3
  done
  echo "$(date +%H:%M:%S) [$(( $(date +%s) - t0 ))s] $*" >> $LOG
}
for w in 8 9 10 11; do
  run $OUT ./target/release/examples/spoof_utility 4b --thetas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2,1.3 --delta 1e-5 --max-weight $w --mem-gb 2.2
done
for th in 0.3 0.5 0.6 0.7 0.8 1.0; do
  for w in 8 9 10 11 12; do
    run ~/qsim-wt-logs/spoof-patch-w.jsonl timeout 300 ./target/release/examples/spoof_patch 24 $th 20 1e-5 1 20 $w
  done
done
echo DONE >> $LOG
