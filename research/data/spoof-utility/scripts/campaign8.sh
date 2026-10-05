#!/bin/bash
cd ~/qsim-spoof
until grep -q DONE ~/qsim-wt-logs/spoof-campaign7.err 2>/dev/null; do sleep 30; done
OUT=~/qsim-wt-logs/spoof-campaign8.jsonl
LOG=~/qsim-wt-logs/spoof-campaign8.err
BIN=./target/release/examples/spoof_utility
export RAYON_NUM_THREADS=2
memok() { vm_stat | awk '/Pages free|Pages inactive/{s+=$NF} END{exit !(s*16384/1e9 >= 4)}'; }
run() {
  until memok; do sleep 20; done
  local t0=$(date +%s)
  nice -n 5 $BIN "$@" --mem-gb 2.2 >> $OUT 2>>$LOG &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then kill -STOP $pid; while [ -d /tmp/qsim-mac-bench.lock ]; do sleep 5; done; kill -CONT $pid; fi
    sleep 3
  done
  echo "$(date +%H:%M:%S) [$(( $(date +%s) - t0 ))s] $*" >> $LOG
}
run mz --lattice 1121 --thetas 0.3,0.7,1.0 --delta 1e-8,1e-9
run mz --lattice 433 --thetas 0.3,0.7,1.0 --delta 1e-8,1e-9
for s in 5 10 15 20 25 30; do
  run z559 --lattice 1121 --steps $s --thetas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4 --delta 1e-4,3e-5
done
echo DONE >> $LOG
