#!/bin/bash
# campaign2, trimmed (3e-6 only where the 1e-5 error matters).
cd ~/qsim-spoof
while pgrep -f "spoof_utility 3c --thetas 0.4" >/dev/null; do sleep 10; done
OUT=${OUT:-~/qsim-wt-logs/spoof-campaign2.jsonl}
LOG=~/qsim-wt-logs/spoof-campaign2.err
BIN=./target/release/examples/spoof_utility
export RAYON_NUM_THREADS=2
memok() { vm_stat | awk '/Pages free|Pages inactive/{s+=$NF} END{exit !(s*16384/1e9 >= 4)}'; }
run() {
  until memok; do sleep 20; done
  local t0=$(date +%s)
  nice -n 5 $BIN "$@" --mem-gb 2.2 >> $OUT 2>>$LOG &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then
      kill -STOP $pid; while [ -d /tmp/qsim-mac-bench.lock ]; do sleep 5; done; kill -CONT $pid
    fi
    sleep 3
  done
  echo "$(date +%H:%M:%S) [$(( $(date +%s) - t0 ))s] $*" >> $LOG
}
run 4a --thetas 0.6,0.8,0.9,1.0,1.1,1.2,1.25 --delta 1e-5 --stream 2
for s in 2 4 6 8 10 12 14 16 18 20; do
  run 4b --steps $s --thetas 0.6,0.8,1.0 --delta 1e-4,3e-5
done
run 4b --thetas 0.1,0.2,0.25,0.3,0.4,0.5 --delta 1e-5
run 3c --thetas 0.9,1.0,1.1 --delta 3e-6 --stream 2
run 4b --thetas 0.6 --delta 1e-5
echo DONE >> $LOG
