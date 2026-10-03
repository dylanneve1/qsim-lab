#!/bin/bash
# Locked single-thread timing on the Mac: one lock hold per distance (< ~1 min each).
cd ~/qsim-qec-r4
OUT=~/qsim-wt-logs/qec-r4-timing.jsonl
BIN=~/qsim-qec-r4-target/release/examples/stim_compare
PY=~/qsim-bench-venv/bin/python
for spec in "3 2000000" "7 500000" "11 200000" "15 100000"; do
  set -- $spec
  until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
  # wait (max 120 s) for the 1-min load to fall below 3
  for i in $(seq 1 24); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk "BEGIN{exit !($l < 3.0)}" && break
    sleep 5
  done
  echo "{\"note\":\"d=$1 load1_at_start=$(sysctl -n vm.loadavg | awk '{print $2}')\"}" >> $OUT
  RAYON_NUM_THREADS=1 $PY research/data/qec-r4/stim_timing.py $BIN ~/qsim-wt-logs/qec-r4-stim $1 $2 3 >> $OUT 2>&1
  echo "{\"note\":\"d=$1 load1_at_end=$(sysctl -n vm.loadavg | awk '{print $2}')\"}" >> $OUT
  rmdir /tmp/qsim-mac-bench.lock
  sleep 2
done
echo DONE >> $OUT
