#!/bin/bash
# Locked single-thread timing on the Mac (M1 Pro): one lock hold per (d, p) cell, each < ~1 min.
# pip Stim only: Stim has no NEON backend, so a native arm64 build is no faster (qec-r4 §1.5).
cd ~/qsim-fs
OUT=~/qsim-wt-logs/fs-timing-mac.jsonl
BIN=~/qsim-fs-target/release/examples/stim_compare
PY=~/qsim-bench-venv/bin/python
for p in 0.001 0.003; do
for spec in "3 4000000" "7 1000000" "11 256000" "15 128000"; do
  set -- $spec
  until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
  for i in $(seq 1 24); do   # wait (max 120 s) for 1-min load < 3
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk "BEGIN{exit !($l < 3.0)}" && break
    sleep 5
  done
  echo "{\"note\":\"d=$1 p=$p load1_at_start=$(sysctl -n vm.loadavg | awk '{print $2}')\"}" >> $OUT
  RAYON_NUM_THREADS=1 $PY research/data/fast-sampler/timing.py $BIN ~/qsim-wt-logs/fs-stim $1 $p $2 3 1 >> $OUT 2>&1
  rmdir /tmp/qsim-mac-bench.lock
  sleep 2
done
done
echo DONE >> $OUT
