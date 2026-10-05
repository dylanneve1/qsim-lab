#!/bin/bash
cd ~/qsim-spoof
until grep -q DONE ~/qsim-wt-logs/spoof-campaign5.err 2>/dev/null; do sleep 30; done
OUT=~/qsim-wt-logs/spoof-patch.jsonl
export RAYON_NUM_THREADS=2
for th in 0.3 0.6 1.0; do
  until vm_stat | awk '/Pages free|Pages inactive/{s+=$NF} END{exit !(s*16384/1e9 >= 4)}'; do sleep 20; done
  nice -n 5 ./target/release/examples/spoof_patch 24 $th 20 1e-3,1e-4,3e-5,1e-5 >> $OUT 2>/dev/null &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ -d /tmp/qsim-mac-bench.lock ]; then kill -STOP $pid; while [ -d /tmp/qsim-mac-bench.lock ]; do sleep 5; done; kill -CONT $pid; fi
    sleep 3
  done
done
echo DONE >> ~/qsim-wt-logs/spoof-campaign6.err
