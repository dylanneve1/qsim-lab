#!/bin/bash
# Locked timing run: interleaved, 3 repetitions, 1 and 2 rayon threads.
cd ~/qsim-spoof
BIN=./target/release/examples/spoof_utility
OUT=~/qsim-wt-logs/spoof-bench.jsonl
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/qsim-mac-bench.lock' EXIT
echo "{\"load_at_start\":\"$(sysctl -n vm.loadavg)\",\"date\":\"$(date -u +%FT%TZ)\"}" >> $OUT
for rep in 1 2 3; do
  for thr in 1 2; do
    export RAYON_NUM_THREADS=$thr
    $BIN 3a --thetas 0.6 --delta 1e-8 2>/dev/null | sed "s/^{/{\"threads\":$thr,\"rep\":$rep,/" >> $OUT
    $BIN 3b --thetas 1.0 --delta 1e-6 2>/dev/null | sed "s/^{/{\"threads\":$thr,\"rep\":$rep,/" >> $OUT
    $BIN 4a --thetas 1.2 --delta 1e-4 2>/dev/null | sed "s/^{/{\"threads\":$thr,\"rep\":$rep,/" >> $OUT
    $BIN 4b --thetas 0.6 --delta 1e-4 2>/dev/null | sed "s/^{/{\"threads\":$thr,\"rep\":$rep,/" >> $OUT
    $BIN 4b --thetas 0.6 --delta 3e-5 2>/dev/null | sed "s/^{/{\"threads\":$thr,\"rep\":$rep,/" >> $OUT
  done
done
echo "{\"load_at_end\":\"$(sysctl -n vm.loadavg)\"}" >> $OUT
