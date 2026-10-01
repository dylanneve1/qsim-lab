#!/bin/bash
# Threshold sweep: one bench.sh lock per (p, d) point; each well under 2 minutes.
# Usage: research/data/qec/run_sweep.sh <out.csv> <shots> <seed> p1 p2 ...
set -e
B=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh
E=${E:-./target/release/examples/surface_threshold}
out=$1; shots=$2; seed=$3; shift 3
for p in "$@"; do
  for d in 3 5 7 9; do
    $B $E point $d $p $shots $seed dem circuit >> "$out"
  done
done
