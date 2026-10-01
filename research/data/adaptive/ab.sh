#!/bin/bash
# Interleaved A/B: ab.sh REPS "METHODS (space-separated)" <qsim adaptive expect args...>
# Each (rep, method) is its own locked run, so load drifts hit all methods alike.
R=$1; shift; M=$1; shift
B=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh
Q=$(dirname "$0")/../../../target/release/qsim
for r in $(seq 1 "$R"); do
  for m in $M; do
    timeout 120 $B $Q adaptive expect --methods "$m" "$@"
  done
done
