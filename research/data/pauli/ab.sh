#!/bin/bash
# Interleaved A/B: ab.sh <reps> <observable> <t> <engine>... ; each run is its own bench.sh lock.
B=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh
reps=$1; obs=$2; t=$3; shift 3
for i in $(seq 1 $reps); do
  for e in "$@"; do
    out=$($B ./target/release/qsim bench clifford-t --observable $obs --engine $e --min-t $t --max-t $t --step 1 2>&1)
    load=$(echo "$out" | grep -o "load at start: [0-9.]*" | cut -d' ' -f4)
    row=$(echo "$out" | grep "^| $t |")
    echo "rep=$i engine=$e load=$load ${RAYON_NUM_THREADS:+threads=$RAYON_NUM_THREADS} $row"
  done
done
